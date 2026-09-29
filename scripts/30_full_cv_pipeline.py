#!/usr/bin/env python3
from __future__ import annotations

"""One-command compute-efficient vehicle-ReID research pipeline (v0.9.4).

The expensive outer K-fold training from v0.9.3 is removed.  A single deterministic,
identity/source-image-disjoint development train/validation split is used for all model and
hyperparameter decisions.  The immutable vehicle_reid_v5_official shared validation remains
reporting-only and is evaluated only after the recipe has been fixed and refit on all development
identities.

Stages
------
1. Reconstruct the exact shared V5 validation pool and one fixed development split.
2. Train the vendored vehicle_reid_v5_official BASE recipe once for DINOv3 ConvNeXt-Base and once
   for DINOv3 ViT-Base, both at 384x576.
3. Select the better global backbone on the common inner validation protocol.
4. Only for that winner, train Carparts multiscale spatial/semantic parts, part-aware ReID,
   hard-negative/detail tuning, listwise reranker and k-reciprocal retrieval.
5. Select the best stage on a dedicated identity-disjoint selection subset of inner validation;
   global-only is an explicit safety fallback.
6. Refit the chosen stage on all development identities for fixed epoch counts learned above.
7. Evaluate the exact four shared V5 query/gallery protocols and export the deployment artifacts.

All subprocess output is teed live to the terminal and persistent logs.  The pipeline is resumable.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.cv_experiments import checkpoint_official_metrics, fit_pooled_refusal, select_recipe_across_folds
from vehicle_fingerprint.official_validation import evaluate_recipe, generate_official_artifacts, run_official_script
from vehicle_fingerprint.models.backbone_registry import profile_config


def res_name(hw):
    return f"{int(hw[0])}x{int(hw[1])}"


def family(name):
    return "vit" if str(name).startswith("vit") else ("convnext" if str(name).startswith("convnext") else "other")


def dump_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    return path


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tee_chunk(chunk: bytes, fh) -> None:
    fh.write(chunk)
    fh.flush()
    try:
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    except AttributeError:
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()


def run_cmd(cmd, *, log, marker=None, resume=True, dry=False, keep_going=False):
    marker = Path(marker) if marker else None
    if resume and marker is not None and marker.exists():
        print(f"[SKIP] {marker}", flush=True)
        return True
    cmd = [str(x) for x in cmd]
    log = Path(log)
    print("[RUN]", " ".join(cmd), flush=True)
    print(f"[LOG] {log}", flush=True)
    if dry:
        return True
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log, "ab", buffering=0) as fh:
        fh.write((f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n$ {' '.join(cmd)}\n").encode("utf-8"))
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0)
        assert p.stdout is not None
        fd = p.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 8192)
            except InterruptedError:
                continue
            if not chunk:
                break
            _tee_chunk(chunk, fh)
        rc = p.wait()
    elapsed = time.time() - started
    if rc:
        print(f"\n[FAIL {rc}] elapsed={elapsed:.1f}s see {log}", flush=True)
        if keep_going:
            return False
        raise subprocess.CalledProcessError(rc, cmd)
    print(f"\n[DONE] elapsed={elapsed:.1f}s log={log}", flush=True)
    return True


def _capacity_factor(backbone, resolution):
    h, w = map(int, resolution)
    scale = (h * w) / (256 * 384)
    b = str(backbone).lower()
    if "vit_base" in b:
        mf = 1.55
    elif "convnext_base" in b:
        mf = 1.35
    elif "vit_large" in b:
        mf = 2.6
    elif "convnext_large" in b:
        mf = 2.0
    else:
        mf = 1.0
    return max(1.0, float(scale) * mf)


def adaptive_resources(c, backbone, resolution):
    d = c.setdefault("data", {})
    fac = _capacity_factor(backbone, resolution)
    if "p" in d and "k" in d:
        k = max(2, int(d.get("k", 4)))
        base = max(k, int(d.get("p", 16)) * k)
        micro = max(k * 4, int(round(base / fac / k)) * k)
        p = max(4, micro // k)
        d["p"], d["k"] = p, k
        c["effective_micro_batch"] = int(p * k)
    if "batch_size" in d:
        d["batch_size"] = max(4, int(round(float(d["batch_size"]) / fac)))
    if "eval_batch_size" in d:
        d["eval_batch_size"] = max(4, int(round(float(d["eval_batch_size"]) / fac)))
    return c


def part_cfg(template, *, run_dir, warmstart, resolution, pipe, backbone):
    c = read_yaml(ROOT / template)
    c["run_dir"] = str(run_dir)
    c["warmstart"] = str(warmstart)
    c["part_dataset"] = str(pipe["paths"]["part_bootstrap"])
    c["epochs"] = int(pipe["parts"]["epochs"])
    c["data"]["image_size"] = list(map(int, resolution))
    c["data"]["part_target_size"] = [max(32, int(resolution[0]) // 4), max(48, int(resolution[1]) // 4)]
    mo = c.setdefault("model_overrides", {})
    mo["multiscale_spatial"] = bool(pipe["parts"].get("multiscale_spatial", True))
    mo["spatial_fusion_dim"] = int(pipe["parts"].get("spatial_fusion_dim", 256))
    return adaptive_resources(c, backbone, resolution)


def reid_cfg(template, *, train, val, official, run_dir, warmstart, resolution, pipe, section, backbone, teacher=None, hard=None, no_eval=False, epochs_override=None):
    c = read_yaml(ROOT / template)
    sec = pipe[section]
    c["run_dir"] = str(run_dir)
    c["train_manifest"] = str(train)
    c["val_manifest"] = None if val is None else str(val)
    c["warmstart"] = str(warmstart)
    c["epochs"] = int(epochs_override if epochs_override is not None else sec["epochs"])
    c["data"]["image_size"] = list(map(int, resolution))
    c["data"]["part_target_size"] = [max(32, int(resolution[0]) // 4), max(48, int(resolution[1]) // 4)]
    c["data"]["return_weak_view"] = bool(sec.get("return_weak_view", False))
    c["model"]["multiscale_spatial"] = bool(sec.get("multiscale_spatial", True))
    c["model"]["spatial_fusion_dim"] = int(sec.get("spatial_fusion_dim", 256))
    c["model"].setdefault("backbone", {})["global_feature_mode"] = "v5_avg" if str(backbone).startswith("vit") else "timm_prelogits"
    c["teacher_checkpoint"] = str(teacher) if teacher else None
    c["hard_negative_map"] = str(hard) if hard else None
    l = c.setdefault("loss", {})
    l["cross_camera_triplet"] = bool(sec.get("cross_camera_triplet", True))
    l["cross_camera_positive_weight"] = float(sec.get("cross_camera_positive_weight", .8))
    l["same_camera_positive_weight"] = float(sec.get("same_camera_positive_weight", .2))
    l["memory_bank_size"] = int(sec.get("memory_bank_size", 0))
    l["memory_triplet_weight"] = float(sec.get("memory_triplet_weight", 0))
    for k in ("global_supcon", "ema_part_presence", "ema_teacher_decay", "ema_presence_topk"):
        if k in sec:
            l[k] = sec[k]
    c["online_hard_mining_every"] = int(sec.get("online_hard_mining_every", 0))
    c["online_hard_topk"] = int(sec.get("online_hard_topk", 40))
    if no_eval:
        c["evaluation"]["official"] = {}
    else:
        c["evaluation"]["official"] = {
            "evaluator": "official/evaluate.py",
            "gt": str(Path(official) / "ground_truth.csv"),
            "query_manifest": str(Path(official) / "query.csv"),
            "gallery_manifest": str(Path(official) / "gallery.csv"),
            "top_k": 10,
        }
    return adaptive_resources(c, backbone, resolution)


def extract_cmd(manifest, ckpt, out, pipe):
    f = pipe["features"]
    return [
        sys.executable, "scripts/11_extract_features.py",
        "--manifest", str(manifest), "--checkpoint", str(ckpt), "--out", str(out),
        "--device", str(pipe["device"]), "--precision", str(pipe["precision"]),
        "--batch", str(f.get("batch", 24)), "--workers", str(f.get("workers", 8)),
    ]


def ensure_data(pipe, args):
    p = pipe["paths"]
    hack = Path(p["hackathon"])
    cv = Path(p["cv"])
    log = Path(p["runs"]) / "logs" / "prepare.log"
    run_cmd([
        sys.executable, "scripts/00_prepare_hackathon.py", "--csv", p["raw_csv"], "--images", p["raw_images"],
        "--out", p["hackathon"], "--pad", "0.03", "--val-fraction", "0", "--eval-fraction", "0",
        "--min-eval-cameras", str(pipe["cv"]["min_eval_cameras"]), "--seed", str(pipe["seed"]), "--no-official-protocols",
    ], log=log, marker=hack / "manifest.csv", resume=args.resume, dry=args.dry, keep_going=args.keep_going)
    split_cmd = [
        sys.executable, "scripts/25_prepare_single_dev_split.py",
        "--manifest", str(hack / "manifest.csv"), "--out", p["cv"],
        "--fixed-validation-spec", str(pipe["cv"]["shared_validation_spec"]),
        "--inner-val-fraction", str(pipe["cv"].get("inner_val_fraction", .25)),
        "--reranker-fit-fraction-of-val", str(pipe["cv"].get("reranker_fit_fraction_of_val", .50)),
        "--seed", str(pipe["seed"]), "--min-eval-cameras", str(pipe["cv"]["min_eval_cameras"]),
        "--open-set-fraction", str(pipe["cv"]["open_set_fraction"]),
        "--max-queries-per-id", str(pipe["cv"]["max_queries_per_id"]),
    ]
    if not pipe["cv"].get("compute_brightness", True):
        split_cmd.append("--no-brightness")
    run_cmd(split_cmd, log=log, marker=cv / "single_split_summary.json", resume=args.resume, dry=args.dry, keep_going=args.keep_going)

    vp = pipe.get("veri_pretrain", {})
    if bool(vp.get("enabled", False)) and bool(vp.get("prepare_if_needed", True)):
        run_cmd([
            sys.executable, "scripts/09_prepare_external_reid.py",
            "--root", str(Path(p["veri"]).parent),
            "--veri-root", str(p["veri"]),
            "--out-dir", str(p["external_processed"]),
        ], log=log, marker=Path(vp.get("manifest_audit", Path(p["external_processed"]) / "veri_v5_transfer_manifest.json")),
           resume=args.resume, dry=args.dry, keep_going=args.keep_going)

    if Path(p["carparts"]).exists():
        run_cmd([
            sys.executable, "scripts/04_build_part_dataset.py", "--carparts", p["carparts"], "--out", p["part_bootstrap"]
        ], log=log, marker=Path(p["part_bootstrap"]) / "dataset.yaml", resume=args.resume, dry=args.dry, keep_going=args.keep_going)


def _v5_exact_model_name(backbone: str) -> str:
    return str(profile_config(str(backbone))["model_name"])


def _v5_exact_microbatch(pipe: dict, backbone: str) -> int:
    mb = pipe.get("baseline", {}).get("train_microbatch", 4)
    if isinstance(mb, dict):
        return int(mb.get(str(backbone), mb.get("default", 4)))
    return int(mb)


def _optional_checkpoint(value):
    """Normalize nullable checkpoint values crossing pandas/JSON boundaries.

    Direct-training rows intentionally store ``None`` for ``init_checkpoint``.  Pandas
    promotes that column to NaN when transfer rows contain strings, and ``float('nan')``
    is truthy.  Without normalization the final refit incorrectly selects the transfer
    entrypoint and literally executes ``--init-checkpoint nan``.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return None
    return text


def _run_v5_exact_baseline(pipe, args, *, backbone, train_manifest, val_manifest, run_dir, epochs=None, final_refit=False, init_checkpoint=None, images_dir=None, plate_mask_prob=None, profile=None):
    init_checkpoint = _optional_checkpoint(init_checkpoint)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    h, w = map(int, pipe["resolutions"][0])
    native_name = "last.pt" if final_refit else "best.pt"
    project_name = "project_last.pt" if final_refit else "project_best.pt"
    # Transfer warm-starts use a dedicated entrypoint.  The generic v0.9.4 wrapper is
    # intentionally not trusted for transfer because old project snapshots may overwrite
    # it while a long multi-stage run is in progress.
    entrypoint = "scripts/10b_train_baseline_v5_transfer.py" if init_checkpoint else "scripts/10_train_baseline_v5_exact.py"
    if init_checkpoint:
        _assert_veri_transfer_plumbing(check_generic_cli=False)
    cmd = [
        sys.executable, entrypoint,
        "--train-manifest", str(train_manifest), "--images-dir", str(images_dir or pipe["paths"]["raw_images"]),
        "--out", str(run_dir), "--model", _v5_exact_model_name(backbone),
        "--profile", str(profile or pipe["baseline"].get("profile", "base")),
        "--height", str(h), "--width", str(w), "--device", str(pipe["device"]),
        "--precision", str(pipe["precision"]), "--train-microbatch", str(_v5_exact_microbatch(pipe, backbone)),
    ]
    if val_manifest is not None:
        cmd += ["--val-manifest", str(val_manifest)]
    if init_checkpoint:
        cmd += ["--init-checkpoint", str(init_checkpoint)]
    if plate_mask_prob is not None:
        cmd += ["--plate-mask-prob", str(float(plate_mask_prob))]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    if final_refit:
        cmd.append("--final-refit")
    ok = run_cmd(cmd, log=run_dir / "train.log", marker=run_dir / native_name, resume=args.resume, dry=args.dry, keep_going=args.keep_going)
    if not ok:
        return False, run_dir / project_name
    conv = [sys.executable, "scripts/10c_convert_v5_checkpoint.py", "--src", str(run_dir / native_name), "--out", str(run_dir / project_name)]
    ok = run_cmd(conv, log=run_dir / "convert.log", marker=run_dir / project_name, resume=args.resume, dry=args.dry, keep_going=args.keep_going)
    return ok, run_dir / project_name


def _selection_key(m: dict):
    return (float(m["mAP@10"]), float(m["Rank-1"]), float(m["Rank-5"]), float(m["mAP_full"]), float(m["mINP"]))


def _veri_transfer_meta(pipe):
    vp = pipe.get("veri_pretrain", {})
    meta_path = Path(vp.get("manifest_audit", Path(pipe["paths"]["external_processed"]) / "veri_v5_transfer_manifest.json"))
    if not meta_path.exists():
        raise FileNotFoundError(
            f"VeRi transfer manifest audit not found: {meta_path}. "
            "Extract the Kaggle VeRi-776 dataset under data/external/VeRi and rerun."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _run_veri_pretrain(pipe, args, backbone):
    vp = pipe.get("veri_pretrain", {})
    if not bool(vp.get("enabled", False)):
        return None
    meta = None if args.dry else _veri_transfer_meta(pipe)
    images_root = str(meta["resolved_root"]) if meta else str(pipe["paths"]["veri"])
    root = Path(pipe["paths"]["runs"]) / "veri_pretrain_v5_exact" / backbone / "384x576"
    ok, project_ckpt = _run_v5_exact_baseline(
        pipe, args, backbone=backbone,
        train_manifest=vp.get("train_manifest", Path(pipe["paths"]["external_processed"]) / "veri_v5_train.csv"),
        val_manifest=vp.get("val_manifest", Path(pipe["paths"]["external_processed"]) / "veri_v5_val.csv"),
        run_dir=root, epochs=int(vp.get("epochs", 48)), final_refit=False,
        images_dir=images_root, plate_mask_prob=float(vp.get("plate_mask_prob", 0.0)),
        profile=str(vp.get("profile", pipe["baseline"].get("profile", "base"))),
    )
    if not ok:
        return None
    return {
        "native_checkpoint": str(root / "best.pt"),
        "project_checkpoint": str(project_ckpt),
        "run_dir": str(root),
        "resolved_root": images_root,
    }


def _assert_veri_transfer_plumbing(*, check_generic_cli=True):
    """Fail before transfer if the local tree mixes old and VeRi-aware files.

    The dedicated transfer CLI is checked on every warm-start launch, so a long-running
    pipeline cannot silently reach an old generic wrapper hours later.
    """
    import inspect
    from vehicle_fingerprint.baseline_v5_exact import train_v5_exact_presplit

    params = inspect.signature(train_v5_exact_presplit).parameters
    missing = [x for x in ("init_checkpoint", "plate_mask_prob") if x not in params]
    transfer_cli = ROOT / "scripts" / "10b_train_baseline_v5_transfer.py"
    transfer_text = transfer_cli.read_text(encoding="utf-8") if transfer_cli.exists() else ""
    for opt in ("--init-checkpoint", "--plate-mask-prob"):
        if opt not in transfer_text:
            missing.append(f"scripts/10b_train_baseline_v5_transfer.py:{opt}")
    if check_generic_cli:
        cli_path = ROOT / "scripts" / "10_train_baseline_v5_exact.py"
        cli_text = cli_path.read_text(encoding="utf-8") if cli_path.exists() else ""
        # Generic CLI compatibility is useful but no longer required for transfer execution.
        if "--init-checkpoint" not in cli_text:
            print("[WARN] generic V5 CLI lacks --init-checkpoint; dedicated transfer CLI will be used")
    if missing:
        raise RuntimeError(
            "VeRi transfer plumbing is incomplete; the project appears to mix original v0.9.4 "
            "and VeRi-transfer files. Missing: " + ", ".join(missing)
        )


def global_baseline_compare(pipe, args):
    root = Path(pipe["paths"]["runs"])
    cv = Path(pipe["paths"]["cv"])
    inner = cv / "inner"
    if pipe.get("backbones") != ["convnext_base", "vit_base"]:
        raise ValueError("v0.9.4 expects exactly backbones: [convnext_base, vit_base]")
    if pipe.get("resolutions") != [[384, 576]]:
        raise ValueError("v0.9.4 expects exactly resolutions: [[384,576]]")
    vp = pipe.get("veri_pretrain", {})
    use_veri = bool(vp.get("enabled", False))
    if use_veri:
        _assert_veri_transfer_plumbing(check_generic_cli=True)
    run_direct = bool(vp.get("run_direct_target_control", True)) if use_veri else True
    rows = []
    for b in pipe["backbones"]:
        veri = _run_veri_pretrain(pipe, args, b) if use_veri else None
        if run_direct:
            rd = root / "global_v5_exact_direct" / b / "384x576"
            ok, project_ckpt = _run_v5_exact_baseline(pipe, args, backbone=b, train_manifest=inner / "train.csv", val_manifest=inner / "val.csv", run_dir=rd)
            if ok and not args.dry and project_ckpt.exists():
                m = checkpoint_official_metrics(project_ckpt)
                rows.append({"backbone": b, "family": family(b), "resolution": "384x576", "initialization": "dinov3_direct", "init_checkpoint": None, "veri_pretrain_checkpoint": None, **m, "checkpoint": str(project_ckpt), "native_checkpoint": str(rd / "best.pt")})
        if use_veri and veri is not None:
            rd = root / "global_v5_exact_from_veri" / b / "384x576"
            ok, project_ckpt = _run_v5_exact_baseline(pipe, args, backbone=b, train_manifest=inner / "train.csv", val_manifest=inner / "val.csv", run_dir=rd, init_checkpoint=veri["native_checkpoint"])
            if ok and not args.dry and project_ckpt.exists():
                m = checkpoint_official_metrics(project_ckpt)
                rows.append({"backbone": b, "family": family(b), "resolution": "384x576", "initialization": "veri776_v5", "init_checkpoint": veri["native_checkpoint"], "veri_pretrain_checkpoint": veri["project_checkpoint"], **m, "checkpoint": str(project_ckpt), "native_checkpoint": str(rd / "best.pt")})
    if args.dry:
        return None
    if not rows:
        raise RuntimeError("No global baseline completed")
    tab = pd.DataFrame(rows).sort_values(["mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP"], ascending=False, kind="stable")
    tab.to_csv(root / "global_v5_exact_comparison.csv", index=False)
    winner = tab.iloc[0].to_dict()
    for key in ("init_checkpoint", "veri_pretrain_checkpoint"):
        winner[key] = _optional_checkpoint(winner.get(key))
    (root / "global_winner.json").write_text(json.dumps(winner, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("\nGLOBAL V5-EXACT BASE + VeRi TRANSFER COMPARISON\n", tab.to_string(index=False))
    print(f"\nGLOBAL WINNER: {winner['backbone']} init={winner.get('initialization')} mAP@10={winner['mAP@10']:.6f}")
    return winner


def _stage_eval(pipe, args, *, ckpt, stage, backbone, selection_dir, root, allow_parts):
    sd = root / "stage_eval" / stage
    q = sd / "query.npz"
    g = sd / "gallery.npz"
    run_cmd(extract_cmd(selection_dir / "query.csv", ckpt, q, pipe), log=sd / "features.log", marker=q, resume=args.resume, dry=False, keep_going=args.keep_going)
    run_cmd(extract_cmd(selection_dir / "gallery.csv", ckpt, g, pipe), log=sd / "features.log", marker=g, resume=args.resume, dry=False, keep_going=args.keep_going)
    rr = pipe["reranker"]
    alphas = rr["alpha_grid"] if allow_parts else [0.0]
    desc = [{"query_cache": str(q), "gallery_cache": str(g), "gt": str(selection_dir / "ground_truth.csv")}]
    recipe, search = select_recipe_across_folds(
        desc, alpha_grid=alphas, beta_grid=[0.0], kreciprocal_grid=[0.0], same_camera_grid=[False],
        rerank_topk=rr["rerank_topk"], kreciprocal_k=rr["kreciprocal_k"], device="cuda"
    )
    search.to_csv(sd / "recipe_search.csv", index=False)
    rep, *_ = evaluate_recipe(q, g, selection_dir / "ground_truth.csv", base_alpha=recipe["base_alpha"], device="cuda", evaluator_path="official/evaluate.py")
    r, fr = rep["ranking"], rep["full_ranking"]
    return {
        "stage": stage, "checkpoint": str(ckpt), "query_cache": str(q), "gallery_cache": str(g), "recipe": recipe,
        "mAP@10": float(r["mAP@10"]), "Rank-1": float(r["Rank-1"]), "Rank-5": float(r["Rank-5"]),
        "mAP_full": float(fr["mAP_full"]), "mINP": float(fr["mINP"]),
    }


def advanced_single(pipe, args, global_sel):
    root = Path(pipe["paths"]["runs"])
    cv = Path(pipe["paths"]["cv"])
    inner = cv / "inner"
    cfgdir = root / "generated_configs" / "advanced"
    b = str(global_sel["backbone"])
    fam = family(b)
    rn = "384x576"
    res = [384, 576]
    aroot = root / "advanced" / fam / f"{b}_{rn}"
    base = Path(global_sel["checkpoint"])

    pr = aroot / "parts"
    pc = part_cfg(pipe["parts"]["template"], run_dir=pr, warmstart=base, resolution=res, pipe=pipe, backbone=b)
    pcp = dump_yaml(pc, cfgdir / "parts.yaml")
    run_cmd([sys.executable, "scripts/05_pretrain_dino_part_head.py", str(pcp), "--run-dir", str(pr), "--warmstart", str(base), "--expect-backbone", b], log=pr / "train.log", marker=pr / "best.pt", resume=args.resume, dry=args.dry, keep_going=args.keep_going)

    pa = aroot / "part_aware"
    rc = reid_cfg(pipe["part_aware"]["template"], train=inner / "train.csv", val=inner / "val.csv", official=inner / "official_val", run_dir=pa, warmstart=pr / "best.pt", resolution=res, pipe=pipe, section="part_aware", backbone=b)
    rcp = dump_yaml(rc, cfgdir / "partaware.yaml")
    run_cmd([sys.executable, "scripts/10_train_reid.py", str(rcp), "--backbone", b, "--run-dir", str(pa), "--warmstart", str(pr / "best.pt"), "--hard-negative-map", "none"], log=pa / "train.log", marker=pa / "best.pt", resume=args.resume, dry=args.dry, keep_going=args.keep_going)

    traincache = pa / "train_features.npz"
    run_cmd(extract_cmd(inner / "train.csv", pa / "best.pt", traincache, pipe), log=pa / "features.log", marker=traincache, resume=args.resume, dry=args.dry, keep_going=args.keep_going)
    hard = pa / "hard_negatives.json"
    run_cmd([sys.executable, "scripts/12_mine_hard_negatives.py", "--cache", str(traincache), "--out", str(hard), "--topk", "40", "--representation", "blend", "--alpha", "0.15", "--refine-factor", "4", "--top-pair-mean", "3"], log=pa / "mine.log", marker=hard, resume=args.resume, dry=args.dry, keep_going=args.keep_going)

    dt = aroot / "detail"
    dc = reid_cfg(pipe["detail"]["template"], train=inner / "train.csv", val=inner / "val.csv", official=inner / "official_val", run_dir=dt, warmstart=pa / "best.pt", resolution=res, pipe=pipe, section="detail", backbone=b, teacher=pr / "best.pt", hard=hard)
    dcp = dump_yaml(dc, cfgdir / "detail.yaml")
    run_cmd([sys.executable, "scripts/10_train_reid.py", str(dcp), "--backbone", b, "--run-dir", str(dt), "--warmstart", str(pa / "best.pt"), "--teacher-checkpoint", str(pr / "best.pt"), "--hard-negative-map", str(hard)], log=dt / "train.log", marker=dt / "best.pt", resume=args.resume, dry=args.dry, keep_going=args.keep_going)

    if args.dry:
        return {"family": fam, "backbone": b, "resolution": rn, "root": str(aroot)}

    # Fit the listwise reranker only on identities reserved for reranker fitting.
    rrfit = aroot / "reranker_fit"
    rr_cache = rrfit / "features.npz"
    run_cmd(extract_cmd(inner / "reranker_fit.csv", dt / "best.pt", rr_cache, pipe), log=rrfit / "features.log", marker=rr_cache, resume=args.resume, dry=False, keep_going=args.keep_going)
    rr_hard = rrfit / "hard.json"
    run_cmd([sys.executable, "scripts/12_mine_hard_negatives.py", "--cache", str(rr_cache), "--out", str(rr_hard), "--topk", "30", "--representation", "global", "--refine-factor", "3", "--top-pair-mean", "2"], log=rrfit / "mine.log", marker=rr_hard, resume=args.resume, dry=False, keep_going=args.keep_going)
    rr_pairs = rrfit / "pairs.npz"
    rr = pipe["reranker"]
    run_cmd([sys.executable, "scripts/28_build_reranker_pairs.py", "--cache", str(rr_cache), "--hard-map", str(rr_hard), "--out", str(rr_pairs), "--mode", rr["mode"], "--candidates-per-query", str(rr["candidates_per_query"]), "--groups-per-id", str(rr["groups_per_id"]), "--knn-k", str(rr["knn_k"])], log=rrfit / "pairs.log", marker=rr_pairs, resume=args.resume, dry=False, keep_going=args.keep_going)
    rr_model = aroot / "reranker" / "best.pt"
    run_cmd([sys.executable, "scripts/29_train_reranker_from_pairs.py", "--pairs", str(rr_pairs), "--out", str(rr_model), "--epochs", str(rr["epochs"]), "--device", "cuda", "--mode", rr["mode"]], log=aroot / "reranker" / "train.log", marker=rr_model, resume=args.resume, dry=False, keep_going=args.keep_going)

    selection = inner / "selection_official"
    # Evaluate all learned stages on exactly the same selection protocol.  This preserves a true
    # global-only fallback if spatial/detail training happens to hurt.
    stages = [
        _stage_eval(pipe, args, ckpt=base, stage="v5_exact_global", backbone=b, selection_dir=selection, root=aroot, allow_parts=False),
        _stage_eval(pipe, args, ckpt=pa / "best.pt", stage="spatial_part_aware", backbone=b, selection_dir=selection, root=aroot, allow_parts=True),
        _stage_eval(pipe, args, ckpt=dt / "best.pt", stage="detail_tuning", backbone=b, selection_dir=selection, root=aroot, allow_parts=True),
    ]

    detail_desc = [{
        "query_cache": stages[-1]["query_cache"], "gallery_cache": stages[-1]["gallery_cache"],
        "gt": str(selection / "ground_truth.csv"), "reranker": str(rr_model),
    }]
    final_recipe, search = select_recipe_across_folds(
        detail_desc, alpha_grid=rr["alpha_grid"], beta_grid=rr["beta_grid"],
        kreciprocal_grid=rr["kreciprocal_lambda_grid"], same_camera_grid=rr["same_camera_filter_grid"],
        rerank_topk=rr["rerank_topk"], kreciprocal_k=rr["kreciprocal_k"], device="cuda"
    )
    search.to_csv(aroot / "retrieval_recipe_search.csv", index=False)
    rep, *_ = evaluate_recipe(
        stages[-1]["query_cache"], stages[-1]["gallery_cache"], selection / "ground_truth.csv",
        base_alpha=final_recipe["base_alpha"], reranker_path=rr_model,
        reranker_beta=final_recipe["reranker_beta"], rerank_topk=final_recipe["rerank_topk"],
        kreciprocal_lambda=final_recipe["kreciprocal_lambda"], kreciprocal_k=final_recipe["kreciprocal_k"],
        same_camera_filter=final_recipe["same_camera_filter"], device="cuda", evaluator_path="official/evaluate.py"
    )
    r, fr = rep["ranking"], rep["full_ranking"]
    stages.append({
        "stage": "listwise_reranker_kreciprocal", "checkpoint": str(dt / "best.pt"),
        "query_cache": stages[-1]["query_cache"], "gallery_cache": stages[-1]["gallery_cache"],
        "recipe": final_recipe, "reranker": str(rr_model),
        "mAP@10": float(r["mAP@10"]), "Rank-1": float(r["Rank-1"]), "Rank-5": float(r["Rank-5"]),
        "mAP_full": float(fr["mAP_full"]), "mINP": float(fr["mINP"]),
    })

    stage_tab = pd.DataFrame([{k: x.get(k) for k in ("stage", "mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP")} for x in stages])
    stage_tab.to_csv(aroot / "stage_ablation.csv", index=False)
    chosen = max(stages, key=lambda x: (x["mAP@10"], x["Rank-1"], x["Rank-5"], x["mAP_full"], x["mINP"]))
    chosen_desc = [{"query_cache": chosen["query_cache"], "gallery_cache": chosen["gallery_cache"], "gt": str(selection / "ground_truth.csv")}]
    chosen_reranker = chosen.get("reranker")
    if chosen_reranker:
        chosen_desc[0]["reranker"] = chosen_reranker
    refusal_path = aroot / "refusal_selection.json"
    fit_pooled_refusal(chosen_desc, chosen["recipe"], refusal_path, device="cuda")

    pa_epoch = checkpoint_official_metrics(pa / "best.pt")["epoch"]
    dt_epoch = checkpoint_official_metrics(dt / "best.pt")["epoch"]
    summary = {
        "family": fam, "backbone": b, "resolution": rn, "source": "v5_exact_base_single_split",
        "initialization": global_sel.get("initialization", "dinov3_direct"),
        "init_checkpoint": _optional_checkpoint(global_sel.get("init_checkpoint")),
        "veri_pretrain_checkpoint": _optional_checkpoint(global_sel.get("veri_pretrain_checkpoint")),
        "global_best_epoch": int(global_sel["epoch"]), "part_aware_best_epoch": int(pa_epoch), "detail_best_epoch": int(dt_epoch),
        "selected_stage": chosen["stage"], "mAP@10": chosen["mAP@10"], "Rank-1": chosen["Rank-1"], "Rank-5": chosen["Rank-5"],
        "mAP_full": chosen["mAP_full"], "mINP": chosen["mINP"], "recipe": chosen["recipe"],
        "reranker": chosen_reranker, "refusal": str(refusal_path), "stages": stages, "root": str(aroot),
        "global_checkpoint": str(base), "part_checkpoint": str(pr / "best.pt"), "part_aware_checkpoint": str(pa / "best.pt"), "detail_checkpoint": str(dt / "best.pt"),
    }
    (aroot / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nADVANCED STAGE ABLATION\n", stage_tab.to_string(index=False))
    print(f"\nSELECTED STAGE: {chosen['stage']} mAP@10={chosen['mAP@10']:.6f}")
    return summary


def fit_final(pipe, args, summary):
    root = Path(pipe["paths"]["runs"]) / "final_fit"
    cv = Path(pipe["paths"]["cv"])
    b = summary["backbone"]
    res = [384, 576]
    cfgdir = Path(pipe["paths"]["runs"]) / "generated_configs" / "final"

    initialization = str(summary.get("initialization", "dinov3_direct"))
    init_checkpoint = _optional_checkpoint(summary.get("init_checkpoint"))
    if initialization == "dinov3_direct":
        init_checkpoint = None
    elif initialization == "veri776_v5" and init_checkpoint is None:
        raise RuntimeError("Final refit selected VeRi initialization but no valid init_checkpoint is available")

    _ok, base = _run_v5_exact_baseline(
        pipe, args, backbone=b, train_manifest=cv / "dev.csv", val_manifest=None,
        run_dir=root / "baseline_v5_exact", epochs=int(summary["global_best_epoch"]), final_refit=True,
        init_checkpoint=init_checkpoint
    )
    stage = summary["selected_stage"]
    if stage == "v5_exact_global":
        return {"checkpoint": str(base), "stage": stage, "backbone": b, "resolution": "384x576"}

    pr = root / "parts"
    pc = part_cfg(pipe["parts"]["template"], run_dir=pr, warmstart=base, resolution=res, pipe=pipe, backbone=b)
    pcp = dump_yaml(pc, cfgdir / "parts.yaml")
    run_cmd([sys.executable, "scripts/05_pretrain_dino_part_head.py", str(pcp), "--run-dir", str(pr), "--warmstart", str(base), "--expect-backbone", b], log=pr / "train.log", marker=pr / "best.pt", resume=args.resume, dry=False, keep_going=args.keep_going)

    pa = root / "part_aware"
    pac = reid_cfg(pipe["part_aware"]["template"], train=cv / "dev.csv", val=None, official=None, run_dir=pa, warmstart=pr / "best.pt", resolution=res, pipe=pipe, section="part_aware", backbone=b, no_eval=True, epochs_override=int(summary["part_aware_best_epoch"]))
    pacp = dump_yaml(pac, cfgdir / "partaware.yaml")
    run_cmd([sys.executable, "scripts/10_train_reid.py", str(pacp), "--backbone", b, "--run-dir", str(pa), "--warmstart", str(pr / "best.pt"), "--hard-negative-map", "none"], log=pa / "train.log", marker=pa / "last.pt", resume=args.resume, dry=False, keep_going=args.keep_going)
    if stage == "spatial_part_aware":
        return {"checkpoint": str(pa / "last.pt"), "stage": stage, "backbone": b, "resolution": "384x576"}

    cache = pa / "dev_features.npz"
    run_cmd(extract_cmd(cv / "dev.csv", pa / "last.pt", cache, pipe), log=pa / "features.log", marker=cache, resume=args.resume, dry=False, keep_going=args.keep_going)
    hard = pa / "hard.json"
    run_cmd([sys.executable, "scripts/12_mine_hard_negatives.py", "--cache", str(cache), "--out", str(hard), "--topk", "40", "--representation", "blend", "--alpha", "0.15"], log=pa / "mine.log", marker=hard, resume=args.resume, dry=False, keep_going=args.keep_going)

    dt = root / "detail"
    dc = reid_cfg(pipe["detail"]["template"], train=cv / "dev.csv", val=None, official=None, run_dir=dt, warmstart=pa / "last.pt", resolution=res, pipe=pipe, section="detail", backbone=b, teacher=pr / "best.pt", hard=hard, no_eval=True, epochs_override=int(summary["detail_best_epoch"]))
    dcp = dump_yaml(dc, cfgdir / "detail.yaml")
    run_cmd([sys.executable, "scripts/10_train_reid.py", str(dcp), "--backbone", b, "--run-dir", str(dt), "--warmstart", str(pa / "last.pt"), "--teacher-checkpoint", str(pr / "best.pt"), "--hard-negative-map", str(hard)], log=dt / "train.log", marker=dt / "last.pt", resume=args.resume, dry=False, keep_going=args.keep_going)
    return {"checkpoint": str(dt / "last.pt"), "stage": stage, "backbone": b, "resolution": "384x576"}


def _shared_validation_fold_paths(pipe):
    root = Path(pipe["paths"]["cv"]) / "shared_validation"
    spec_path = ROOT / pipe["cv"]["shared_validation_spec"] if not Path(pipe["cv"]["shared_validation_spec"]).is_absolute() else Path(pipe["cv"]["shared_validation_spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return [root / f"fold_{i}" for i in range(int(spec["official_protocol"]["folds"]))]


def _aggregate_official_reports(reports):
    rows = []
    for fold, rep in enumerate(reports):
        r, fr, c = rep.get("ranking", {}), rep.get("full_ranking", {}), rep.get("candidates", {})
        rows.append({
            "fold": fold, "mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"), "Rank-5": r.get("Rank-5"),
            "mAP_full": fr.get("mAP_full"), "mINP": fr.get("mINP"), "Precision": c.get("Precision"),
            "Recall": c.get("Recall"), "F1": c.get("F1"), "TNR": c.get("TNR"), "PR-AUC": c.get("PR-AUC"),
            "ranking_queries": r.get("n_scored"), "openset_queries": r.get("n_openset_excluded"),
        })
    df = pd.DataFrame(rows)
    summary = {"folds": len(rows), "per_fold": rows}
    for m in ["mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP", "Precision", "Recall", "F1", "TNR", "PR-AUC"]:
        vals = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy(float)
        summary[m + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
        summary[m + "_std"] = float(vals.std()) if len(vals) else float("nan")
    return df, summary


def export_deployment(pipe, args, summary, fit, shared_dir):
    deploy = Path(pipe["paths"]["deploy"])
    deploy.mkdir(parents=True, exist_ok=True)
    dst = deploy / "reid.pt"
    run_cmd([sys.executable, "scripts/16_export_inference_checkpoint.py", "--src", fit["checkpoint"], "--out", str(dst)], log=deploy / "export.log", marker=dst, resume=args.resume, dry=False, keep_going=args.keep_going)
    reranker_dst = None
    if summary.get("reranker"):
        reranker_dst = deploy / "reranker.pt"
        shutil.copy2(summary["reranker"], reranker_dst)
    shutil.copy2(summary["refusal"], deploy / "refusal.json")
    (deploy / "retrieval_recipe.json").write_text(json.dumps(summary["recipe"], ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "mode": "single", "backbone": summary["backbone"], "resolution": summary["resolution"],
        "selected_stage": summary["selected_stage"], "checkpoint": "reid.pt",
        "initialization": summary.get("initialization", "dinov3_direct"),
        "reranker": "reranker.pt" if reranker_dst else None, "refusal": "refusal.json",
        "retrieval_recipe": "retrieval_recipe.json", "shared_validation_report": str(Path(shared_dir) / "shared_validation_report.json"),
        "inference_script": "scripts/21_infer_from_csv.py",
    }
    (deploy / "deployment.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DEPLOY] {deploy}")


def final_shared_validation(pipe, args, summary):
    out = Path(pipe["paths"]["runs"]) / "final_shared_validation"
    out.mkdir(parents=True, exist_ok=True)
    fit = fit_final(pipe, args, summary)
    recipe = summary["recipe"]
    refusal = json.loads(Path(summary["refusal"]).read_text(encoding="utf-8"))
    reranker = summary.get("reranker")
    reports = []
    for i, fd in enumerate(_shared_validation_fold_paths(pipe)):
        fo = out / f"fold_{i}"
        fo.mkdir(parents=True, exist_ok=True)
        q, g = fo / "query.npz", fo / "gallery.npz"
        run_cmd(extract_cmd(fd / "query.csv", fit["checkpoint"], q, pipe), log=fo / "features.log", marker=q, resume=args.resume, dry=False, keep_going=args.keep_going)
        run_cmd(extract_cmd(fd / "gallery.csv", fit["checkpoint"], g, pipe), log=fo / "features.log", marker=g, resume=args.resume, dry=False, keep_going=args.keep_going)
        gt = fd / "ground_truth.csv"
        generate_official_artifacts(q, g, gt, fo, recipe, refusal, reranker_path=reranker, device="cuda", evaluator_path="official/evaluate.py")
        rep = run_official_script(gt, fo / "submission.csv", candidates=fo / "candidates.csv", embeddings=fo / "embeddings.npy", query_csv=fd / "query.csv", gallery_csv=fd / "gallery.csv", json_out=fo / "official_report.json", evaluator_path="official/evaluate.py")
        reports.append(rep)
    df, report = _aggregate_official_reports(reports)
    df.to_csv(out / "shared_validation_metrics.csv", index=False)
    spec_path = ROOT / pipe["cv"]["shared_validation_spec"] if not Path(pipe["cv"]["shared_validation_spec"]).is_absolute() else Path(pipe["cv"]["shared_validation_spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    report.update({
        "benchmark": "vehicle_reid_v5_official shared validation",
        "validation_rows": spec["dataset_expectations"]["shared_validation_rows"],
        "validation_vehicle_ids": spec["dataset_expectations"]["shared_validation_vehicle_ids"],
        "development_rows": spec["dataset_expectations"]["development_rows"],
        "development_vehicle_ids": spec["dataset_expectations"]["development_vehicle_ids"],
        "development_selection_mode": "one fixed identity/source-image-disjoint train/validation split",
        "selected_backbone": summary["backbone"], "selected_stage": summary["selected_stage"],
        "selected_initialization": summary.get("initialization", "dinov3_direct"),
        "veri_pretraining_used": bool(summary.get("init_checkpoint")),
        "selection_leakage_policy": "shared validation never used for architecture/epoch/part/reranker/refusal selection",
        "validation_spec": pipe["cv"]["shared_validation_spec"],
    })
    (out / "shared_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "development_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    export_deployment(pipe, args, summary, fit, out)
    print("\nFIXED SHARED VALIDATION\n", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()
    a.resume = not a.no_resume
    a.dry = a.dry_run
    pipe = read_yaml(ROOT / a.config)
    Path(pipe["paths"]["runs"]).mkdir(parents=True, exist_ok=True)
    ensure_data(pipe, a)
    global_sel = global_baseline_compare(pipe, a)
    if a.dry:
        return
    advanced = advanced_single(pipe, a, global_sel)
    (Path(pipe["paths"]["runs"]) / "development_winner.json").write_text(json.dumps(advanced, ensure_ascii=False, indent=2), encoding="utf-8")
    final_shared_validation(pipe, a, advanced)


if __name__ == "__main__":
    main()
