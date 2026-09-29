#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

FINAL_FILES = (
    "submission.csv",
    "candidates.csv",
    "embeddings.npy",
)


def _resolve_path(value: str, *, relative_to_root: bool = False) -> Path:
    p = Path(value).expanduser()

    if not p.is_absolute():
        if relative_to_root:
            p = ROOT / p
        else:
            p = Path.cwd() / p

    return p.resolve()


def _runtime_env() -> dict[str, str]:
    """
    Force subprocesses to import vehicle_fingerprint from THIS dist package.

    This also prevents an old development checkout from shadowing the packaged
    src/ directory through PYTHONPATH.
    """
    env = os.environ.copy()

    local_src = str((ROOT / "src").resolve())
    old_pythonpath = env.get("PYTHONPATH", "")

    if old_pythonpath:
        env["PYTHONPATH"] = local_src + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = local_src

    return env


def _validate_internal_outputs(out: Path) -> None:
    missing = [
        name
        for name in FINAL_FILES
        if not (out / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Inference completed, but required submission files are missing: "
            + ", ".join(missing)
        )


def _check_safe_output_dir(out: Path, input_dir: Path) -> None:
    """
    --out is cleaned before publishing the final three files,
    therefore guard against obviously unsafe paths.
    """
    out = out.resolve()
    input_dir = input_dir.resolve()

    blocked = {
        ROOT.resolve(),
        input_dir,
        Path.home().resolve(),
    }

    # Drive/filesystem root, e.g. C:\\ or /
    try:
        blocked.add(Path(out.anchor).resolve())
    except Exception:
        pass

    if out in blocked:
        raise SystemExit(
            f"Refusing to use unsafe --out directory: {out}"
        )


def _clear_output_dir(out: Path) -> None:
    """
    Make the final submission directory contain exactly the organizer files.
    """
    out.mkdir(parents=True, exist_ok=True)

    for child in list(out.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _publish_submission(temp_out: Path, final_out: Path) -> None:
    _validate_internal_outputs(temp_out)

    _clear_output_dir(final_out)

    for name in FINAL_FILES:
        shutil.copy2(
            temp_out / name,
            final_out / name,
        )

    remaining = sorted(
        p.name
        for p in final_out.iterdir()
    )

    expected = sorted(FINAL_FILES)

    if remaining != expected:
        raise RuntimeError(
            "Unexpected files in final output directory.\n"
            f"Expected: {expected}\n"
            f"Found:    {remaining}"
        )


def _build_command(
    args: argparse.Namespace,
    internal_out: Path,
) -> list[str]:

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "56_infer_deployment_folder_v094.py"),

        "--input-dir",
        str(_resolve_path(args.input_dir)),

        "--deployment-dir",
        str(args.deployment_dir),

        "--out",
        str(internal_out),

        "--device",
        str(args.device),

        "--precision",
        str(args.precision),

        "--batch",
        str(args.batch),

        "--workers",
        str(args.workers),

        "--member",
        str(args.member),
    ]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Falcon ReID v0.9.4 submission runner. "
            "Production default: base_full."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--deployment-dir",
        default="deploy/models_current",
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--member",
        default="base_full",
        help=(
            "Runtime deployment member. "
            "Production default is base_full. "
            "Use ensemble only for compatibility/debug."
        ),
    )

    parser.add_argument(
        "--keep-debug-artifacts",
        action="store_true",
        help=(
            "Disable clean-output mode and write all internal artifacts "
            "directly into --out. Intended only for debugging."
        ),
    )

    args = parser.parse_args()

    input_dir = _resolve_path(args.input_dir)
    final_out = _resolve_path(args.out)

    if not input_dir.is_dir():
        raise SystemExit(
            f"Input directory does not exist: {input_dir}"
        )

    _check_safe_output_dir(
        final_out,
        input_dir,
    )

    env = _runtime_env()

    # Debug/legacy mode:
    # preserve the previous behavior and expose internal artifacts.
    if args.keep_debug_artifacts:
        final_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        cmd = _build_command(
            args,
            final_out,
        )

        print(
            "[RUN]",
            " ".join(cmd),
            flush=True,
        )

        subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=True,
        )

        _validate_internal_outputs(
            final_out
        )

        print(
            f"[OK] inference completed: {final_out}",
            flush=True,
        )

        return

    # Production mode:
    #
    # 56 -> 21 may create:
    #   work/
    #   retrieval_recipe_used.json
    #   hackathon_io_validation.json
    # and other diagnostic files.
    #
    # They are intentionally isolated in a TemporaryDirectory and never
    # published into the organizer output directory.
    with tempfile.TemporaryDirectory(
        prefix="falcon_reid_v094_"
    ) as tmp:

        temp_root = Path(tmp)
        internal_out = temp_root / "inference_output"

        cmd = _build_command(
            args,
            internal_out,
        )

        print(
            "[RUN]",
            " ".join(cmd),
            flush=True,
        )

        subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=True,
        )

        # 56 already performs organizer I/O validation.
        # Only after that subprocess succeeds do we publish the final files.
        _publish_submission(
            internal_out,
            final_out,
        )

    print()
    print("[OK] Falcon ReID submission ready")
    print(f"[OUT] {final_out}")
    print()

    for name in FINAL_FILES:
        p = final_out / name
        print(
            f"  {name:<16} "
            f"{p.stat().st_size / (1024 ** 2):.2f} MiB"
        )

    print()
    print(
        "[OK] Final output contains only: "
        "submission.csv, candidates.csv, embeddings.npy"
    )


if __name__ == "__main__":
    main()