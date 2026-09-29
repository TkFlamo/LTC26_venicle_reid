from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dedicated_transfer_cli_declares_required_options():
    txt = (ROOT / "scripts" / "10b_train_baseline_v5_transfer.py").read_text(encoding="utf-8")
    assert '"--init-checkpoint"' in txt
    assert 'required=True' in txt
    assert '"--plate-mask-prob"' in txt


def test_pipeline_uses_dedicated_cli_for_transfer():
    txt = (ROOT / "scripts" / "30_full_cv_pipeline.py").read_text(encoding="utf-8")
    assert 'entrypoint = "scripts/10b_train_baseline_v5_transfer.py" if init_checkpoint else "scripts/10_train_baseline_v5_exact.py"' in txt
    assert '_assert_veri_transfer_plumbing(check_generic_cli=False)' in txt
