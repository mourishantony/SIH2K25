"""Patch mask_classifier.py: add version-mismatch auto-recovery in _load_or_train."""
import sys
from pathlib import Path

mask_file = Path("src/mask_classifier.py")
if not mask_file.exists():
    print("ERROR: src/mask_classifier.py not found")
    sys.exit(1)

content = mask_file.read_text(encoding="utf-8")

if "version_err" in content:
    print("[FIX 3] mask_classifier.py already patched -- skipping")
    sys.exit(0)

old_code = """\
    def _load_or_train(self):
        if _MODEL_PATH.exists():
            return load(_MODEL_PATH)
        if not _DATASET_ROOT.exists():
            rprint("[yellow]Mask dataset folder missing; using fallback probabilities.[/]")
            return _FallbackModel()
        return _train_model(self.max_samples)"""

new_code = """\
    def _load_or_train(self):
        if _MODEL_PATH.exists():
            # Detect sklearn version mismatch and retrain automatically
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    model = load(_MODEL_PATH)
                return model
            except Exception as version_err:
                rprint(
                    f"[yellow]Mask model version mismatch ({version_err}). "
                    f"Deleting stale model and retraining...[/]"
                )
                try:
                    _MODEL_PATH.unlink()
                except OSError:
                    pass
        if not _DATASET_ROOT.exists():
            rprint("[yellow]Mask dataset folder missing; using fallback probabilities.[/]")
            return _FallbackModel()
        return _train_model(self.max_samples)"""

if old_code in content:
    patched = content.replace(old_code, new_code)
    mask_file.write_text(patched, encoding="utf-8")
    print("[FIX 3] Patched _load_or_train in mask_classifier.py")
else:
    print("[FIX 3] WARNING: Could not find exact old _load_or_train code.")
    print("        FIX 1 (deleting the .joblib) is sufficient -- this is optional.")
