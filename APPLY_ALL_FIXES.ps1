# ============================================================
# APPLY ALL FIXES -- run this from your project root:
#   cd C:\Users\mourish\Desktop\SIH2k25\SIH-MDR_PATHOGEN-Final\script
#   .\APPLY_ALL_FIXES.ps1
# ============================================================

Write-Host ""
Write-Host "=== Patient Contact Tracing -- Applying All Fixes ===" -ForegroundColor Cyan
Write-Host ""

# -- FIX 1: Delete stale sklearn mask model (version mismatch) --
$maskModel = "src\mask_datas\mask_detector.joblib"
if (Test-Path $maskModel) {
    Remove-Item $maskModel -Force
    Write-Host "[FIX 1] Deleted stale mask_detector.joblib (sklearn version mismatch)" -ForegroundColor Green
    Write-Host "        It will be retrained automatically on next startup." -ForegroundColor Gray
} else {
    Write-Host "[FIX 1] mask_detector.joblib not found -- nothing to delete (OK)" -ForegroundColor Yellow
}

# -- FIX 2: Check monitor_service.py _build_tracker --
$monitorFile = "src\monitor_service.py"
if (Test-Path $monitorFile) {
    $content = Get-Content $monitorFile -Raw
    if ($content -match "_build_tracker") {
        if ($content -match "candidate_src") {
            Write-Host "[FIX 2] _build_tracker already patched -- skipping" -ForegroundColor Yellow
        } else {
            Write-Host "[FIX 2] _build_tracker found but not yet patched -- applying..." -ForegroundColor Cyan
            # Apply patch via Python to avoid PowerShell here-string issues
            python src\apply_monitor_patch.py
        }
    } else {
        Write-Host "[FIX 2] ERROR: Could not find _build_tracker in monitor_service.py" -ForegroundColor Red
    }
} else {
    Write-Host "[FIX 2] monitor_service.py not found -- skipping" -ForegroundColor Yellow
}

# -- FIX 3: Check mask_classifier.py --
$maskFile = "src\mask_classifier.py"
if (Test-Path $maskFile) {
    $maskContent = Get-Content $maskFile -Raw
    if ($maskContent -match "version_err") {
        Write-Host "[FIX 3] mask_classifier.py already patched -- skipping" -ForegroundColor Yellow
    } else {
        Write-Host "[FIX 3] Applying mask_classifier.py patch..." -ForegroundColor Cyan
        python src\apply_mask_patch.py
    }
} else {
    Write-Host "[FIX 3] mask_classifier.py not found -- skipping" -ForegroundColor Yellow
}

# -- FIX 4: Upgrade scikit-learn --
Write-Host ""
Write-Host "[FIX 4] Upgrading scikit-learn..." -ForegroundColor Cyan
pip install scikit-learn --upgrade --quiet
Write-Host "[FIX 4] scikit-learn upgrade done" -ForegroundColor Green

# -- VERIFY --
Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
python src\verify_fixes.py

Write-Host ""
Write-Host "=== Done! Now restart your backend: ===" -ForegroundColor Cyan
Write-Host "   python backend/main.py" -ForegroundColor White
Write-Host "   -- or --" -ForegroundColor Gray
Write-Host "   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host ""
