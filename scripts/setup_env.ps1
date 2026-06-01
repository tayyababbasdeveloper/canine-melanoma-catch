# =====================================================================
# One-command environment setup (Windows PowerShell)
# Creates an ISOLATED virtual environment in .venv so this project's
# packages never disturb your other projects or global Python.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# =====================================================================

Write-Host "==> Creating virtual environment in .venv ..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "==> Upgrading pip ..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "==> Installing dependencies (requirements.txt) ..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. To activate the environment, run:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then run the pipeline:" -ForegroundColor Green
Write-Host "    python scripts\run_week1_2_pipeline.py --demo"
