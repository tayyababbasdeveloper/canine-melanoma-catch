# =====================================================================
# One-command environment setup for Windows (PowerShell).
#
# Creates an ISOLATED virtual environment (.venv), installs the CPU build
# of PyTorch and all project dependencies, then verifies the install.
# The .venv keeps this project's packages from disturbing other projects
# or your global Python.
#
# Usage (from the project root, in PowerShell):
#     powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# Options:
#     -Gpu              install the CUDA build of PyTorch instead of CPU
#     -Python <path>    use a specific python.exe (default: 'python')
# =====================================================================
param(
    [switch]$Gpu,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # project root

Write-Host "==> Project root: $(Get-Location)" -ForegroundColor Cyan

# 1. Check Python -----------------------------------------------------
Write-Host "==> Checking Python..." -ForegroundColor Cyan
& $Python --version
if ($LASTEXITCODE -ne 0) {
    throw "Python not found. Install Python 3.10-3.14 from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
}

# 2. Create venv ------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & $Python -m venv .venv
} else {
    Write-Host "==> .venv already exists - reusing it." -ForegroundColor Yellow
}
$Py = ".\.venv\Scripts\python.exe"

# 3. Upgrade pip ------------------------------------------------------
Write-Host "==> Upgrading pip..." -ForegroundColor Cyan
& $Py -m pip install --upgrade pip

# 4. Install PyTorch (CPU by default) --------------------------------
# Installed from PyTorch's own index FIRST so we get the right (CPU/CUDA)
# build before requirements.txt is resolved.
if ($Gpu) {
    Write-Host "==> Installing PyTorch (CUDA 12.1 build)..." -ForegroundColor Cyan
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --retries 10 --timeout 180
} else {
    Write-Host "==> Installing PyTorch (CPU build)..." -ForegroundColor Cyan
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --retries 10 --timeout 180
}

# 5. Install the rest -------------------------------------------------
Write-Host "==> Installing project dependencies (requirements.txt)..." -ForegroundColor Cyan
& $Py -m pip install -r requirements.txt --retries 10 --timeout 180

# 6. Verify -----------------------------------------------------------
Write-Host "==> Verifying installation..." -ForegroundColor Cyan
& $Py -c "import torch, torchvision, segmentation_models_pytorch as smp, albumentations, cv2, sklearn, scipy; print('OK  torch', torch.__version__, '| smp', smp.__version__, '| cuda', torch.cuda.is_available())"

Write-Host ""
Write-Host "================ SETUP COMPLETE ================" -ForegroundColor Green
Write-Host "Activate the environment with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then run the demo pipelines:" -ForegroundColor Green
Write-Host "    python scripts\run_week1_2_pipeline.py --demo"
Write-Host "    python scripts\run_week3_4_pipeline.py --demo --quick"
