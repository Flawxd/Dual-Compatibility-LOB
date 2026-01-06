param(
    [string]$GpuArch = "gfx1200",
    [switch]$Clean
)

Write-Host "================================================" -ForegroundColor Green
Write-Host "GPU LOB - Windows Build Script (AMD HIP/ROCm)" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

if (-not $env:VIRTUAL_ENV) {
    Write-Host "WARNING: Not in a virtual environment!" -ForegroundColor Yellow
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\Activate.ps1
}

$env:ROCM_PATH = "C:\Program Files\AMD\ROCm\6.4"
$env:HIP_PATH = $env:ROCM_PATH

if (-not (Test-Path $env:ROCM_PATH)) {
    Write-Host "ERROR: ROCm not found at $env:ROCM_PATH" -ForegroundColor Red
    Write-Host "Please install AMD Software: Adrenalin Edition with ROCm support" -ForegroundColor Red
    Write-Host "Download from: https://www.amd.com/en/support" -ForegroundColor Red
    exit 1
}

Write-Host "✓ ROCm found at: $env:ROCM_PATH" -ForegroundColor Green

$env:PATH = "$env:ROCM_PATH\bin;$env:PATH"

Write-Host "`nInstalling Python dependencies..." -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel
python -m pip install scikit-build-core pybind11 numpy matplotlib

if (Test-Path "CMakeLists.txt") {
    if (-not (Test-Path "CMakeLists.txt.backup")) {
        Write-Host "Backing up original CMakeLists.txt..." -ForegroundColor Yellow
        Copy-Item "CMakeLists.txt" "CMakeLists.txt.backup"
    }
}

if (Test-Path "CMakeLists.txt.windows") {
    Write-Host "Using Windows-specific CMakeLists.txt..." -ForegroundColor Green
    Copy-Item "CMakeLists.txt.windows" "CMakeLists.txt" -Force
}
else {
    Write-Host "ERROR: CMakeLists.txt.windows not found!" -ForegroundColor Red
    exit 1
}

if ($Clean) {
    Write-Host "`nCleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, _skbuild, dist, *.egg-info
}

$env:GPU_BACKEND = "HIP"
$env:GPU_TARGETS = $GpuArch

Write-Host "`nBuild Configuration:" -ForegroundColor Green
Write-Host "  GPU Architecture: $GpuArch"
Write-Host "  ROCm Path: $env:ROCM_PATH"
Write-Host "  Python: $(python --version)"
Write-Host ""

Write-Host "Building package..." -ForegroundColor Green
Write-Host "This may take 5-10 minutes on first build..." -ForegroundColor Yellow

python -m pip install -e . -v --no-build-isolation

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n================================================" -ForegroundColor Green
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green

    Write-Host "`nVerifying installation..." -ForegroundColor Green
    python -c "import gpu_lob; print(f'✓ Package imported successfully'); print(f'  Device count: {gpu_lob.device_count()}'); print(f'  Device info: {gpu_lob.device_info()}'); print(f'  Version: {gpu_lob.__version__}')"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "  1. Run tests: python test_lob.py"
        Write-Host "  2. Check benchmark plot: benchmark_plot.png"
        Write-Host ""
    }
}
else {
    Write-Host "`n================================================" -ForegroundColor Red
    Write-Host "BUILD FAILED!" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Red
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  1. Visual Studio not installed (need VS 2019 or 2022)"
    Write-Host "  2. ROCm not installed properly"
    Write-Host "  3. AMD GPU drivers not up to date"
    exit 1
}
