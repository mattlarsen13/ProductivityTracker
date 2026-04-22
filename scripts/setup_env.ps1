Param(
    [switch]$RunSmokeTest,
    [int]$Camera = 0
)

$ErrorActionPreference = "Stop"

function Get-PythonCmd {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            py -3 --version | Out-Null
            if ($LASTEXITCODE -eq 0) { return "py -3" }
        } catch {}
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            python --version | Out-Null
            if ($LASTEXITCODE -eq 0) { return "python" }
        } catch {}
    }
    throw "No Python executable found. Install Python and enable either 'python' or 'py' command."
}

$pythonCmd = Get-PythonCmd

Write-Host "Creating virtual environment..."
Invoke-Expression "$pythonCmd -m venv .venv"

Write-Host "Upgrading pip..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing dependencies..."
.\.venv\Scripts\pip.exe install -r requirements.txt

if ($RunSmokeTest) {
    Write-Host "Running headless webcam smoke test..."
    .\.venv\Scripts\python.exe .\scripts\smoke_webcam.py --no-gui --camera $Camera
}

Write-Host "Setup complete."
