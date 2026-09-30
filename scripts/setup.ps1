$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceCache = Join-Path $projectRoot ".uv-cache"
$env:UV_CACHE_DIR = $workspaceCache

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -ne $uvCommand) {
    $uvExecutable = $uvCommand.Source
}
else {
    $uvExecutable = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path -LiteralPath $uvExecutable)) {
        throw "uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    }
}

Push-Location $projectRoot
try {
    Write-Host "[1/2] Synchronizing dependencies from uv.lock..."
    & $uvExecutable sync --locked --dev --link-mode copy
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }

    Write-Host "[2/2] Verifying the project Python version..."
    $pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
    & $pythonExecutable --version
    if ($LASTEXITCODE -ne 0) {
        throw "The project Python environment could not be verified."
    }
}
finally {
    Pop-Location
}
