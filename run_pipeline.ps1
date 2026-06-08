$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$envMdPath = Join-Path $PSScriptRoot "config\environment\env_config.md"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path $envMdPath)) {
    throw "Environment config not found: $envMdPath"
}

$env:PYTHONPATH = "$repoRoot;$($repoRoot)\src"

& $pythonExe (Join-Path $PSScriptRoot "run_full_project_pipeline.py") --env-md $envMdPath @args
