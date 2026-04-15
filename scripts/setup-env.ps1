<#
.SYNOPSIS
  Sync the canonical .env file using scripts/sync_env.py.

.DESCRIPTION
  Thin PowerShell wrapper around scripts/sync_env.py. There is exactly one
  env file at runtime: .env, and exactly one schema source: src/config.py
  (Settings). Pass -FromAzd to overlay values from the active azd environment
  on top of the schema-rendered template.

.EXAMPLE
  .\scripts\setup-env.ps1 -Force

.EXAMPLE
  .\scripts\setup-env.ps1 -Force -FromAzd -EnvironmentName production -CreateAzdEnvIfMissing
#>
param(
  [switch]$Force,
  [switch]$FromAzd,
  [string]$EnvironmentName,
  [switch]$CreateAzdEnvIfMissing
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VENV_PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"

function Resolve-PythonExecutable {
  if (Test-Path $VENV_PYTHON) {
    return $VENV_PYTHON
  }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return $python.Source
  }

  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    return $py.Source
  }

  throw "Python was not found. Install Python 3.14+ or run scripts/bootstrap-host.ps1 first."
}

$useAzd = $FromAzd
if (-not $useAzd -and -not [string]::IsNullOrWhiteSpace($env:AZURE_ENV_NAME)) {
  $useAzd = $true
  Write-Host "AZURE_ENV_NAME detected; using azd environment values as source of truth." -ForegroundColor Cyan
}

$pythonExe = Resolve-PythonExecutable
$syncScript = Join-Path $ROOT "scripts\sync_env.py"
[System.Collections.Generic.List[string]]$syncCommand = @($syncScript)

if ($Force) {
  $syncCommand.Add("--force")
}

if ($useAzd) {
  $syncCommand.Add("--from-azd")
  if ($EnvironmentName) {
    $syncCommand.Add("--environment-name")
    $syncCommand.Add($EnvironmentName)
  }
  if ($CreateAzdEnvIfMissing) {
    $syncCommand.Add("--create-azd-env-if-missing")
  }
  $syncCommand.Add("--seed-azd-from-local")
}

Push-Location $ROOT
try {
  & $pythonExe @syncCommand
  if ($LASTEXITCODE -ne 0) {
    throw "Environment sync failed with exit code $LASTEXITCODE"
  }
} finally {
  Pop-Location
}

if ($useAzd) {
  # Prevent stale process vars from hijacking the freshly synced .env.
  Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:DB_SSL -ErrorAction SilentlyContinue
  Remove-Item Env:ODDS_API_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:BASKETBALL_API_KEY -ErrorAction SilentlyContinue
}

Write-Host "Environment sync complete: .env" -ForegroundColor Green
if ($useAzd) {
  $azdLabel = if ($EnvironmentName) { $EnvironmentName } else { '<current>' }
  Write-Host "Source: azd ($azdLabel)" -ForegroundColor Green
  Write-Host "Cleared conflicting process vars: DATABASE_URL, DB_SSL, ODDS_API_KEY, BASKETBALL_API_KEY" -ForegroundColor Green
} else {
  Write-Host "Source: src/config.py (Settings schema)" -ForegroundColor Green
}
