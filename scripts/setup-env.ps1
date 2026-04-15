<#
.SYNOPSIS
  Sync environment files using the canonical Python env sync tool.

.DESCRIPTION
  This PowerShell wrapper delegates to scripts/sync_env.py so there is one
  implementation for env synchronization across Windows and cross-platform use.

  It also persists the selected host env file to .env.profile so subsequent
  processes resolve the same profile without relying on shell-local env vars.

  If AZURE_ENV_NAME is set and -FromAzd is not explicitly passed, this script
  defaults to using azd values as the source of truth.

.EXAMPLE
  .\scripts\setup-env.ps1 -Force

.EXAMPLE
  .\scripts\setup-env.ps1 -Force -FromAzd -EnvironmentName validation -OutputPath .env.azure -CreateAzdEnvIfMissing
#>
param(
  [switch]$Force,
  [switch]$FromAzd,
  [string]$EnvironmentName,
  [string]$OutputPath = ".env",
  [switch]$CreateAzdEnvIfMissing
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VENV_PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
$ACTIVE_PROFILE_FILE = Join-Path $ROOT ".env.profile"

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
[System.Collections.Generic.List[string]]$syncCommand = @($syncScript, "--output", $OutputPath)

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
  # Prevent stale process vars from hijacking the selected Azure profile.
  Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:DB_SSL -ErrorAction SilentlyContinue
  Remove-Item Env:ODDS_API_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:BASKETBALL_API_KEY -ErrorAction SilentlyContinue
}

$resolvedOutputPath = $OutputPath
if ([System.IO.Path]::IsPathRooted($OutputPath)) {
  $resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
} else {
  $resolvedOutputPath = [System.IO.Path]::GetFullPath((Join-Path $ROOT $OutputPath))
}

$profileSelection = if ($resolvedOutputPath.StartsWith($ROOT, [System.StringComparison]::OrdinalIgnoreCase)) {
  [System.IO.Path]::GetRelativePath($ROOT, $resolvedOutputPath)
} else {
  $resolvedOutputPath
}

[System.IO.File]::WriteAllText($ACTIVE_PROFILE_FILE, "$profileSelection`n", [System.Text.UTF8Encoding]::new($false))
$env:G_BSV_ENV_FILE = $profileSelection

if ($OutputPath -eq ".env") {
  Write-Host "Environment sync complete: $OutputPath" -ForegroundColor Green
  Write-Host "Active profile: $profileSelection" -ForegroundColor Green
  Write-Host "Run: python -m src migrate" -ForegroundColor Green
} else {
  Write-Host "Environment sync complete: $OutputPath" -ForegroundColor Green
  Write-Host "Active profile: $profileSelection" -ForegroundColor Green
  Write-Host "Applied current shell profile: G_BSV_ENV_FILE=$profileSelection" -ForegroundColor Green
  if ($useAzd) {
    Write-Host "Cleared conflicting process vars: DATABASE_URL, DB_SSL, ODDS_API_KEY, BASKETBALL_API_KEY" -ForegroundColor Green
  }
}
