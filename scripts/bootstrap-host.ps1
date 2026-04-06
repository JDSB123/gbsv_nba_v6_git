<#
.SYNOPSIS
  Bootstrap the Windows host development stack for this repo.

.DESCRIPTION
  This script creates or refreshes `.venv`, installs the repo's dev
  dependencies from `pyproject.toml`, syncs `.env` from `.env.example`,
  and optionally installs the recommended VS Code extensions.

  Use this when you want to work on the Windows host instead of the
  default dev-container workflow.

.EXAMPLE
  .\scripts\bootstrap-host.ps1

.EXAMPLE
  .\scripts\bootstrap-host.ps1 -RecreateVenv

.EXAMPLE
  .\scripts\bootstrap-host.ps1 -SkipVSCodeExtensions
#>
param(
  [switch]$RecreateVenv,
  [switch]$SkipDependencyInstall,
  [switch]$SkipVSCodeExtensions
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VENV_DIR = Join-Path $ROOT ".venv"
$VENV_PYTHON = Join-Path $VENV_DIR "Scripts\python.exe"
$EXTENSIONS_FILE = Join-Path $ROOT ".vscode\extensions.json"
$PYTHON_VERSION_FILE = Join-Path $ROOT ".python-version"
$MINIMUM_PYTHON_VERSION = "3.14"
if (Test-Path $PYTHON_VERSION_FILE) {
  $MINIMUM_PYTHON_VERSION = (Get-Content -Path $PYTHON_VERSION_FILE -TotalCount 1 | Out-String).Trim()
}
$MINIMUM_PYTHON_VERSION_OBJECT = [version]("$MINIMUM_PYTHON_VERSION.0")

function Invoke-CheckedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter()]
    [string[]]$Arguments = @(),

    [Parameter(Mandatory = $true)]
    [string]$FailureMessage
  )

  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw $FailureMessage
  }
}

function Resolve-BootstrapPython {
  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($pyLauncher) {
    return @{
      Executable = $pyLauncher.Source
      Arguments = @("-$MINIMUM_PYTHON_VERSION")
      Display = "py -$MINIMUM_PYTHON_VERSION"
    }
  }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return @{
      Executable = $python.Source
      Arguments = @()
      Display = $python.Source
    }
  }

  throw "Python $MINIMUM_PYTHON_VERSION+ was not found. Install Python $MINIMUM_PYTHON_VERSION and rerun .\\scripts\\bootstrap-host.ps1."
}

function Get-PythonVersion {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter()]
    [string[]]$Arguments = @()
  )

  $versionOutput = & $Executable @($Arguments + @("-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"))
  if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine the Python version for $Executable."
  }

  return [version]($versionOutput | Out-String).Trim()
}

function Install-RecommendedExtensions {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ExtensionsPath
  )

  if (-not (Test-Path $ExtensionsPath)) {
    Write-Warning "Missing VS Code recommendations file: $ExtensionsPath"
    return
  }

  $code = Get-Command code -ErrorAction SilentlyContinue
  if (-not $code) {
    Write-Warning "VS Code CLI 'code' is not on PATH. Skipping extension installation."
    return
  }

  $payload = Get-Content -Path $ExtensionsPath -Raw | ConvertFrom-Json
  $recommendations = @($payload.recommendations | Where-Object { $_ }) | Sort-Object -Unique
  foreach ($extension in $recommendations) {
    Write-Host "Installing VS Code extension: $extension" -ForegroundColor Cyan
    Invoke-CheckedCommand `
      -Executable $code.Source `
      -Arguments @("--install-extension", $extension, "--force") `
      -FailureMessage "Failed to install VS Code extension '$extension'."
  }
}

if ($RecreateVenv -and (Test-Path $VENV_DIR)) {
  Write-Host "Removing existing .venv" -ForegroundColor Yellow
  Remove-Item -Path $VENV_DIR -Recurse -Force
}

if (-not (Test-Path $VENV_PYTHON)) {
  $bootstrapPython = Resolve-BootstrapPython
  $bootstrapVersion = Get-PythonVersion -Executable $bootstrapPython.Executable -Arguments $bootstrapPython.Arguments
  if ($bootstrapVersion -lt $MINIMUM_PYTHON_VERSION_OBJECT) {
    throw "Resolved bootstrap Python is $bootstrapVersion. Python $MINIMUM_PYTHON_VERSION+ is required."
  }

  Write-Host "Creating .venv with $($bootstrapPython.Display)" -ForegroundColor Cyan
  Invoke-CheckedCommand `
    -Executable $bootstrapPython.Executable `
    -Arguments @($bootstrapPython.Arguments + @("-m", "venv", $VENV_DIR)) `
    -FailureMessage "Failed to create .venv."
}

if (-not (Test-Path $VENV_PYTHON)) {
  throw "Expected virtual environment interpreter was not created: $VENV_PYTHON"
}

$venvVersion = Get-PythonVersion -Executable $VENV_PYTHON
if ($venvVersion -lt $MINIMUM_PYTHON_VERSION_OBJECT) {
  throw ".venv is using Python $venvVersion. Re-run with -RecreateVenv against Python $MINIMUM_PYTHON_VERSION+."
}

if (-not $SkipDependencyInstall) {
  Write-Host "Upgrading pip in .venv" -ForegroundColor Cyan
  Invoke-CheckedCommand `
    -Executable $VENV_PYTHON `
    -Arguments @("-m", "pip", "install", "--upgrade", "pip") `
    -FailureMessage "Failed to upgrade pip in .venv."

  Write-Host "Installing repo dev dependencies into .venv" -ForegroundColor Cyan
  Push-Location $ROOT
  try {
    Invoke-CheckedCommand `
      -Executable $VENV_PYTHON `
      -Arguments @("-m", "pip", "install", "-e", ".[dev]") `
      -FailureMessage "Failed to install repo dependencies into .venv."
  } finally {
    Pop-Location
  }
}

Write-Host "Syncing .env from .env.example" -ForegroundColor Cyan
Push-Location $ROOT
try {
  Invoke-CheckedCommand `
    -Executable $VENV_PYTHON `
    -Arguments @("scripts/sync_env.py") `
    -FailureMessage "Failed to sync .env from .env.example."
} finally {
  Pop-Location
}

if (-not $SkipVSCodeExtensions) {
  Install-RecommendedExtensions -ExtensionsPath $EXTENSIONS_FILE
}

Write-Host "Host bootstrap complete." -ForegroundColor Green
Write-Host "Use .venv\\Scripts\\python.exe for host commands, or reopen the repo in the dev container for the default workflow." -ForegroundColor Green