<#
.SYNOPSIS
  Sync `.env` for local development and optional Azure sync.

.DESCRIPTION
    After cloning the repo, run this script to:
  1. Sync `.env` from `.env.example`
  2. Preserve existing local values unless `-Force` is used
  3. Optionally merge values from an `azd` environment

  Use `scripts/bootstrap-host.ps1` when you want the full Windows host
  bootstrap: `.venv`, Python dependencies, `.env`, and recommended
  VS Code extensions.

  This is the Windows wrapper for the same env contract used by
  `python scripts/sync_env.py`.

    The default workflow still uses the dev container. This script does
    not create a virtual environment or install Python packages.

.EXAMPLE
    .\scripts\setup-env.ps1
    .\scripts\setup-env.ps1 -Force
    .\scripts\setup-env.ps1 -Force -FromAzd -EnvironmentName dev
    .\scripts\setup-env.ps1 -Force -FromAzd -EnvironmentName validation -OutputPath .env.azure
  .\scripts\bootstrap-host.ps1
#>
param(
  [switch]$Force,
  [switch]$FromAzd,
  [string]$EnvironmentName,
  [string]$OutputPath = ".env"
)

$ErrorActionPreference = "Stop"

$ROOT = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ENV_FILE = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
  $OutputPath
} else {
  Join-Path $ROOT $OutputPath
}
$ENV_TEMPLATE_FILE = Join-Path $ROOT ".env.example"

function Get-DotEnvValues {
  param(
    [string]$Path
  )

  $values = @{}
  if (-not (Test-Path $Path)) {
    return $values
  }

  foreach ($line in Get-Content -Path $Path) {
    if ($line -match '^\s*([A-Z][A-Z0-9_]*)=(.*)$') {
      $values[$Matches[1]] = $Matches[2]
    }
  }

  return $values
}

function New-SyncedEnvContent {
  param(
    [string[]]$TemplateLines,
    [hashtable]$ExistingValues,
    [hashtable]$OverrideValues
  )

  $templateKeys = New-Object System.Collections.Generic.List[string]
  $rendered = New-Object System.Collections.Generic.List[string]

  foreach ($line in $TemplateLines) {
    if ($line -match '^\s*([A-Z][A-Z0-9_]*)=(.*)$') {
      $key = $Matches[1]
      $templateValue = $Matches[2]
      $value = $templateValue

      if ($ExistingValues.ContainsKey($key) -and -not [string]::IsNullOrWhiteSpace($ExistingValues[$key])) {
        $value = [string]$ExistingValues[$key]
      }

      if ($OverrideValues.ContainsKey($key) -and -not [string]::IsNullOrWhiteSpace($OverrideValues[$key])) {
        $value = [string]$OverrideValues[$key]
      }

      $templateKeys.Add($key) | Out-Null
      $rendered.Add("$key=$value") | Out-Null
      continue
    }

    $rendered.Add($line) | Out-Null
  }

  $extraKeys = @($ExistingValues.Keys | Where-Object { $templateKeys -notcontains $_ })
  if ($extraKeys.Count -gt 0) {
    $rendered.Add("") | Out-Null
    $rendered.Add("# Preserved local-only keys") | Out-Null
    foreach ($key in $extraKeys) {
      $rendered.Add("$key=$($ExistingValues[$key])") | Out-Null
    }
  }

  return $rendered
}

function Get-AzdEnvironmentValues {
  param(
    [string]$RootPath,
    [string]$Name
  )

  if (-not (Get-Command azd -ErrorAction SilentlyContinue)) {
    throw "azd CLI was not found on PATH."
  }

  $arguments = @("env", "get-values", "--output", "json")
  if ($Name) {
    $arguments += @("--environment", $Name)
  }

  Push-Location $RootPath
  try {
    $output = & azd @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
      throw ($output | Out-String).Trim()
    }
  } finally {
    Pop-Location
  }

  $json = ($output | Out-String).Trim()
  if (-not $json) {
    return @{}
  }

  $parsed = $json | ConvertFrom-Json
  $values = @{}
  foreach ($property in $parsed.PSObject.Properties) {
    $values[$property.Name] = [string]$property.Value
  }

  return $values
}

if (-not (Test-Path $ENV_TEMPLATE_FILE)) {
  throw "Missing template file: $ENV_TEMPLATE_FILE"
}

$templateLines = Get-Content -Path $ENV_TEMPLATE_FILE
$existingValues = @{}
if ((Test-Path $ENV_FILE) -and -not $Force) {
  $existingValues = Get-DotEnvValues -Path $ENV_FILE
}

$sourceDescription = ".env.example"
$azdValues = @{}
$syncedKeys = @()

if ($FromAzd) {
  Write-Host "Loading values from azd environment..." -ForegroundColor Cyan
  $azdValues = Get-AzdEnvironmentValues -RootPath $ROOT -Name $EnvironmentName
  $syncedKeys = @($azdValues.Keys | Sort-Object)

  if ($EnvironmentName) {
    $sourceDescription = ".env.example + azd environment '$EnvironmentName'"
  } else {
    $sourceDescription = ".env.example + current azd environment"
  }
} elseif (-not $Force -and (Test-Path (Join-Path $ROOT ".env.azure"))) {
  Write-Host "Loading fallback values from .env.azure..." -ForegroundColor Cyan
  $azureValues = Get-DotEnvValues -Path (Join-Path $ROOT ".env.azure")
  foreach ($key in $azureValues.Keys) {
    if ($azureValues[$key] -and $azureValues[$key] -notmatch "placeholder") {
      $azdValues[$key] = $azureValues[$key]
    }
  }
  $syncedKeys = @($azdValues.Keys | Sort-Object)
  $sourceDescription = ".env.example + .env.azure fallback"
}

$rerunCommand = ".\\scripts\\setup-env.ps1 -Force"
if ($FromAzd) {
  $rerunCommand += " -FromAzd"
}
if ($EnvironmentName) {
  $rerunCommand += " -EnvironmentName $EnvironmentName"
}

$syncedTemplateLines = New-SyncedEnvContent -TemplateLines $templateLines -ExistingValues $existingValues -OverrideValues $azdValues
$newContent = ($syncedTemplateLines -join "`n") + "`n"
$existingContent = if (Test-Path $ENV_FILE) { Get-Content -Path $ENV_FILE -Raw } else { $null }

if ($existingContent -eq $newContent) {
  Write-Host ".env is already up to date at $ENV_FILE" -ForegroundColor Green
  exit 0
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ENV_FILE, $newContent, $utf8NoBom)
if ($existingContent) {
  Write-Host "Updated $ENV_FILE" -ForegroundColor Green
} else {
  Write-Host "Created $ENV_FILE" -ForegroundColor Green
}
if ($syncedKeys.Count -gt 0) {
  Write-Host "Synced from azd: $($syncedKeys -join ', ')" -ForegroundColor Green
} elseif ($FromAzd) {
  Write-Host "No matching override values were found in the azd environment; kept repo defaults." -ForegroundColor Yellow
}
if ($OutputPath -eq ".env") {
  Write-Host "Done. Open the repo in the dev container and run: python -m src migrate" -ForegroundColor Green
} else {
  Write-Host "Done. Use G_BSV_ENV_FILE=$OutputPath when you want the Azure-attached host profile." -ForegroundColor Green
}
