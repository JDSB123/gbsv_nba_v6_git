#Requires -Version 5.0
<#
.SYNOPSIS
  Initialize workspace terminal: activate venv, source .env, expose Sync-Env.

.DESCRIPTION
  Dot-sourced by VS Code on every new terminal in this workspace via the
  PowerShell (venv) profile. Activates the venv and exports variables from
  .env into the current shell so subprocesses inherit them.

  .env is the single source of truth at runtime. There are no other env files.

  Functions exposed by dot-sourcing this script:
    Import-DotEnv           re-export .env into the current shell
    Sync-Env [-Azure]       re-sync .env (optionally from azd) and reload it
                            into the current shell — eliminates the
                            stale-terminal residual after switching profiles.
#>

$script:GBSV_ROOT = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$script:GBSV_VENV_ACTIVATE = Join-Path $script:GBSV_ROOT ".venv\Scripts\Activate.ps1"
$script:GBSV_ENV_FILE = Join-Path $script:GBSV_ROOT ".env"
$script:GBSV_SETUP_ENV = Join-Path $script:GBSV_ROOT "scripts\setup-env.ps1"

function Import-DotEnv {
  [CmdletBinding()]
  param()

  if (-not (Test-Path -LiteralPath $script:GBSV_ENV_FILE)) {
    Write-Warning ".env not found at $script:GBSV_ENV_FILE. Run scripts\setup-env.ps1 (or Sync-Env)."
    return 0
  }

  $loaded = 0
  $lineNo = 0
  foreach ($raw in Get-Content -LiteralPath $script:GBSV_ENV_FILE) {
    $lineNo++
    $line = $raw.Trim()
    if (-not $line) { continue }
    if ($line.StartsWith('#')) { continue }
    if ($line.StartsWith('export ')) { $line = $line.Substring(7).TrimStart() }

    $eq = $line.IndexOf('=')
    if ($eq -lt 1) {
      Write-Warning ".env line ${lineNo}: skipping malformed entry: $raw"
      continue
    }

    $key = $line.Substring(0, $eq).Trim()
    $value = $line.Substring($eq + 1)

    # Strip a single pair of wrapping quotes (single or double) if present.
    if ($value.Length -ge 2) {
      $first = $value[0]
      $last = $value[$value.Length - 1]
      if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
        $value = $value.Substring(1, $value.Length - 2)
      }
    }

    try {
      Set-Item -Path "env:$key" -Value $value
      $loaded++
    } catch {
      Write-Warning ".env line ${lineNo}: failed to set $key — $($_.Exception.Message)"
    }
  }

  return $loaded
}

function Sync-Env {
  <#
  .SYNOPSIS
    Re-sync .env and reload it into the current shell.
  .DESCRIPTION
    Eliminates the "open terminal still has stale env vars" residual after
    switching profiles. Runs setup-env.ps1 to rewrite .env, then re-imports
    .env into THIS shell so DATABASE_URL etc. update in place.
  .PARAMETER Azure
    Sync from the active azd environment, overlaid on the schema in src/config.py.
  .PARAMETER EnvironmentName
    Optional azd environment name. Defaults to AZURE_ENV_NAME.
  #>
  [CmdletBinding()]
  param(
    [switch]$Azure,
    [string]$EnvironmentName
  )

  $setupArgs = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $script:GBSV_SETUP_ENV,
    '-Force'
  )
  if ($Azure) {
    $setupArgs += '-FromAzd'
    $setupArgs += '-CreateAzdEnvIfMissing'
    if ($EnvironmentName) {
      $setupArgs += @('-EnvironmentName', $EnvironmentName)
    }
  }

  & pwsh @setupArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Sync-Env: setup-env.ps1 failed (exit $LASTEXITCODE). Shell env was NOT reloaded."
    return
  }

  $count = Import-DotEnv
  Write-Host "[Sync-Env] Reloaded $count vars from .env into current shell." -ForegroundColor Green
}

# --- bootstrap the current shell on (dot-)source ----------------------------
if (Test-Path -LiteralPath $script:GBSV_VENV_ACTIVATE) {
  & $script:GBSV_VENV_ACTIVATE
} else {
  Write-Warning "venv not found at $script:GBSV_VENV_ACTIVATE. Run scripts\bootstrap-host.ps1."
}

$initialLoaded = Import-DotEnv
Write-Host "[init] venv + $initialLoaded vars from .env. Use 'Sync-Env' or 'Sync-Env -Azure' to refresh in place." -ForegroundColor DarkGray
