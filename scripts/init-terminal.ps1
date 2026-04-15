#Requires -Version 5.0
<#
.SYNOPSIS
  Initialize workspace terminal with venv and active env profile.
  
.DESCRIPTION
  Called automatically by VS Code terminal initialization to ensure
  the venv is activated and the selected env profile (.env or .env.azure)
  is loaded every time a terminal opens in this workspace.
#>

$ErrorActionPreference = "Continue"
$ROOT = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ACTIVE_PROFILE_FILE = Join-Path $ROOT ".env.profile"

# Activate venv
$activateScript = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
  & $activateScript
} else {
  Write-Warning "Venv activation script not found: $activateScript"
}

# Load the active env profile into the current shell environment
# so Python and subprocess inherit the right .env file.
if (Test-Path $ACTIVE_PROFILE_FILE) {
  $profileName = (Get-Content -Path $ACTIVE_PROFILE_FILE -TotalCount 1 | Out-String).Trim()
  $profilePath = if ([System.IO.Path]::IsPathRooted($profileName)) {
    $profileName
  } else {
    Join-Path $ROOT $profileName
  }
  
  if (Test-Path $profilePath) {
    # Source the .env file into the current shell so subprocess inherit these values
    $envContent = Get-Content -Path $profilePath -Raw
    $envLines = $envContent -split "`n" | Where-Object { $_ -match '^\s*[^#]' -and $_ -match '=' }
    
    foreach ($line in $envLines) {
      $trimmed = $line.Trim()
      if ($trimmed -and -not $trimmed.StartsWith('#')) {
        $key, $value = $trimmed -split '=', 2
        if ($key) {
          Set-Item -Path "env:$key" -Value $value -ErrorAction SilentlyContinue
        }
      }
    }
  }
}
