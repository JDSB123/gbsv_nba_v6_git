$WorkspacePath = "C:\Users\JDSB\repos\gbsv_nba_v6_git"
Set-Location -Path $WorkspacePath

# Forcefully load variables from .env.azure into the session
Get-Content ".env.azure" | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.*)\s*$') {
        Set-Item -Path "Env:\$($matches[1].Trim())" -Value $matches[2].Trim()
    }
}

# Override specific SSL and file targets
$env:G_BSV_ENV_FILE = ".env.azure"
$env:DB_SSL = "true"
$env:EXPORT_ALLOW_LOCAL_DB = "true"

# Execute the python script
& ".\.venv\Scripts\python.exe" ".\scripts\export_onedrive.py"
