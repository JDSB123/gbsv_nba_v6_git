$WorkspacePath = "C:\Users\JDSB\repos\gbsv_nba_v6_git"
Set-Location -Path $WorkspacePath

# Canonical env sync for Azure-backed export. Writes .env (the only env file).
& "pwsh" -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup-env.ps1" -Force -FromAzd -EnvironmentName production -CreateAzdEnvIfMissing

# Execute the python script
& ".\.venv\Scripts\python.exe" ".\scripts\export_onedrive.py"
