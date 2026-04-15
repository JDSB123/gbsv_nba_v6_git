$WorkspacePath = "C:\Users\JDSB\repos\gbsv_nba_v6_git"
Set-Location -Path $WorkspacePath

# Canonical env sync + active profile selection for Azure-backed export.
& "pwsh" -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup-env.ps1" -Force -FromAzd -EnvironmentName production -OutputPath ".env.azure" -CreateAzdEnvIfMissing

# Execute the python script
& ".\.venv\Scripts\python.exe" ".\scripts\export_onedrive.py"
