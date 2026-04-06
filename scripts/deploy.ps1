<#
.SYNOPSIS
    Build and deploy the shared ACA prediction image using the stack contract.

.DESCRIPTION
    This script reads Azure deployment metadata from `infra/stack-config.json`
    so local deployment uses the same logical stack contract as Bicep and CI.

.EXAMPLE
    .\scripts\deploy.ps1
    .\scripts\deploy.ps1 -Rollback -RollbackTag 20260406-120000
#>
param(
    [switch]$Rollback,
    [string]$RollbackTag = ""
)

$ErrorActionPreference = "Stop"

$StackConfigPath = Join-Path $PSScriptRoot "..\infra\stack-config.json"
if (-not (Test-Path $StackConfigPath)) {
    throw "Missing stack contract: $StackConfigPath"
}

$Stack = Get-Content -Path $StackConfigPath -Raw | ConvertFrom-Json

$ResourceGroup = [string]$Stack.resourceGroupName
$AcrName = [string]$Stack.registryName
$AcrLoginServer = "$AcrName.azurecr.io"
$ImageRepository = [string]$Stack.imageRepository
$Tag = (Get-Date -Format "yyyyMMdd-HHmmss")
$ImageName = "${AcrLoginServer}/${ImageRepository}:${Tag}"
$ApiApp = [string]$Stack.containerApps.api
$WorkerApp = [string]$Stack.containerApps.worker

if ($Rollback) {
    if (-not $RollbackTag) {
        Write-Host "Fetching available tags..."
        $tags = az acr repository show-tags --name $AcrName --repository $ImageRepository --orderby time_desc --top 5 -o tsv
        Write-Host "Recent tags:"
        $tags | ForEach-Object { Write-Host "  $_" }
        Write-Error "Specify -RollbackTag <tag> to roll back. Use one of the tags above."
        return
    }
    $rollbackImage = "${AcrLoginServer}/${ImageRepository}:${RollbackTag}"
    Write-Host "Rolling back to image: $rollbackImage"
    az containerapp update -n $ApiApp -g $ResourceGroup --image $rollbackImage
    az containerapp update -n $WorkerApp -g $ResourceGroup --image $rollbackImage
    Write-Host "Rollback to $RollbackTag complete."
    return
}

Write-Host "Saving current image tags for rollback..."
$currentApiImage = az containerapp show -n $ApiApp -g $ResourceGroup --query "properties.template.containers[0].image" -o tsv
$currentWorkerImage = az containerapp show -n $WorkerApp -g $ResourceGroup --query "properties.template.containers[0].image" -o tsv
Write-Host "  API:    $currentApiImage"
Write-Host "  Worker: $currentWorkerImage"
Write-Host "  To rollback: .\deploy.ps1 -Rollback -RollbackTag <previous-tag>"

Write-Host "Logging into Azure Container Registry: $AcrName..."
az acr login --name $AcrName

Write-Host "Building Docker Image: $ImageName..."
docker build -t $ImageName .

Write-Host "Pushing Docker Image to ACR..."
docker push $ImageName

Write-Host "Deploying new image to API Container App: $ApiApp..."
az containerapp update -n $ApiApp -g $ResourceGroup --image $ImageName

Write-Host "Running database migrations..."
az containerapp exec -n $ApiApp -g $ResourceGroup --command "python -m src migrate"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Database migrations failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host "Deploying new image to Worker Container App: $WorkerApp..."
az containerapp update -n $WorkerApp -g $ResourceGroup --image $ImageName

Write-Host "Deployments successful!"
