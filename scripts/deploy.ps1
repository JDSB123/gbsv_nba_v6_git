# deployment script targeting existing nba_gbsv_v6_az environment
#
# Infrastructure coupling:
#   Resource Group:  nba_gbsv_v6_az
#   ACR:             acrnbagbsvv6  (Azure Container Registry)
#   Container Apps:  ca-nba-gbsv-v6-api, ca-nba-gbsv-v6-worker
#
# Usage:
#   .\deploy.ps1                         # build, push, migrate, deploy
#   .\deploy.ps1 -Rollback -RollbackTag <tag>  # roll back to a previous image
param(
    [switch]$Rollback,
    [string]$RollbackTag = ""
)

$ErrorActionPreference = "Stop"

$ResourceGroup = "nba_gbsv_v6_az"
$AcrName = "acrnbagbsvv6"
$AcrLoginServer = "acrnbagbsvv6.azurecr.io"
$Tag = (Get-Date -Format "yyyyMMdd-HHmmss")
$ImageName = "$AcrLoginServer/nba-gbsv-v6:$Tag"
$ApiApp = "ca-nba-gbsv-v6-api"
$WorkerApp = "ca-nba-gbsv-v6-worker"

if ($Rollback) {
    if (-not $RollbackTag) {
        Write-Host "Fetching available tags..."
        $tags = az acr repository show-tags --name $AcrName --repository nba-gbsv-v6 --orderby time_desc --top 5 -o tsv
        Write-Host "Recent tags:"
        $tags | ForEach-Object { Write-Host "  $_" }
        Write-Error "Specify -RollbackTag <tag> to roll back. Use one of the tags above."
        return
    }
    $rollbackImage = "$AcrLoginServer/nba-gbsv-v6:$RollbackTag"
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
az containerapp exec -n $ApiApp -g $ResourceGroup --command "python -m alembic upgrade head"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Database migrations failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host "Deploying new image to Worker Container App: $WorkerApp..."
az containerapp update -n $WorkerApp -g $ResourceGroup --image $ImageName

Write-Host "Deployments successful!"
