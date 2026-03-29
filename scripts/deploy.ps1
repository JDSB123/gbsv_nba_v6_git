# deployment script targeting existing nba_gbsv_v6_az environment
$ErrorActionPreference = "Stop"

$ResourceGroup = "nba_gbsv_v6_az"
$AcrName = "acrnbagbsvv6"
$AcrLoginServer = "acrnbagbsvv6.azurecr.io"
$Tag = (Get-Date -Format "yyyyMMdd-HHmmss")
$ImageName = "$AcrLoginServer/nba-gbsv-v6:$Tag"
$ApiApp = "ca-nba-gbsv-v6-api"
$WorkerApp = "ca-nba-gbsv-v6-worker"

Write-Host "Logging into Azure Container Registry: $AcrName..."
az acr login --name $AcrName

Write-Host "Building Docker Image: $ImageName..."
docker build -t $ImageName .

Write-Host "Pushing Docker Image to ACR..."
docker push $ImageName

Write-Host "Deploying new image to API Container App: $ApiApp..."
az containerapp update -n $ApiApp -g $ResourceGroup --image $ImageName

Write-Host "Deploying new image to Worker Container App: $WorkerApp..."
az containerapp update -n $WorkerApp -g $ResourceGroup --image $ImageName

Write-Host "Deployments successful!"
