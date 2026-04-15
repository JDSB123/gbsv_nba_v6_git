<#
.SYNOPSIS
    Runs post-deploy smoke checks for the Azure Container Apps deployment.

.DESCRIPTION
    Validates both API and worker apps are running on the expected image,
    then verifies /health and /health/deep return HTTP 200.

.EXAMPLE
    .\scripts\post-deploy-smoke.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$stackConfigPath = Join-Path $PSScriptRoot "..\infra\stack-config.json"
if (-not (Test-Path $stackConfigPath)) {
    throw "Missing stack contract: $stackConfigPath"
}

$stack = Get-Content -Path $stackConfigPath -Raw | ConvertFrom-Json
$resourceGroup = [string]$stack.resourceGroupName
$apiApp = [string]$stack.containerApps.api
$workerApp = [string]$stack.containerApps.worker

function Get-AppSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AppName
    )

    $json = az containerapp show -g $resourceGroup -n $AppName --output json
    if (-not $json) {
        throw "az containerapp show returned no output for $AppName"
    }

    $app = $json | ConvertFrom-Json
    return [pscustomobject]@{
        Name = $app.name
        RunningStatus = $app.properties.runningStatus
        LatestRevision = $app.properties.latestRevisionName
        LatestReadyRevision = $app.properties.latestReadyRevisionName
        Image = $app.properties.template.containers[0].image
        Fqdn = $app.properties.configuration.ingress.fqdn
    }
}

function Assert-Running {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Summary
    )

    if ($Summary.RunningStatus -ne "Running") {
        throw "$($Summary.Name) is not running (status=$($Summary.RunningStatus))"
    }
    if (-not $Summary.LatestRevision -or $Summary.LatestRevision -ne $Summary.LatestReadyRevision) {
        throw "$($Summary.Name) latest revision is not ready (latest=$($Summary.LatestRevision), ready=$($Summary.LatestReadyRevision))"
    }
}

function Assert-HealthEndpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 45
    }
    catch {
        throw "$Name failed: $($_.Exception.Message)"
    }

    if ($resp.StatusCode -ne 200) {
        throw "$Name returned HTTP $($resp.StatusCode)"
    }

    Write-Host "$Name OK (200)"
}

Write-Host "Reading Azure Container App state..."
$api = Get-AppSummary -AppName $apiApp
$worker = Get-AppSummary -AppName $workerApp

Assert-Running -Summary $api
Assert-Running -Summary $worker

Write-Host "API app:    $($api.Name)"
Write-Host "  Revision: $($api.LatestReadyRevision)"
Write-Host "  Image:    $($api.Image)"
Write-Host "Worker app: $($worker.Name)"
Write-Host "  Revision: $($worker.LatestReadyRevision)"
Write-Host "  Image:    $($worker.Image)"

if ($api.Image -ne $worker.Image) {
    throw "Image mismatch between API and worker: api=$($api.Image), worker=$($worker.Image)"
}

$baseUrl = "https://$($api.Fqdn)"
Assert-HealthEndpoint -Url "$baseUrl/health" -Name "GET /health"
Assert-HealthEndpoint -Url "$baseUrl/health/deep" -Name "GET /health/deep"

Write-Host "Post-deploy smoke checks passed."
