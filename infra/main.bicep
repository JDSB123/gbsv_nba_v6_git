targetScope = 'resourceGroup'

var stack = loadJsonContent('stack-config.json')

@description('Location for all resources')
param location string = resourceGroup().location

@description('Location for PostgreSQL (use if primary region is restricted)')
param postgresLocation string = 'centralus'

@description('Environment name')
@allowed(['development', 'production'])
param environment string = 'production'

@secure()
@description('PostgreSQL administrator password')
param postgresPassword string

@secure()
@description('The Odds API key')
param oddsApiKey string

@secure()
@description('Basketball API key')
param basketballApiKey string

@secure()
@description('Teams webhook URL (Power Automate Workflow)')
param teamsWebhookUrl string = ''

@secure()
@description('API key for X-API-Key authentication')
param apiKey string = ''

@description('Optional existing Container Apps Environment resource ID. Leave empty to create one from this template.')
param containerAppsEnvironmentResourceId string = ''

var applicationName = string(stack.applicationName)
var imageRepository = string(stack.imageRepository)
var registryName = string(stack.registryName)
var keyVaultName = string(stack.keyVaultName)
var postgresServerName = string(stack.postgresServerName)
var logAnalyticsName = string(stack.logAnalyticsName)
var appInsightsName = string(stack.applicationInsightsName)
var managedEnvironmentName = string(stack.containerAppsEnvironmentName)
var configuredStorageAccountName = string(stack.storageAccountName)
var storageAccountPrefix = string(stack.storageAccountPrefix)
var apiAppName = string(stack.containerApps.api)
var workerAppName = string(stack.containerApps.worker)
var uniqueSuffix = toLower(uniqueString(subscription().id, resourceGroup().id, applicationName))
var storageAccountName = !empty(configuredStorageAccountName) ? configuredStorageAccountName : take('${storageAccountPrefix}${uniqueSuffix}', 24)
var storageAccountUrl = 'https://${storageAccountName}.blob.${az.environment().suffixes.storage}'
var useExistingManagedEnvironment = !empty(containerAppsEnvironmentResourceId)

// ── Log Analytics ────────────────────────────────────────────────

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 90
  }
}

// ── Application Insights ─────────────────────────────────────────

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ── Key Vault ────────────────────────────────────────────────────

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
}

resource secretOdds 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'odds-api-key'
  properties: { value: oddsApiKey }
}

resource secretBasketball 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'basketball-api-key'
  properties: { value: basketballApiKey }
}

resource secretDbPassword 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'postgres-password'
  properties: { value: postgresPassword }
}

resource secretApiKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'api-key'
  properties: { value: apiKey }
}

resource secretDbUrl 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'database-url'
  properties: { value: dbConnectionString }
}

// ── Storage Account (model artifacts) ────────────────────────────

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource modelContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'models'
  properties: { publicAccess: 'None' }
}

// ── Container Registry ───────────────────────────────────────────

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: registryName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

// ── User-Assigned Managed Identity (used for ACR pull) ───────────

resource acrPullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${applicationName}-acr-pull'
  location: location
}

resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, acrPullIdentity.id, '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  properties: {
    principalId: acrPullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
  }
}

// ── PostgreSQL Flexible Server ───────────────────────────────────

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: postgresServerName
  location: postgresLocation
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    administratorLogin: 'pgadmin'
    administratorLoginPassword: postgresPassword
    storage: { storageSizeGB: 32 }
    backup: {
      backupRetentionDays: 14
      geoRedundantBackup: 'Disabled'
    }
      highAvailability: { mode: 'Disabled' }
  }
}

resource postgresDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgres
  name: 'nba_gbsv'
  properties: { charset: 'UTF8', collation: 'en_US.utf8' }
}

resource postgresFirewall 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// ── Container App: API ───────────────────────────────────────────

var dbConnectionString = 'postgresql+asyncpg://pgadmin:${postgresPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/nba_gbsv?ssl=require'

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = if (!useExistingManagedEnvironment) {
  name: managedEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

var managedEnvironmentId = useExistingManagedEnvironment ? containerAppsEnvironmentResourceId : managedEnvironment.id
var managedEnvironmentNameOutput = useExistingManagedEnvironment ? last(split(containerAppsEnvironmentResourceId, '/')) : managedEnvironmentName

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiAppName
  location: location
  dependsOn: [acrPullRole]
  identity: {
    type: 'SystemAssigned,UserAssigned'
    userAssignedIdentities: {
      '${acrPullIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: managedEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
          maxAge: 3600
        }
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: acrPullIdentity.id
        }
      ]
      secrets: concat([
        { name: 'database-url', value: dbConnectionString }
        { name: 'odds-api-key', value: oddsApiKey }
        { name: 'basketball-api-key', value: basketballApiKey }
      ], !empty(teamsWebhookUrl) ? [
        { name: 'teams-webhook-url', value: teamsWebhookUrl }
      ] : [], !empty(apiKey) ? [
        { name: 'api-key', value: apiKey }
      ] : [])
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acr.properties.loginServer}/${imageRepository}:latest'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat([
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'ODDS_API_KEY', secretRef: 'odds-api-key' }
            { name: 'BASKETBALL_API_KEY', secretRef: 'basketball-api-key' }
            { name: 'APP_ENV', value: environment }
            { name: 'LOG_LEVEL', value: 'INFO' }
            { name: 'AZURE_KEY_VAULT_URL', value: keyVault.properties.vaultUri }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            { name: 'AZURE_STORAGE_ACCOUNT_URL', value: storageAccountUrl }
          ], !empty(teamsWebhookUrl) ? [
            { name: 'TEAMS_WEBHOOK_URL', secretRef: 'teams-webhook-url' }
          ] : [], !empty(apiKey) ? [
            { name: 'API_KEY', secretRef: 'api-key' }
          ] : [])
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health/deep', port: 8000 }
              initialDelaySeconds: 45
              periodSeconds: 60
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '50' } }
          }
        ]
      }
    }
  }
}

// ── Container App: Worker ────────────────────────────────────────

resource workerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: workerAppName
  location: location
  dependsOn: [acrPullRole]
  identity: {
    type: 'SystemAssigned,UserAssigned'
    userAssignedIdentities: {
      '${acrPullIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: managedEnvironmentId
    configuration: {
      registries: [
        {
          server: acr.properties.loginServer
          identity: acrPullIdentity.id
        }
      ]
      secrets: concat([
        { name: 'database-url', value: dbConnectionString }
        { name: 'odds-api-key', value: oddsApiKey }
        { name: 'basketball-api-key', value: basketballApiKey }
      ], !empty(teamsWebhookUrl) ? [
        { name: 'teams-webhook-url', value: teamsWebhookUrl }
      ] : [], !empty(apiKey) ? [
        { name: 'api-key', value: apiKey }
      ] : [])
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: '${acr.properties.loginServer}/${imageRepository}:latest'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          command: ['python', '-m', 'src', 'work']
          env: concat([
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'ODDS_API_KEY', secretRef: 'odds-api-key' }
            { name: 'BASKETBALL_API_KEY', secretRef: 'basketball-api-key' }
            { name: 'API_BASE_URL', value: 'https://${apiApp.properties.configuration.ingress.fqdn}' }
            { name: 'APP_ENV', value: environment }
            { name: 'LOG_LEVEL', value: 'INFO' }
            { name: 'AZURE_KEY_VAULT_URL', value: keyVault.properties.vaultUri }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            { name: 'AZURE_STORAGE_ACCOUNT_URL', value: storageAccountUrl }
          ], !empty(teamsWebhookUrl) ? [
            { name: 'TEAMS_WEBHOOK_URL', secretRef: 'teams-webhook-url' }
          ] : [], !empty(apiKey) ? [
            { name: 'API_KEY', secretRef: 'api-key' }
          ] : [])
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 1 }
    }
  }
}

// ── RBAC: Key Vault Secrets User for API managed identity ─────────

resource apiKeyVaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, apiApp.id, '4633458b-17de-408a-b874-0445c86b69e6')
  properties: {
    principalId: apiApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6') // Key Vault Secrets User
  }
}

resource workerKeyVaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, workerApp.id, '4633458b-17de-408a-b874-0445c86b69e6')
  properties: {
    principalId: workerApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6') // Key Vault Secrets User
  }
}

// ── RBAC: Storage Blob Data Contributor ──────────────────────────

resource apiStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, apiApp.id, 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  properties: {
    principalId: apiApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
  }
}

resource workerStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, workerApp.id, 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  properties: {
    principalId: workerApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
  }
}

// ── Outputs ──────────────────────────────────────────────────────

output apiUrl string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
output postgresHost string = postgres.properties.fullyQualifiedDomainName
output keyVaultUri string = keyVault.properties.vaultUri
output API_BASE_URL string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output AZURE_ACR_LOGIN_SERVER string = acr.properties.loginServer
output ACR_NAME string = acr.name
output IMAGE_NAME string = imageRepository
output RESOURCE_GROUP string = resourceGroup().name
output API_APP string = apiApp.name
output WORKER_APP string = workerApp.name
output PG_SERVER_NAME string = postgres.name
output KV_NAME string = keyVault.name
output ACA_ENVIRONMENT_NAME string = managedEnvironmentNameOutput
output ACA_ENVIRONMENT_ID string = managedEnvironmentId
output POSTGRES_HOST string = postgres.properties.fullyQualifiedDomainName
output AZURE_KEY_VAULT_URL string = keyVault.properties.vaultUri
output AZURE_STORAGE_ACCOUNT_URL string = storageAccountUrl
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appInsights.properties.ConnectionString
