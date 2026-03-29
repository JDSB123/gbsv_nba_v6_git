targetScope = 'resourceGroup'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Location for PostgreSQL (use if primary region is restricted)')
param postgresLocation string = 'centralus'

@description('Environment name')
@allowed(['dev', 'prod'])
param environment string = 'dev'

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

@description('Existing Container Apps Environment resource ID to host the API and worker apps.')
param containerAppsEnvironmentResourceId string

var prefix = 'nba-gbsv-v6'
var uniqueSuffix = toLower(uniqueString(subscription().id, resourceGroup().id, prefix))
var shortSuffix = take(uniqueSuffix, 6)
var postgresSuffix = take(uniqueString(postgresLocation, resourceGroup().id, prefix), 6)

// ── Log Analytics ────────────────────────────────────────────────

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${prefix}-${shortSuffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Key Vault ────────────────────────────────────────────────────

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${prefix}-${shortSuffix}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
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

// ── Storage Account (model artifacts) ────────────────────────────

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${uniqueSuffix}'
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
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
  name: 'acr${uniqueSuffix}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

// ── PostgreSQL Flexible Server ───────────────────────────────────

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: 'psql-${prefix}-${postgresSuffix}'
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
      backupRetentionDays: 7
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

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${prefix}-api'
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironmentResourceId
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
        { name: 'database-url', value: dbConnectionString }
        { name: 'odds-api-key', value: oddsApiKey }
        { name: 'basketball-api-key', value: basketballApiKey }
        { name: 'teams-webhook-url', value: teamsWebhookUrl }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acr.properties.loginServer}/nba-gbsv-v6:latest'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'ODDS_API_KEY', secretRef: 'odds-api-key' }
            { name: 'BASKETBALL_API_KEY', secretRef: 'basketball-api-key' }
            { name: 'TEAMS_WEBHOOK_URL', secretRef: 'teams-webhook-url' }
            { name: 'APP_ENV', value: environment }
            { name: 'LOG_LEVEL', value: 'INFO' }
          ]
        }
      ]
      scale: {
        minReplicas: 0
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
  name: 'ca-${prefix}-worker'
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironmentResourceId
    configuration: {
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
        { name: 'database-url', value: dbConnectionString }
        { name: 'odds-api-key', value: oddsApiKey }
        { name: 'basketball-api-key', value: basketballApiKey }
        { name: 'teams-webhook-url', value: teamsWebhookUrl }
      ]
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: '${acr.properties.loginServer}/nba-gbsv-v6:latest'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          command: ['python', '-m', 'src', 'work']
          env: [
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'ODDS_API_KEY', secretRef: 'odds-api-key' }
            { name: 'BASKETBALL_API_KEY', secretRef: 'basketball-api-key' }
            { name: 'TEAMS_WEBHOOK_URL', secretRef: 'teams-webhook-url' }
            { name: 'API_BASE_URL', value: 'https://${apiApp.properties.configuration.ingress.fqdn}' }
            { name: 'APP_ENV', value: environment }
            { name: 'LOG_LEVEL', value: 'INFO' }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 1 }
    }
  }
}

// ── Outputs ──────────────────────────────────────────────────────

output apiUrl string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
output postgresHost string = postgres.properties.fullyQualifiedDomainName
output keyVaultUri string = keyVault.properties.vaultUri
