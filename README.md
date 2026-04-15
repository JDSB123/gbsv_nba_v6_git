# NBA GBSV v6

Container-first Python service for NBA predictions, backfills, model training, and Teams publishing.

This repo uses one default development workflow.

One dependency source: `pyproject.toml`

## Stack Contract

This repo now has one primary source of truth per concern:

| Concern | Source of truth | Purpose |
| --- | --- | --- |
| Python dependencies and tooling | `pyproject.toml` | Runtime and dev dependencies, pytest, Ruff, package metadata |
| Local env schema and safe defaults | `.env.example` | Canonical local env keys and default values |
| Active host env profile selection | `.env.profile` (local state) | Persists the currently selected host env file for cross-process consistency |
| Azure runtime environment values | `azd` environment (`.azure/<env>/.env`) and ACA app settings | Canonical cloud values consumed by provision/deploy/runtime |
| Local env sync | `scripts/sync_env.py` | Cross-platform `.env` sync and optional azd overlay |
| Windows env wrapper | `scripts/setup-env.ps1` | PowerShell wrapper around the same env contract |
| App entry points | `src/__main__.py` | Canonical `python -m src ...` command family |
| Data source config | `src/config.py` | Single source of truth for regions, markets, intervals, enable/disable flags |
| Azure stack metadata | `infra/stack-config.json` | Shared logical names for ACA, ACR, Key Vault, PostgreSQL, resource group |
| Azure provisioning | `azure.yaml`, `infra/main.parameters.json`, `infra/main.bicep` | azd + Bicep deployment contract |
| GitHub CI/CD stack export | `scripts/export_stack_env.py` | Exports shared Azure stack values to Actions |
| Local deploy helper | `scripts/deploy.ps1` | Uses the same stack contract and deploys the shared ACA image |

## Data Pipeline

All external data sources are configured centrally in `src/config.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `ODDS_API_REGIONS` | `us,us2,eu` | Regions to poll — captures retail, offshore, AND sharp books (Pinnacle, bet365) |
| `ODDS_API_MARKETS_FG` | `h2h,spreads,totals` | Full-game market types |
| `ODDS_API_MARKETS_1H` | `h2h_h1,spreads_h1,totals_h1` | First-half market types |
| `ODDS_API_QUOTA_MIN` | `50` | Skip fetches when remaining quota drops below this |

### Data Sources

| Source | Status | What it provides |
| --- | --- | --- |
| The Odds API v4 | ✅ Active (24M+ quota) | FG odds, 1H odds, player props across all `us,us2,eu` books |
| Basketball API v1 | ✅ Active (Mega plan, 150K/day) | Games, scores, box scores, team stats, player stats |

### Reliability

- **Startup health check**: Worker validates all APIs + DB on boot, logs a status table
- **Data freshness monitoring**: `check_data_freshness` job runs every 30 min, sends Teams alerts when polls go stale
- **Circuit breakers**: Per-service failure tracking (threshold=5, cooldown=10 min)
- **Dead-letter queue**: Failed ingestion jobs logged and retried automatically
- **Live dashboard**: `GET /health/data-sources` returns API status, active config, odds freshness, and all bookmakers seen in last 24h

### Health Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Basic liveness |
| `GET /health/deep` | DB + models + odds freshness |
| `GET /health/freshness` | Latest timestamps per data source |
| `GET /health/data-sources` | Full pipeline dashboard — API checks, config, bookmaker list |

## Default Development Workflow

The default path is the VS Code dev container.

1. Clone the repo.
2. Open it in VS Code.
3. Reopen in the dev container.
4. Let the container run `python scripts/sync_env.py --quiet` on create and on each start.
5. Run `python -m src migrate` if you need to re-apply migrations manually.
6. Use the command family in `src/__main__.py` or the matching VS Code tasks.

The dev container provides the Python runtime, system packages, Docker tooling, local Postgres, and the default `DATABASE_URL` for local development.

## Fresh Clone Behavior

Fresh clone behavior is now intentionally self-healing:

- `.env.example` defines the local env contract.
- `scripts/sync_env.py` creates or repairs `.env` without overwriting non-empty local values.
- `.devcontainer/devcontainer.json` runs the sync on create and on every start.
- The same sync contract is used by CI smoke setup and by Azure post-provision hooks.

That means missing or newly added env keys are repaired automatically when the repo is reopened.

## Fresh Host Clone

If you want to work on the Windows host instead of the default dev container, use one committed entrypoint:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap-host.ps1
```

That script will:

- create `.venv` with Python 3.14+
- install the repo's dev dependencies from `pyproject.toml`
- sync `.env` from `.env.example`
- install the recommended VS Code extensions when the `code` CLI is available

Useful variants:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap-host.ps1 -RecreateVenv
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap-host.ps1 -SkipVSCodeExtensions
```

## Entry Points

The canonical runtime entry points are all under `python -m src`:

- `python -m src serve`
- `python -m src work`
- `python -m src train`
- `python -m src predict`
- `python -m src backfill`
- `python -m src migrate`
- `python -m src perf`
- `python -m src audit`

VS Code tasks in `.vscode/tasks.json` mirror the main operational commands so the editor and CLI stay aligned.

## Local Environment

Use two explicit host-side profiles instead of one mixed file.

- `.env` is the local host-development profile. VS Code and Python default to this file.
- `.env.azure` is the optional Azure-attached host profile.
- `.env.profile` stores the active host profile selection (`.env` or `.env.azure`).
- `.env.example` is the schema and default template for both.
- `python scripts/sync_env.py --force` resets `.env` to local defaults.
- `python scripts/sync_env.py --from-azd --output .env.azure --force --create-azd-env-if-missing` creates or refreshes `.env.azure` from the active azd environment.
- When syncing from azd, required keys (`ODDS_API_KEY`, `BASKETBALL_API_KEY`, `DATABASE_URL`) now fail fast if missing or placeholder values.
- `scripts/setup-env.ps1 -FromAzd` will attempt to seed missing azd keys from real local shell/.env values before fail-fast validation.
- Use `--allow-incomplete-azd` only for deliberate partial sync scenarios.
- `powershell -ExecutionPolicy Bypass -File .\scripts\setup-env.ps1 -Force -OutputPath .env` syncs local profile and marks `.env` active.
- `powershell -ExecutionPolicy Bypass -File .\scripts\setup-env.ps1 -Force -FromAzd -EnvironmentName production -OutputPath .env.azure -CreateAzdEnvIfMissing` syncs Azure profile and marks `.env.azure` active.
- `G_BSV_ENV_FILE` remains an explicit per-process override, but normal host flows should rely on `.env.profile` instead of manual shell exports.

Local database defaults:

- Dev container: `postgresql+asyncpg://postgres:postgres@db:5432/nba_gbsv`
- Host-only: `postgresql+asyncpg://postgres:postgres@localhost:5432/nba_gbsv`

The default development workflow remains the dev container. The dev container stays local and isolated through `.devcontainer/docker-compose.yml`, even if `.env.azure` exists.

Production does not depend on repo-local `.env` or `.env.azure`; production values come from ACA app settings and Azure-managed resources. Local env files are generated mirrors for host workflows.

## Host Export

The OneDrive export helper uses the same host env contract as the rest of the repo.

For an Azure-backed export on the Windows host:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-env.ps1 -Force -FromAzd -EnvironmentName production -OutputPath .env.azure -CreateAzdEnvIfMissing
$env:ONEDRIVE_EXPORT_ROOT = 'C:\Users\<you>\OneDrive - Green Bier Capital\Early Stage Sport Ventures - Documents\NBA - Green Bier Sports'
.\.venv\Scripts\python.exe .\scripts\export_onedrive.py
```

`scripts/export_onedrive.py` refuses to use a localhost database unless `EXPORT_ALLOW_LOCAL_DB=true` is set explicitly.

## Clean And Prune

For a fresh local reset without touching tracked source files, use Git only for generated state:

```powershell
git clean -fdX
git fetch --prune --tags
git remote prune origin
```

`git clean -fdX` removes ignored local artifacts such as `.venv`, `.env`, `.env.azure`, `.azure`, `infra/main.json`, and local run outputs. After that, rerun `scripts/bootstrap-host.ps1` for host work or reopen in the dev container for the default workflow.

Commit the stack contract files. Do not commit local state.

- Commit: `pyproject.toml`, `.devcontainer/*`, `.vscode/tasks.json`, `.vscode/extensions.json`, `.vscode/settings.json`, `.env.example`, `scripts/*.ps1`, `scripts/*.py`, `azure.yaml`, and `infra/*` source files.
- Do not commit: `.venv/`, `.env`, `.env.azure*`, `.env.profile`, `.azure/`, `infra/main.json`, and local `*_output*.txt` files.

## Git and CI

`master` is a protected branch.

`master` is protected. The expected change flow is:

1. Branch from `master`.
2. Make changes against the same stack contract files described above.
3. Run `ruff check src tests` and `pytest`.
4. Open a pull request.

The GitHub Actions workflow now:

- uses `pyproject.toml` for installs
- uses `scripts/sync_env.py` for fresh-checkout smoke setup
- uses `infra/stack-config.json` through `scripts/export_stack_env.py` for shared Azure names
- builds one production image and deploys that same tag to both ACA services

## Azure and ACA

The Azure deployment contract is:

- `azure.yaml`
- `infra/main.parameters.json`
- `infra/main.bicep`
- `infra/stack-config.json`

Key behaviors:

- `infra/stack-config.json` defines the logical Azure stack names consumed by Bicep, CI, and deploy helpers.
- `infra/main.bicep` provisions the Azure resources and emits uppercase outputs used by azd.
- `azure.yaml` runs a post-provision hook that syncs provisioned azd values back into local `.env`.
- If `CONTAINER_APPS_ENVIRONMENT_RESOURCE_ID` is not supplied, Bicep now creates the Container Apps managed environment by default.

Typical Azure flow:

1. `azd env new <environment>`
2. `azd env set AZURE_RUNTIME_ENVIRONMENT production`
3. `azd env set POSTGRES_PASSWORD <password>`
4. `azd env set ODDS_API_KEY <value>`
5. `azd env set BASKETBALL_API_KEY <value>`
6. `azd provision`
7. `azd deploy`

After provision, azd writes outputs such as `API_BASE_URL`, `AZURE_KEY_VAULT_URL`, `AZURE_STORAGE_ACCOUNT_URL`, `ACR_NAME`, `API_APP`, `WORKER_APP`, and related stack values into `.azure/<environment>/.env`.

## Production Artifact

The production deployment unit is one ACA image repository:

- Image repository: `nba-gbsv-v6`
- Build once
- Deploy the same image tag to the API Container App
- Deploy the same image tag to the Worker Container App

Model artifacts under `src/models/artifacts/` are included in the built image, which means the ACA runtime is the prediction-model runtime for both services.

## Local Azure Helper

`scripts/deploy.ps1` is the local deployment helper.

It now:

- reads shared Azure names from `infra/stack-config.json`
- builds and pushes the shared production image
- deploys the image to the API and worker ACA apps
- runs migrations using the canonical entry point `python -m src migrate`

## Compatibility Files

`requirements.txt` is a compatibility shim only. All dependencies live in
`pyproject.toml`.

- Runtime install: `pip install .`
- Dev install: `pip install -e ".[dev]"`
