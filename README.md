# NBA GBSV v6

Container-first Python service for NBA predictions, backfills, model training, and Teams publishing.

## Dev Stack

This repo uses one default development workflow:

1. Open the repo in the VS Code dev container.
2. Copy `.env.example` to `.env` and add real API keys if you need live API calls.
3. Run `python -m src migrate`.
4. Use `python -m src serve`, `python -m src work`, `python -m src train`, or `pytest`.

The dev container provides the Python runtime, system packages, and dev tools. We do not rely on a local `.venv` as the primary workflow.

## Fresh Clone Path

The supported onboarding path is:

1. Clone the repo.
2. Open it in VS Code.
3. Reopen in the dev container.

On first container creation, the repo now:

- installs dependencies from `pyproject.toml`
- creates `.env` from `.env.example` if it is missing
- injects local Postgres and placeholder API keys for development
- runs `python -m src migrate`

That means a fresh clone is ready for local boot and tests without hand-building a virtualenv. Live external API calls still need real API keys.

## Why This Is Sustainable

This setup is sustainable because the same install source and entrypoints are used across local development, CI, and production packaging.

- One dependency source: `pyproject.toml`
- One default dev path: VS Code dev container
- One command family: `python -m src ...`
- CI smoke-checks a fresh-checkout install, migration, and app import

## Change Flow

`master` is a protected branch. Changes should land through a branch and pull request, not direct pushes.

The expected flow is:

1. Branch from `master`.
2. Make the change with `pyproject.toml`, `.env.example`, and this README as the only repo-level sources of truth.
3. Run `ruff check src tests` and `pytest`.
4. Open a pull request and merge only after the required GitHub checks pass.

## Dependency Source Of Truth

`pyproject.toml` is the canonical source for Python dependencies and tooling configuration.

- Runtime install: `pip install .`
- Dev install: `pip install -e ".[dev]"`
- `requirements.txt` and `requirements-dev.txt` remain as compatibility shims only.

## Environment

Use `.env` for local secrets and overrides.

- The dev container injects `DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/nba_gbsv`.
- Host-only runs can keep `DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/nba_gbsv`.
- Production uses Azure-injected environment variables rather than repo-local env files.

Optional Windows helper:

- `powershell -ExecutionPolicy Bypass -File .\scripts\setup-env.ps1`

That script prepares `.env`. It does not create a virtual environment or install Python.
