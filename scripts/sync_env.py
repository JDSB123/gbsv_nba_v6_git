from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# src/ is the schema source of truth — Settings declares every env-template field
# via json_schema_extra metadata, and generate_env_template() renders it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from src.config import generate_env_template  # noqa: E402

ENV_LINE_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
PLACEHOLDER_PATTERN = re.compile(r"placeholder", re.IGNORECASE)
REQUIRED_AZD_KEYS = ("ODDS_API_KEY", "BASKETBALL_API_KEY", "DATABASE_URL")


def parse_env_values(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        match = ENV_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        key, value = match.groups()
        values[key] = value
        order.append(key)
    return values, order


def build_synced_lines(
    template_lines: list[str],
    existing_lines: list[str],
    override_values: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    template_values, template_order = parse_env_values(template_lines)
    existing_values, existing_order = parse_env_values(existing_lines)
    override_values = override_values or {}

    synced_lines: list[str] = []
    added_keys: list[str] = []

    for line in template_lines:
        match = ENV_LINE_PATTERN.match(line.strip())
        if not match:
            synced_lines.append(line)
            continue

        key = match.group(1)
        template_value = template_values[key]
        existing_value = existing_values.get(key, "")
        resolved_value = existing_value if existing_value.strip() else template_value
        override_value = override_values.get(key, "")
        if override_value.strip():
            resolved_value = override_value
        if key not in existing_values:
            added_keys.append(key)
        synced_lines.append(f"{key}={resolved_value}")

    extra_keys = [key for key in existing_order if key not in template_order]
    if extra_keys:
        synced_lines.extend(
            [
                "",
                "# Preserved local-only keys",
            ]
        )
        for key in extra_keys:
            synced_lines.append(f"{key}={existing_values[key]}")

    return synced_lines, added_keys


def get_azd_values(environment_name: str | None) -> dict[str, str]:
    try:
        if environment_name:
            result = subprocess.run(
                [
                    "azd",
                    "env",
                    "get-values",
                    "--output",
                    "json",
                    "--environment",
                    environment_name,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                ["azd", "env", "get-values", "--output", "json"],
                check=True,
                capture_output=True,
                text=True,
            )
    except FileNotFoundError as exc:
        raise RuntimeError("azd CLI was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown azd error"
        raise RuntimeError(f"azd env get-values failed: {stderr}") from exc

    payload = result.stdout.strip()
    if not payload:
        return {}

    raw_values = json.loads(payload)
    return {str(key): str(value) for key, value in raw_values.items()}


def resolve_azd_environment_name(explicit_name: str | None) -> str | None:
    if explicit_name and explicit_name.strip():
        return explicit_name.strip()
    env_name = (os.getenv("AZURE_ENV_NAME", "") or "").strip()
    return env_name or None


def ensure_azd_environment_exists(environment_name: str) -> None:
    try:
        subprocess.run(
            ["azd", "env", "new", environment_name],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown azd error"
        raise RuntimeError(f"azd env new failed: {stderr}") from exc


def set_azd_env_value(environment_name: str | None, key: str, value: str) -> None:
    command = ["azd", "env", "set", key, value]
    if environment_name:
        command.extend(["--environment", environment_name])
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown azd error"
        raise RuntimeError(f"azd env set failed for {key}: {stderr}") from exc


def is_real_secret(value: str | None) -> bool:
    if value is None:
        return False
    trimmed = value.strip()
    if not trimmed:
        return False
    return PLACEHOLDER_PATTERN.search(trimmed) is None


def validate_required_azd_values(values: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_AZD_KEYS:
        if not is_real_secret(values.get(key, "")):
            missing.append(key)
    return missing


def collect_local_seed_values(repo_root: Path, output_env_path: Path) -> dict[str, str]:
    candidates: dict[str, str] = {}

    # Highest priority: shell environment values.
    for key in REQUIRED_AZD_KEYS:
        shell_value = os.getenv(key, "")
        if is_real_secret(shell_value):
            candidates[key] = shell_value

    # Next priority: .env and then target output env file.
    for source_path in (repo_root / ".env", output_env_path):
        if not source_path.exists():
            continue
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        source_values, _ = parse_env_values(source_lines)
        for key in REQUIRED_AZD_KEYS:
            if key in candidates:
                continue
            value = source_values.get(key, "")
            if is_real_secret(value):
                candidates[key] = value

    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync .env from the Settings schema in src/config.py without overwriting "
            "non-empty local values. src/config.py is the single schema source — there "
            "is no committed .env template file."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=("Ignore existing values and rewrite from the template plus overrides."),
    )
    parser.add_argument(
        "--from-azd",
        action="store_true",
        help="Overlay values from the current azd environment.",
    )
    parser.add_argument(
        "--environment-name",
        default=None,
        help="Optional azd environment name to query. Defaults to AZURE_ENV_NAME when unset.",
    )
    parser.add_argument(
        "--create-azd-env-if-missing",
        action="store_true",
        help="When used with --from-azd, create the azd environment if it does not exist.",
    )
    parser.add_argument(
        "--allow-incomplete-azd",
        action="store_true",
        help=(
            "Allow writing output even when required azd keys are missing. "
            "By default, missing critical keys are treated as an error."
        ),
    )
    parser.add_argument(
        "--seed-azd-from-local",
        action="store_true",
        help=(
            "When required azd keys are missing, attempt to seed them from local "
            "shell/.env values before failing."
        ),
    )
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    env_path = repo_root / ".env"

    # Template comes from the Settings class in src/config.py, NOT a sibling
    # .env.example file. There is no longer any committed env template on disk.
    template_lines = generate_env_template().splitlines()
    existing_lines = []
    if env_path.exists() and not args.force:
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()
    override_values: dict[str, str] = {}
    if args.from_azd:
        azd_environment_name = resolve_azd_environment_name(args.environment_name)
        try:
            override_values = get_azd_values(azd_environment_name)
        except RuntimeError as exc:
            error_message = str(exc)
            can_create = (
                args.create_azd_env_if_missing
                and azd_environment_name is not None
                and "environment does not exist" in error_message.lower()
            )
            if can_create:
                assert azd_environment_name is not None
                if not args.quiet:
                    print(
                        "azd environment "
                        f"'{azd_environment_name}' not found; creating it because "
                        "--create-azd-env-if-missing was set."
                    )
                try:
                    ensure_azd_environment_exists(azd_environment_name)
                    override_values = get_azd_values(azd_environment_name)
                except RuntimeError as create_exc:
                    print(str(create_exc), file=sys.stderr)
                    return 1
            else:
                print(error_message, file=sys.stderr)
                return 1

        missing_keys = validate_required_azd_values(override_values)
        if missing_keys and args.seed_azd_from_local:
            local_seed_values = collect_local_seed_values(repo_root, env_path)
            seeded_keys: list[str] = []
            for key in missing_keys:
                seed_value = local_seed_values.get(key, "")
                if not is_real_secret(seed_value):
                    continue
                try:
                    set_azd_env_value(azd_environment_name, key, seed_value)
                    seeded_keys.append(key)
                except RuntimeError as seed_exc:
                    print(str(seed_exc), file=sys.stderr)
                    return 1

            if seeded_keys:
                if not args.quiet:
                    print("Seeded missing azd keys from local values: " + ", ".join(seeded_keys))
                try:
                    override_values = get_azd_values(azd_environment_name)
                except RuntimeError as refresh_exc:
                    print(str(refresh_exc), file=sys.stderr)
                    return 1
                missing_keys = validate_required_azd_values(override_values)

        if missing_keys and not args.allow_incomplete_azd:
            display_env = azd_environment_name or "<current>"
            print(
                "azd environment is missing required non-placeholder values: "
                + ", ".join(missing_keys),
                file=sys.stderr,
            )
            print(
                "Set them before syncing, for example:",
                file=sys.stderr,
            )
            print(
                f"  azd env set ODDS_API_KEY <value> --environment {display_env}",
                file=sys.stderr,
            )
            print(
                f"  azd env set BASKETBALL_API_KEY <value> --environment {display_env}",
                file=sys.stderr,
            )
            print(
                f"  azd env set DATABASE_URL <value> --environment {display_env}",
                file=sys.stderr,
            )
            print(
                "If you intentionally need partial sync, re-run with --allow-incomplete-azd.",
                file=sys.stderr,
            )
            return 2
    synced_lines, added_keys = build_synced_lines(
        template_lines,
        existing_lines,
        override_values,
    )
    synced_content = "\n".join(synced_lines) + "\n"
    previous_content = env_path.read_text(encoding="utf-8") if env_path.exists() else None

    if previous_content == synced_content:
        if not args.quiet:
            print(f"{env_path} is already up to date")
        return 0

    env_path.write_text(synced_content, encoding="utf-8")

    if not args.quiet:
        if previous_content is None:
            print(f"Created {env_path}")
        else:
            print(f"Updated {env_path}")
        if added_keys:
            print("Added missing keys: " + ", ".join(added_keys))
        if args.from_azd:
            override_keys = sorted(key for key, value in override_values.items() if value)
            if override_keys:
                print("Overlayed azd keys: " + ", ".join(override_keys))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
