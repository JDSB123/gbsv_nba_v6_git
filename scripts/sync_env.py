from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ENV_LINE_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Sync .env from .env.example without overwriting non-empty local values.")
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
        help="Optional azd environment name to query.",
    )
    parser.add_argument(
        "--output",
        default=".env",
        help="Repo-relative or absolute path of the env file to write.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    template_path = repo_root / ".env.example"
    env_path = Path(args.output)
    if not env_path.is_absolute():
        env_path = repo_root / env_path

    if not template_path.exists():
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 1

    template_lines = template_path.read_text(encoding="utf-8").splitlines()
    existing_lines = []
    if env_path.exists() and not args.force:
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()
    override_values: dict[str, str] = {}
    if args.from_azd:
        try:
            override_values = get_azd_values(args.environment_name)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    elif not args.force:
        azure_path = repo_root / ".env.azure"
        if azure_path.exists():
            azure_lines = azure_path.read_text(encoding="utf-8").splitlines()
            azure_values, _ = parse_env_values(azure_lines)
            for k, v in azure_values.items():
                if v and "placeholder" not in v:
                    override_values[k] = v

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
