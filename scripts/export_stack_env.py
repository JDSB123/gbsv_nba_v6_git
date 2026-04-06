from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_stack_config() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "infra" / "stack-config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_variables(config: dict[str, object]) -> dict[str, str]:
    container_apps = config["containerApps"]
    if not isinstance(container_apps, dict):
        raise ValueError(
            "containerApps must be an object in stack-config.json"
        )

    registry_name = str(config["registryName"])

    return {
        "STACK_APPLICATION_NAME": str(config["applicationName"]),
        "IMAGE_NAME": str(config["imageRepository"]),
        "RESOURCE_GROUP": str(config["resourceGroupName"]),
        "ACR_NAME": registry_name,
        "ACR_LOGIN_SERVER": f"{registry_name}.azurecr.io",
        "API_APP": str(container_apps["api"]),
        "WORKER_APP": str(container_apps["worker"]),
        "PG_SERVER_NAME": str(config["postgresServerName"]),
        "KV_NAME": str(config["keyVaultName"]),
        "ACA_ENVIRONMENT_NAME": str(config["containerAppsEnvironmentName"]),
    }


def write_env_file(path: str, variables: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in variables.items()]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_output_file(path: str, variables: dict[str, str]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in variables.items():
            handle.write(f"{key.lower()}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export shared stack variables from infra/stack-config.json"
        )
    )
    parser.add_argument(
        "--github-env",
        default=None,
        help="Append uppercase variables to the provided GitHub env file.",
    )
    parser.add_argument(
        "--github-output",
        default=None,
        help=(
            "Append lowercase step outputs to the provided GitHub output file."
        ),
    )
    args = parser.parse_args()

    variables = build_variables(load_stack_config())

    if args.github_env:
        write_env_file(args.github_env, variables)
    if args.github_output:
        write_output_file(args.github_output, variables)

    if not args.github_env and not args.github_output:
        print(json.dumps(variables, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
