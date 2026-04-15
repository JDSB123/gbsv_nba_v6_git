from __future__ import annotations

from pathlib import Path

from src.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_pyproject_is_dependency_source_of_truth():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'readme = "README.md"' in pyproject
    assert "dependencies = [" in pyproject
    assert "[project.optional-dependencies]" in pyproject
    assert _non_comment_lines(ROOT / "requirements.txt") == ["-e .[dev]"]
    assert not (ROOT / "requirements-dev.txt").exists()


def test_repo_uses_single_env_schema_source():
    assert (ROOT / ".gitattributes").exists()
    # src/config.py (Settings) is the single env schema source — there is NO
    # committed .env template file. scripts/sync_env.py renders .env from
    # Settings.generate_env_template().
    forbidden_env_files = (
        ".env.example",
        ".env.smoke",
        ".env.production",
        ".env.local",
        ".env.test",
        ".env.azure",
        ".env.profile",
    )
    for name in forbidden_env_files:
        assert not (ROOT / name).exists(), f"forbidden env file present: {name}"
    assert Settings.model_config["env_file"] == ".env"

    # The generator must declare every required runtime key.
    from src.config import generate_env_template

    rendered = generate_env_template()
    for required_key in ("ODDS_API_KEY", "BASKETBALL_API_KEY", "DATABASE_URL"):
        assert f"{required_key}=" in rendered, f"missing required env key: {required_key}"


def test_readme_describes_single_dev_workflow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "one default development workflow" in readme.lower()
    assert "One dependency source: `pyproject.toml`" in readme
    assert "`master` is a protected branch" in readme
    assert "pull request" in readme.lower()
