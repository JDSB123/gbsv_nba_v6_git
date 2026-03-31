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
    assert _non_comment_lines(ROOT / "requirements.txt") == ["."]
    assert _non_comment_lines(ROOT / "requirements-dev.txt") == ["-e .[dev]"]


def test_repo_uses_single_committed_env_template():
    assert (ROOT / ".gitattributes").exists()
    assert (ROOT / ".env.example").exists()
    assert not (ROOT / ".env.smoke").exists()
    assert not (ROOT / ".env.production").exists()
    assert not (ROOT / ".env.local").exists()
    assert not (ROOT / ".env.test").exists()
    assert Settings.model_config["env_file"] == ".env"


def test_readme_describes_single_dev_workflow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "one default development workflow" in readme.lower()
    assert "One dependency source: `pyproject.toml`" in readme
    assert "`master` is a protected branch" in readme
    assert "pull request" in readme.lower()
