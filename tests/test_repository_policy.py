"""Keep private files and local tooling out of the repository."""

import pytest

from scripts.check_repository import EMOJI, forbidden_path


@pytest.mark.parametrize(
    "path",
    [
        ".agents/session.json",
        "AGENTS.md",
        ".codex/config.toml",
        ".env",
        ".aws/credentials",
        ".github/agents/review.md",
        "data/raw/image.jpg",
    ],
)
def test_local_files_are_rejected(path):
    assert forbidden_path(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "data/raw.dvc",
        "data/.gitignore",
        ".github/workflows/ci.yml",
        "scripts/prepare_splits.py",
        "reports/data/width_histogram.png",
    ],
)
def test_project_files_are_allowed(path):
    assert not forbidden_path(path)


def test_decorative_emoji_is_rejected():
    assert EMOJI.search("Prepare data \N{ROCKET}")
    assert not EMOJI.search("Prepare data")
