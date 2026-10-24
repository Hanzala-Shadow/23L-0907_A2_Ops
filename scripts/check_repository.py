"""Check Git content for local-only files, credentials and decorative emoji."""

import argparse
import re
import subprocess
from pathlib import PurePosixPath

FORBIDDEN_DIRS = {
    ".agents",
    ".codex",
    ".aws",
    ".claude",
    ".cursor",
    ".gemini",
    ".windsurf",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".dvc-storage",
}
FORBIDDEN_NAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".cursorrules",
    ".windsurfrules",
    "kaggle.json",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
EMOJI = re.compile("[\\U0001F000-\\U0001FAFF\\u2600-\\u27BF]")
AGENT_TRAILER = re.compile(
    r"co-authored-by:.*(?:codex|chatgpt|openai|claude|copilot|gemini|ai agent)",
    re.IGNORECASE,
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".in",
    ".dvc",
    ".html",
    ".css",
    ".js",
}


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def forbidden_path(value):
    path = PurePosixPath(value)
    if set(path.parts) & FORBIDDEN_DIRS or path.name in FORBIDDEN_NAMES:
        return True
    if path.name.startswith(".env") and path.name != ".env.example":
        return True
    if value.startswith(
        (
            ".github/agents/",
            ".github/instructions/",
            "reports/setup/",
            "reports/versioning/",
        )
    ):
        return True
    if value == ".github/copilot-instructions.md":
        return True
    return (
        value.startswith("data/")
        and path.suffix != ".dvc"
        and path.name != ".gitignore"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    paths = set(filter(None, git("ls-files", "-z").decode().split("\0")))
    problems = []
    if args.history:
        changed = git("log", "--all", "--format=", "--name-only", "-z").decode()
        paths.update(
            value.strip("\n") for value in changed.split("\0") if value.strip("\n")
        )
        messages = git("log", "--all", "--format=%B%x00").decode()
        if EMOJI.search(messages) or AGENT_TRAILER.search(messages):
            problems.append(
                "Commit history contains decorative emoji or an agent trailer."
            )
    for path in sorted(paths):
        if forbidden_path(path):
            problems.append(f"Local-only or credential file is tracked: {path}")
    current_paths = filter(None, git("ls-files", "-z").decode().split("\0"))
    for path in current_paths:
        # Read staged bytes, so an unstaged edit cannot hide unwanted content.
        content = git("show", f":{path}")
        suffix = PurePosixPath(path).suffix
        if suffix in TEXT_SUFFIXES or PurePosixPath(path).name.startswith("."):
            text = content.decode("utf-8", errors="replace")
            if EMOJI.search(text):
                problems.append(f"Decorative emoji in text: {path}")
            if PRIVATE_KEY.search(text):
                problems.append(f"Private key material in file: {path}")
        if len(content) > 10 * 1024**2:
            problems.append(f"Large file belongs in DVC: {path}")
    if problems:
        raise SystemExit("\n".join(problems))
    print("Repository content check passed.")


if __name__ == "__main__":
    main()
