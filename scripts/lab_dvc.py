"""Demonstrate two tiny DVC versions and a clean-cache remote restore."""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert shutil.disk_usage(ROOT).free > 2.125 * 1024**3
    base = ROOT / "reports/training-runtime/labs/dvc"
    repo, remote = base / "repository", base / "remote"
    if base.exists():
        raise FileExistsError("Use a fresh lab directory to preserve its history")
    repo.mkdir(parents=True)
    remote.mkdir()
    transcript = []

    def run(*args):
        p = subprocess.run(args, cwd=repo, text=True, capture_output=True, check=True)
        transcript.append("$ " + " ".join(args).replace(str(ROOT), "."))
        transcript.append(p.stdout + p.stderr)
        return p.stdout.strip()

    run("git", "init", "-b", "dvc-lab")
    for key in ["user.name", "user.email"]:
        value = subprocess.check_output(
            ["git", "config", key], cwd=ROOT, text=True
        ).strip()
        run("git", "config", key, value)
    dvc = str(ROOT / ".venv/bin/dvc")
    run(dvc, "init")
    run(dvc, "remote", "add", "-d", "storage", "../remote")
    initial = b"image,label\nphoto.jpg,real\nart.png,ai_generated\n"
    updated = initial + b"landscape.jpg,real\n"
    target = repo / "examples.csv"
    versions = []
    for content, message in [
        (initial, "Track the initial example data"),
        (updated, "Add a landscape example"),
    ]:
        target.write_bytes(content)
        run(dvc, "add", "examples.csv")
        run("git", "add", ".dvc", ".dvcignore", ".gitignore", "examples.csv.dvc")
        run("git", "commit", "-m", message)
        versions.append(run("git", "rev-parse", "HEAD"))
        run(dvc, "push")
    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True
    ).strip()
    run("git", "remote", "add", "origin", origin)
    # A dedicated branch in the existing private project avoids an extra account
    # or repository. Only toy pointers and DVC configuration are published.
    run("git", "push", "origin", "dvc-lab")
    run("git", "checkout", versions[0])
    run(dvc, "checkout")
    assert target.read_bytes() == initial
    target.unlink()
    shutil.rmtree(repo / ".dvc/cache")
    run(dvc, "pull")
    assert target.read_bytes() == initial
    status = run(dvc, "status")
    pointer = (repo / "examples.csv.dvc").read_text()
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "branch": "dvc-lab",
        "version_commits": versions,
        "restored_commit": versions[0],
        "restored_sha256": hashlib.sha256(initial).hexdigest(),
        "remote": "local folder outside the toy repository",
        "local_workspace_and_cache_deleted_before_pull": True,
        "restored_content_matches": True,
        "status": status,
        "pointer": pointer,
        "free_disk_gib": shutil.disk_usage(ROOT).free / 1024**3,
        "production_data_changed": False,
    }
    output = ROOT / "reports/labs/dvc"
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "commands.txt").write_text(
        "\n".join(line.rstrip() for line in "\n".join(transcript).splitlines()).rstrip()
        + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ["DVC_NO_ANALYTICS"] = "true"
    main()
