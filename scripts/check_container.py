"""Verify the live Compose stack, image contents and preserved history."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--phase", choices=("initial", "final"), default="final")
    parser.add_argument(
        "--before-snapshot", default="reports/training-runtime/docker/before.json"
    )
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in ("localhost", "127.0.0.1"):
        parser.error("Verification is limited to the local deployment")
    prefix = ["sudo", "-n"] if args.sudo else []

    def docker(*parts):
        return subprocess.check_output(prefix + ["docker", *parts], cwd=ROOT)

    health_response = requests.get(args.base_url + "/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    assert health["database"] == {"backend": "postgresql", "status": "ok"}
    assert health["model"]["version"] == "2" and health["debug"] is False
    assert health["model"]["threshold"] == 0.6196606457233429
    settings = dotenv_values(ROOT / ".env")
    web_id = docker("compose", "ps", "-q", "web").decode().strip()
    db_id = docker("compose", "ps", "-q", "db").decode().strip()
    web = json.loads(docker("inspect", web_id))[0]
    db = json.loads(docker("inspect", db_id))[0]
    assert web["State"]["Health"]["Status"] == "healthy"
    assert db["State"]["Health"]["Status"] == "healthy"
    assert web["Config"]["User"] == "10001:10001"
    assert web["HostConfig"]["ReadonlyRootfs"]
    assert "ALL" in web["HostConfig"]["CapDrop"]
    assert "no-new-privileges:true" in web["HostConfig"]["SecurityOpt"]
    assert all(
        binding["HostIp"] == "127.0.0.1"
        for bindings in web["HostConfig"]["PortBindings"].values()
        for binding in bindings
    )
    environment = dict(value.split("=", 1) for value in web["Config"]["Env"])
    assert "MLFLOW_TRACKING_PASSWORD" not in environment
    assert "MLFLOW_TRACKING_USERNAME" not in environment
    assert environment["DATABASE_URL"] == settings["DATABASE_URL_CONTAINER"]
    model_mount = next(
        mount
        for mount in web["Mounts"]
        if mount["Destination"] == "/app/models/champion"
    )
    assert model_mount["Type"] == "bind" and not model_mount["RW"]
    assert Path(model_mount["Source"]) == ROOT / "models/champion"
    history_mount = next(
        mount
        for mount in db["Mounts"]
        if mount["Destination"] == "/var/lib/postgresql/data"
    )
    assert history_mount["Name"] == settings["POSTGRES_VOLUME_NAME"]
    image = json.loads(docker("image", "inspect", "visionguard:local"))[0]
    assert web["Image"] == image["Id"]
    audit_code = """
import importlib.util, json, os, pathlib, sys
root=pathlib.Path('/app')
excluded=['.env','.git','.agents','.codex','.aws','data','reports','tests','models/champion/best.keras']
assert all(not (root/path).exists() for path in excluded)
assert all(importlib.util.find_spec(name) is None for name in ['dvc','mlflow','dagshub','pytest'])
assert not (root/'visionguard/cnn.py').exists()
assert not list(root.rglob('*.pyc'))
assert not (root/'requirements/locked.txt').exists()
assert not pathlib.Path('/usr/local/lib/python3.12/site-packages/tensorflow/include').exists()
assert os.getuid()==10001
print(json.dumps({'uid':os.getuid(),'python':sys.version.split()[0], 'excluded_files_absent':True,'training_tools_absent':True,'model_not_baked_in':True,'app_files':sorted(str(path.relative_to(root)) for path in root.rglob('*') if path.is_file())}))
"""
    image_audit = json.loads(
        docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:size=32m",
            "--entrypoint",
            "python",
            "visionguard:local",
            "-c",
            audit_code,
        )
    )
    model_hash = digest(ROOT / "models/champion/best.keras")
    policy_hash = digest(ROOT / "models/champion/policy.json")
    policy = json.loads((ROOT / "models/champion/policy.json").read_text())
    assert model_hash == policy["model_sha256"]
    mounted_hash = (
        docker(
            "compose",
            "exec",
            "-T",
            "web",
            "python",
            "-c",
            "import hashlib; print(hashlib.sha256(open('/app/models/champion/best.keras','rb').read()).hexdigest())",
        )
        .decode()
        .strip()
    )
    assert mounted_hash == model_hash
    engine = create_engine(settings["DATABASE_URL"], hide_parameters=True)
    with engine.connect() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM predictions ORDER BY id")
            ).mappings()
        ]
    engine.dispose()
    rows = json.loads(json.dumps(rows, default=str))
    before_path = ROOT / args.before_snapshot
    before = json.loads(before_path.read_text()) if before_path.exists() else None
    preserved = 0
    if before:
        current = {row["id"]: row for row in rows}
        assert all(current.get(row["id"]) == row for row in before["rows"])
        assert before["model_sha256"] == model_hash
        assert before["policy_sha256"] == policy_hash
        preserved = len(before["rows"])
    free = shutil.disk_usage(ROOT).free / 1024**3
    assert free >= 2 and health["disk"]["free_gib"] >= 2
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "base_url": args.base_url,
        "health": health,
        "web_container": web_id[:12],
        "database_container": db_id[:12],
        "history_volume": history_mount["Name"],
        "history_rows": len(rows),
        "original_rows_verified_unchanged": preserved,
        "model_sha256": model_hash,
        "policy_sha256": policy_hash,
        "model_mount_read_only": True,
        "root_filesystem_read_only": True,
        "image_id": image["Id"],
        "image_size_bytes": image["Size"],
        "image_audit": image_audit,
        "free_after_gib": free,
        "test_images_used": False,
    }
    output = ROOT / "reports/container"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.phase}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
