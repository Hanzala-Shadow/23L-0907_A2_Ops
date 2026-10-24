"""Run isolated Docker warm-up and missing-model troubleshooting exercises."""

import json
import os
import secrets
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    if shutil.disk_usage(ROOT).free < 2.25 * 1024**3:
        raise RuntimeError("Keep 2 GiB plus lab headroom")
    output = ROOT / "reports/labs/docker"
    runtime = ROOT / "reports/training-runtime/labs/docker"
    output.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    transcript = []
    minimum = shutil.disk_usage(ROOT).free

    def run(*args, check=True):
        nonlocal minimum
        minimum = min(minimum, shutil.disk_usage(ROOT).free)
        if minimum < 2.125 * 1024**3:
            raise RuntimeError("Stopping before disk reserve")
        command = ["sudo", "-n", "docker", *args]
        result = subprocess.run(command, capture_output=True, text=True, check=check)
        transcript.extend(["$ " + " ".join(command), result.stdout + result.stderr])
        return result.stdout.strip()

    db, web, volume = (
        "visionguard-lab-db",
        "visionguard-lab-web",
        "visionguard-lab-data",
    )
    tag = "visionguard-lab-postgres:temporary"
    for name in [db, web]:
        if (
            subprocess.run(
                ["sudo", "-n", "docker", "inspect", name],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError("A lab container already exists; inspect it first")
    if (
        subprocess.run(
            ["sudo", "-n", "docker", "volume", "inspect", volume],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    ):
        raise RuntimeError("A lab volume already exists; inspect it first")
    password, flask_key = secrets.token_hex(24), secrets.token_hex(24)
    env = runtime / "settings.env"
    env.write_text(
        f"POSTGRES_DB=lab\nPOSTGRES_USER=lab\nPOSTGRES_PASSWORD={password}\n"
        f"FLASK_SECRET_KEY={flask_key}\nDATABASE_URL=postgresql+psycopg://lab:{password}@127.0.0.1:5432/lab\n"
        "MODEL_PATH=models/champion/best.keras\nMODEL_POLICY_PATH=models/champion/policy.json\n"
        "DECISION_THRESHOLD=0.6196606457233429\nDISK_RESERVE_GIB=2\n"
    )
    os.chmod(env, 0o600)
    run("image", "tag", "postgres:17-alpine", tag)
    run("volume", "create", volume)
    try:
        run(
            "run",
            "-d",
            "--name",
            db,
            "--env-file",
            str(env),
            "--publish",
            "127.0.0.1:5544:5432",
            "--publish",
            "127.0.0.1:5500:8000",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "--memory",
            "256m",
            tag,
            "postgres",
            "-c",
            "max_wal_size=64MB",
            "-c",
            "min_wal_size=32MB",
            "-c",
            "shared_buffers=32MB",
        )
        for _ in range(30):
            if run(
                "exec", db, "pg_isready", "-U", "lab", "-d", "lab", check=False
            ).endswith("accepting connections"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Lab database did not start")
        run("exec", db, "sh", "-c", "id; ls -ld /var/lib/postgresql/data")
        run("logs", "--tail", "12", db)
        run("ps", "--filter", "name=visionguard-lab")
        run("stop", db)
        run("start", db)
        run("restart", db)
        time.sleep(2)
        common = [
            "run",
            "-d",
            "--name",
            web,
            "--network",
            f"container:{db}",
            "--env-file",
            str(env),
            "--read-only",
            "--tmpfs",
            "/tmp:size=128m",
            "--memory",
            "1g",
        ]
        run(*common, "visionguard:local")
        for _ in range(30):
            state = run("inspect", "--format", "{{.State.Status}}", web)
            if state == "exited":
                break
            time.sleep(1)
        failure = run("logs", "--tail", "45", web)
        # Docker sends logs to stderr too; retain the actual combined output.
        failure = transcript[-1]
        assert "FileNotFoundError" in failure and "policy.json" in failure
        (output / "missing-model.log").write_text(failure)
        run("rm", web)
        run(
            *common,
            "--mount",
            f"type=bind,source={ROOT / 'models/champion'},target=/app/models/champion,readonly",
            "visionguard:local",
        )
        for _ in range(45):
            try:
                response = requests.get("http://127.0.0.1:5500/health", timeout=3)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Corrected native app did not become healthy")
        health = response.json()
        assert health["model"]["loaded"] and health["database"]["status"] == "ok"
        run("exec", web, "sh", "-c", "id; pwd; test -r models/champion/policy.json")
        run("logs", "--tail", "12", web)
        run("stop", web)
        run("start", web)
        run("restart", web)
        sizes = run("image", "ls", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}")
        layers = {}
        for image in ["visionguard:local", "visionguard:client"]:
            layers[image] = run(
                "history", "--no-trunc", "--format", "{{.Size}} | {{.CreatedBy}}", image
            )
        run("ps", "-a", "--filter", "name=visionguard-lab")
    finally:
        for name in [web, db]:
            run("rm", "-f", name, check=False)
        run("volume", "rm", volume)
        run("image", "rm", tag)
    assert shutil.disk_usage(ROOT).free >= 2 * 1024**3
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "ready_made_image": "postgres:17-alpine",
        "actions": [
            "run",
            "port mapping",
            "named volume",
            "logs",
            "shell",
            "list",
            "stop",
            "start",
            "restart",
            "remove container",
            "remove temporary image tag",
        ],
        "failure": "missing mounted champion policy.json",
        "diagnosis": "FileNotFoundError in Gunicorn worker log",
        "fix": "read-only mount of the existing champion bundle",
        "corrected_health": health,
        "image_sizes_same_docker_ls_command": sizes.splitlines(),
        "layers": layers,
        "minimum_observed_free_gib": minimum / 1024**3,
        "free_after_gib": shutil.disk_usage(ROOT).free / 1024**3,
        "production_containers_or_volume_changed": False,
        "temporary_resources_removed": True,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    commands = "\n".join(transcript)
    assert password not in commands and flask_key not in commands
    (output / "commands.txt").write_text(
        "\n".join(x.rstrip() for x in commands.splitlines()).rstrip() + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
