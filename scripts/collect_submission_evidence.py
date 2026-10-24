"""Create read-only HTML views of actual recorded command output for screenshots."""

import getpass
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert shutil.disk_usage(ROOT).free > 2.125 * 1024**3
    site = ROOT / "reports/training-runtime/submission/evidence-site"
    site.mkdir(parents=True, exist_ok=True)
    snippets = {}

    def command(*args):
        result = subprocess.run(
            args, cwd=ROOT, text=True, capture_output=True, check=True
        )
        return "$ " + " ".join(args) + "\n" + result.stdout

    def recorded(path):
        return command("cat", path)

    snippets["environment"] = command(
        str(ROOT / ".venv/bin/python"),
        "-c",
        "import json; e=json.load(open('reports/setup/environment.json')); "
        "print(json.dumps({k:e[k] for k in ['timestamp','user','python_version',"
        "'tensorflow_cpu_devices','tensorflow_gpu_devices','packages']},indent=2))",
    )
    snippets["model-summary"] = recorded(
        "reports/training/cnn_baseline/model_summary.txt"
    )
    snippets["registry-load"] = recorded("reports/registry/alias_prediction.json")
    snippets["dvc-lab"] = command(
        "git",
        "-C",
        "reports/training-runtime/labs/dvc/repository",
        "log",
        "--all",
        "--oneline",
    ) + recorded("reports/labs/dvc/verification.json")
    docker = json.loads((ROOT / "reports/labs/docker/verification.json").read_text())
    snippets["docker-warmup"] = "$ python -m scripts.lab_docker\n" + json.dumps(
        {
            k: docker[k]
            for k in [
                "checked_at_utc",
                "ready_made_image",
                "actions",
                "minimum_observed_free_gib",
                "temporary_resources_removed",
                "production_containers_or_volume_changed",
            ]
        },
        indent=2,
    )
    failure = (ROOT / "reports/labs/docker/missing-model.log").read_text().splitlines()
    selected = [
        x
        for x in failure
        if any(
            word in x
            for word in [
                "policy.json",
                "FileNotFoundError",
                "Worker failed",
                "Worker exiting",
            ]
        )
    ]
    snippets["docker-failure"] = (
        "$ docker logs visionguard-lab-web (recorded before cleanup)\n"
        + "\n".join(selected)
    )
    snippets["docker-failure"] += (
        "\n\nCorrected read-only bundle mount, recorded health:\n"
        + json.dumps(docker["corrected_health"], indent=2)
    )
    snippets["docker-stack"] = command("sudo", "-n", "docker", "compose", "ps")
    containers = json.loads(
        subprocess.check_output(
            [
                "sudo",
                "-n",
                "docker",
                "inspect",
                *subprocess.check_output(
                    ["sudo", "-n", "docker", "compose", "ps", "-q"], text=True
                ).split(),
            ],
            text=True,
        )
    )
    facts = []
    for c in containers:
        facts.append(
            {
                "service": c["Config"]["Labels"]["com.docker.compose.service"],
                "user": c["Config"]["User"],
                "ports": c["HostConfig"]["PortBindings"],
                "read_only_root": c["HostConfig"]["ReadonlyRootfs"],
                "mounts": [
                    {
                        "type": m["Type"],
                        "destination": m["Destination"],
                        "read_only": not m["RW"],
                    }
                    for m in c["Mounts"]
                ],
            }
        )
    snippets["docker-stack"] += (
        "\nSelected inspect fields (environment excluded):\n"
        + json.dumps(facts, indent=2)
    )
    integration = json.loads(
        (ROOT / "reports/integration/verification.json").read_text()
    )
    snippets["integration-results"] = (
        "$ python scripts/check_integration.py (recorded)\n"
        + json.dumps(
            {
                "local_url": integration["local_url"],
                "cloud_url": integration["cloud_url"],
                "live_requests": integration["live_requests"][:3],
                "original_rows_preserved": integration["original_rows_preserved"],
                "test_set_used": False,
            },
            indent=2,
        )
    )
    snippets["postman-output"] = recorded("reports/labs/postman/commands.txt")
    health = requests.get("http://127.0.0.1:5000/health", timeout=10)
    health.raise_for_status()
    snippets["local-health-output"] = (
        "$ curl --fail http://127.0.0.1:5000/health\n"
        + json.dumps(health.json(), indent=2)
    )
    timestamp = datetime.now(ZoneInfo("Asia/Karachi")).isoformat(timespec="seconds")
    for name, text in snippets.items():
        page = (
            """<!doctype html><html lang="en"><meta charset="utf-8"><title>23L-0907_A2_Ops evidence</title>
<style>body{margin:28px;background:#f7f8fa;color:#172b36;font:15px sans-serif}h1{font-size:22px;margin:0 0 8px}p{margin:6px 0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border:1px solid #cbd5df;border-radius:8px;font:13px/1.45 monospace}</style>
<h1>23L-0907_A2_Ops - """
            + html.escape(name)
            + "</h1><p>Syed Muhammad Hanzala | user "
            + getpass.getuser()
            + " | "
            + timestamp
            + "</p><p>Recorded command output. Read-only transcript view, not a live shell.</p><pre>"
            + html.escape(text)
            + "</pre></html>"
        )
        if name == "model-summary":
            page = page.replace("font:13px/1.45 monospace", "font:11px/1.25 monospace")
        (site / (name + ".html")).write_text(page)
    print("Prepared", len(snippets), "credential-free transcript views")


if __name__ == "__main__":
    main()
