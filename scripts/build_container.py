"""Build the serving image while monitoring the shared filesystem reserve."""

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sudo", action="store_true", help="Use sudo -n for Docker")
    parser.add_argument("--minimum-start-gib", type=float, default=4)
    parser.add_argument("--stop-free-gib", type=float, default=2.5)
    args = parser.parse_args()
    if not 2.5 <= args.stop_free_gib < args.minimum_start_gib:
        parser.error("Keep a stop buffer of at least 2.5 GiB below the start limit")
    start_free = shutil.disk_usage(ROOT).free
    if start_free < args.minimum_start_gib * GIB:
        raise SystemExit("Not enough free space for a guarded build; no build started")
    runtime = ROOT / "reports/training-runtime/docker"
    runtime.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports/container"
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = ["sudo", "-n"] if args.sudo else []
    command = prefix + [
        "docker",
        "buildx",
        "build",
        "--progress",
        "plain",
        "--tag",
        "visionguard:local",
        "--output",
        "type=image,unpack=false",
        ".",
    ]
    minimum = start_free
    stopped = False
    started = time.monotonic()
    print(
        f"Building with a {args.stop_free_gib:g} GiB stop buffer; log: {runtime / 'build.log'}",
        flush=True,
    )
    with (runtime / "build.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                free = shutil.disk_usage(ROOT).free
                minimum = min(minimum, free)
                if free < args.stop_free_gib * GIB:
                    stopped = True
                    subprocess.run(
                        prefix + ["kill", "-TERM", "--", f"-{process.pid}"], check=True
                    )
                    break
                time.sleep(0.1)
            code = process.wait(timeout=45)
        except BaseException:
            if process.poll() is None:
                subprocess.run(
                    prefix + ["kill", "-TERM", "--", f"-{process.pid}"], check=False
                )
            raise
    end_free = shutil.disk_usage(ROOT).free
    minimum = min(minimum, end_free)
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "command": "docker buildx build --tag visionguard:local --output type=image,unpack=false .",
        "return_code": code,
        "stopped_for_disk": stopped,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "reserve_gib": 2,
        "stop_free_gib": args.stop_free_gib,
        "start_free_gib": start_free / GIB,
        "minimum_observed_free_gib": minimum / GIB,
        "end_free_gib": end_free / GIB,
    }
    (report_dir / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if code or stopped or minimum < 2 * GIB:
        raise SystemExit("Build did not complete safely; inspect the local build log")


if __name__ == "__main__":
    main()
