"""Check the shared Linux filesystem before a download, training run or build."""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error("minimum free space must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    free = shutil.disk_usage(root).free / 1024**3
    print(f"Filesystem free: {free:.2f} GiB")
    print(f"Required reserve: {args.minimum_free_gib:.2f} GiB")
    paths = [
        root / name for name in (".venv", "data", ".dvc/cache", "models", "reports")
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        subprocess.run(["du", "-sh", *existing], check=True)
    if free < args.minimum_free_gib:
        raise SystemExit("STOP: make space before starting another large operation.")
    print("Disk reserve check passed.")


if __name__ == "__main__":
    main()
