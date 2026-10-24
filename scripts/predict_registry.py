"""Resolve a registry alias, load the model once, and classify one image."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from scripts.train_cnn import check_space, configure_cpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    check_space(ROOT, margin_gib=0.25)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import mlflow

    from visionguard.inference import load_by_alias

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
    with tempfile.TemporaryDirectory(prefix="visionguard-registry-") as temporary:
        predictor, version = load_by_alias(
            os.environ["PROJECT_NAME"], args.alias, temporary
        )
        print(
            json.dumps(
                {
                    "registry_name": version.name,
                    "version": version.version,
                    "alias": args.alias,
                    "run_id": version.run_id,
                    **predictor.predict(args.image),
                },
                indent=2,
            )
        )
    check_space(ROOT)


if __name__ == "__main__":
    main()
