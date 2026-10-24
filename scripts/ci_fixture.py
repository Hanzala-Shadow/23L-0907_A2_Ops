"""Create an isolated model or configuration for disposable CI checks."""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

THRESHOLD = 0.62


def settings(path):
    if path.exists():
        raise FileExistsError("CI settings must not replace an existing environment")
    password = secrets.token_urlsafe(32)
    values = {
        "FLASK_SECRET_KEY": secrets.token_urlsafe(48),
        "POSTGRES_DB": "visionguard_ci",
        "POSTGRES_USER": "visionguard",
        "POSTGRES_PASSWORD": password,
        "DATABASE_URL_CONTAINER": f"postgresql+psycopg://visionguard:{password}@db:5432/visionguard_ci",
        "POSTGRES_VOLUME_NAME": os.environ.get(
            "POSTGRES_VOLUME_NAME", "visionguard-ci"
        ),
        "DECISION_THRESHOLD": str(THRESHOLD),
        "WEB_PORT": "5000",
        "POSTGRES_PORT": "5433",
        "DISK_RESERVE_GIB": "2",
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write("".join(f"{key}={value}\n" for key, value in values.items()))


def model_bundle(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("CI models must use an empty directory")
    import numpy as np
    import tensorflow as tf

    from visionguard.preprocessing import POLICY_VERSION

    path.mkdir(parents=True, exist_ok=True)
    inputs = tf.keras.Input(shape=(16, 16, 3), dtype="uint8")
    scaled = tf.keras.layers.Rescaling(1 / 255)(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(scaled)
    output = tf.keras.layers.Dense(1, activation="sigmoid")
    model = tf.keras.Model(inputs, output(features))
    output.set_weights(
        [np.array([[-4], [0], [4]], dtype=np.float32), np.zeros(1, dtype=np.float32)]
    )
    checkpoint = path / "best.keras"
    model.save(checkpoint)
    policy = {
        "registry_name": "23L-0907_A2_Ops-ci",
        "registry_version": "synthetic",
        "run_id": "ci-synthetic",
        "threshold": THRESHOLD,
        "preprocessing": {"policy": POLICY_VERSION, "image_size": 16},
        "labels": {"real": 0, "ai_generated": 1},
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (path / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("settings", "model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (settings if args.kind == "settings" else model_bundle)(args.output)
    print(f"Created synthetic CI {args.kind} fixture.")


if __name__ == "__main__":
    main()
