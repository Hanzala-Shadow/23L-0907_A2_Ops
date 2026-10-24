"""Record the installed environment and optionally exercise the ML stack.

Run from the repository root with .venv/bin/python scripts/check_environment.py
--smoke. The smoke check uses synthetic data, temporary files and local MLflow;
it does not contact remote accounts or evaluate an assignment model.
"""

import argparse
import datetime as dt
import getpass
import importlib
import json
import os
import platform
import shutil
import tempfile
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def smoke_check(tf):
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    from flask import Flask
    from sqlalchemy import create_engine, text

    tf.keras.utils.set_random_seed(42)
    x = np.random.default_rng(42).normal(size=(16, 4)).astype("float32")
    y = (x[:, :1] > 0).astype("float32")
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    with tempfile.TemporaryDirectory(prefix="visionguard-smoke-") as temporary:
        directory = Path(temporary)
        mlflow.set_tracking_uri(f"sqlite:///{directory / 'mlflow.db'}")
        experiment_id = mlflow.MlflowClient().create_experiment(
            "environment-smoke-check", artifact_location=str(directory / "artifacts")
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        mlflow.tensorflow.autolog(log_models=False)
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(4,)),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        with mlflow.start_run() as run:
            model.fit(dataset, epochs=1, verbose=0)
            mlflow.tensorflow.log_model(
                model,
                artifact_path="model",
                input_example=x[:2],
                pip_requirements=[
                    f"tensorflow-cpu=={metadata.version('tensorflow-cpu')}"
                ],
            )
            run_id = run.info.run_id
        recorded = mlflow.MlflowClient().get_run(run_id)
        if not any("loss" in key for key in recorded.data.metrics):
            raise RuntimeError("MLflow autologging did not record training loss")
        restored = mlflow.tensorflow.load_model(f"runs:/{run_id}/model")
        np.testing.assert_allclose(
            model(x, training=False).numpy(),
            restored(x, training=False).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        keras_path = directory / "roundtrip.keras"
        model.save(keras_path)
        keras_restored = tf.keras.models.load_model(keras_path)
        np.testing.assert_allclose(
            model(x, training=False).numpy(),
            keras_restored(x, training=False).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        mlflow.tensorflow.autolog(disable=True)

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    with app.test_client() as client:
        response = client.get("/health")
        if response.status_code != 200 or response.json != {"status": "ok"}:
            raise RuntimeError("Flask test request failed")
    with create_engine("sqlite:///:memory:").connect() as connection:
        if connection.execute(text("SELECT 1")).scalar_one() != 1:
            raise RuntimeError("SQLAlchemy test query failed")

    return {
        "synthetic_keras_training": "passed",
        "mlflow_autologging": "passed",
        "mlflow_model_save_load": "passed",
        "keras_model_save_load": "passed",
        "flask_test_request": "passed",
        "sqlalchemy_local_query": "passed",
        "note": "PostgreSQL and remote services require separate checks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    for name, value in {
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "MPLCONFIGDIR": str(ROOT / "reports/setup/.matplotlib"),
        "KERAS_HOME": str(ROOT / "reports/setup/.keras"),
    }.items():
        os.environ[name] = value

    distributions = [
        "tensorflow-cpu",
        "keras",
        "keras-tuner",
        "numpy",
        "pandas",
        "Pillow",
        "matplotlib",
        "scikit-learn",
        "mlflow",
        "dagshub",
        "dvc",
        "Flask",
        "gunicorn",
        "SQLAlchemy",
        "psycopg",
        "python-dotenv",
        "requests",
        "kaggle",
        "pytest",
        "ruff",
    ]
    versions = {name: metadata.version(name) for name in distributions}
    for module in [
        "keras_tuner",
        "pandas",
        "PIL",
        "matplotlib",
        "sklearn",
        "mlflow",
        "dagshub",
        "dvc",
        "flask",
        "sqlalchemy",
        "psycopg",
        "dotenv",
        "requests",
        "pytest",
    ]:
        importlib.import_module(module)
    import tensorflow as tf

    installed = {
        distribution.metadata["Name"].lower()
        for distribution in metadata.distributions()
    }
    unexpected = sorted(
        name for name in installed if name.startswith("nvidia-") or name == "tensorflow"
    )
    if unexpected:
        raise RuntimeError(f"Unexpected GPU/full TensorFlow packages: {unexpected}")

    disk = shutil.disk_usage(ROOT)
    report = {
        "timestamp": dt.datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "user": getpass.getuser(),
        "project_directory": str(ROOT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "tensorflow_cpu_devices": [
            str(device) for device in tf.config.list_physical_devices("CPU")
        ],
        "tensorflow_gpu_devices": [
            str(device) for device in tf.config.list_physical_devices("GPU")
        ],
        "free_disk_gib": round(disk.free / 1024**3, 3),
        "smoke_checks": smoke_check(tf) if args.smoke else "not requested",
    }
    output = ROOT / "reports/setup/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nSaved environment evidence to {output}")


if __name__ == "__main__":
    main()
