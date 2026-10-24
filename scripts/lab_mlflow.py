"""Compare two tiny local MNIST runs, then repeat one on DagsHub."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.train_cnn import configure_cpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-only", action="store_true")
    args = parser.parse_args()
    if shutil.disk_usage(ROOT).free < 2.25 * 1024**3:
        raise RuntimeError("Keep 2 GiB plus lab headroom")
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    # set_tracking_uri updates the process environment. Preserve the cloud
    # address before selecting a local backend.
    cloud_uri = os.environ["MLFLOW_TRACKING_URI"]
    assert cloud_uri.startswith("https://dagshub.com/")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    source = ROOT / "reports/training-runtime/keras/datasets/mnist.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
    )
    with np.load(source, allow_pickle=False) as archive:
        x, y = archive["x_train"][:1400], archive["y_train"][:1400]
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train = tf.data.Dataset.from_tensor_slices((x[:1200], y[:1200])).batch(64)
    validation = tf.data.Dataset.from_tensor_slices((x[1200:], y[1200:])).batch(64)
    train, validation = train.with_options(options), validation.with_options(options)
    runtime = ROOT / "reports/training-runtime/labs/mlflow"
    runtime.mkdir(parents=True, exist_ok=True)
    local_uri = "sqlite:///" + str(runtime / "tracking.db")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    records = []
    if args.remote_only:
        records = json.loads(
            (ROOT / "reports/labs/mlflow/verification.json").read_text()
        )["runs"][:2]
    mlflow.tensorflow.autolog(disable=True)
    choices = (
        [("dagshub", 0.001)]
        if args.remote_only
        else [("local", 0.001), ("local", 0.01), ("dagshub", 0.001)]
    )
    for location, rate in choices:
        if shutil.disk_usage(ROOT).free < 2.125 * 1024**3:
            raise RuntimeError("Disk reserve reached")
        uri = local_uri if location == "local" else cloud_uri
        mlflow.set_tracking_uri(uri)
        name = (
            "23L-0907_A2_Ops" if location == "dagshub" else "23L-0907_A2_Ops-mlflow-lab"
        )
        experiment = mlflow.set_experiment(name)
        if location == "local":
            mlflow.MlflowClient().set_experiment_tag(
                experiment.experiment_id, "lab", "MNIST"
            )
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential(
            [
                tf.keras.Input((28, 28)),
                tf.keras.layers.Rescaling(1 / 255),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        with mlflow.start_run(run_name=f"mlflow-lab-{location}-lr-{rate}") as run:
            mlflow.set_tags(
                {
                    "phase": "mlflow_toy",
                    "git_commit": source_commit,
                    "seed": "42",
                    "test_evaluated": "false",
                }
            )
            mlflow.log_params(
                {
                    "learning_rate": rate,
                    "units": 16,
                    "epochs": 2,
                    "train_images": 1200,
                    "validation_images": 200,
                    "dataset": "MNIST training partition",
                }
            )

            class LogEpoch(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    mlflow.log_metrics(
                        {k: float(v) for k, v in logs.items()}, step=epoch
                    )

            history = model.fit(
                train,
                validation_data=validation,
                epochs=2,
                callbacks=[LogEpoch()],
                verbose=0,
            ).history
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot([1, 2], history["loss"], label="train")
            ax.plot([1, 2], history["val_loss"], label="validation")
            ax.set(
                xlabel="Epoch", ylabel="Cross-entropy", title=f"MNIST lab: Adam {rate}"
            )
            ax.legend()
            fig.tight_layout()
            plot = runtime / f"{location}-{rate}.png"
            fig.savefig(plot, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(plot), "figures")
            mlflow.tensorflow.log_model(
                model,
                artifact_path="model",
                pip_requirements=["tensorflow-cpu==2.19.1", "keras==3.15.1"],
                input_example=x[:1],
                signature=mlflow.models.infer_signature(x[:1], model(x[:1]).numpy()),
            )
            records.append(
                {
                    "location": location,
                    "learning_rate": rate,
                    "experiment_id": experiment.experiment_id,
                    "run_id": run.info.run_id,
                    "validation_accuracy": history["val_accuracy"][-1],
                    "validation_loss": history["val_loss"][-1],
                    "epochs_logged": 2,
                    "model_and_plot_logged": True,
                }
            )
    output = ROOT / "reports/labs/mlflow"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "runs": records,
        "local_tracking_uri": "sqlite database in ignored lab directory",
        "test_set_used": False,
        "free_disk_gib": shutil.disk_usage(ROOT).free / 1024**3,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
