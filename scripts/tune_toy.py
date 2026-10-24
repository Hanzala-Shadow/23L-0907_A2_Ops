"""Demonstrate Keras Tuner on a small subset of MNIST's training partition."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.train_cnn import check_space, configure_cpu, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_space(ROOT, margin_gib=0.25)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import keras_tuner as kt
    import mlflow
    import numpy as np
    import tensorflow as tf

    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "scripts/tune_toy.py"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    if set(changed) - {"dvc.lock"}:
        raise RuntimeError("Commit code before running the toy example")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    path = tf.keras.utils.get_file(
        fname="mnist.npz",
        origin=DATA_URL,
        file_hash=DATA_SHA256,
        cache_dir=str(ROOT / "reports/training-runtime/keras"),
    )
    check_space(ROOT)
    # Read only training entries from the built-in dataset archive.
    with np.load(path, allow_pickle=False) as archive:
        images, labels = archive["x_train"], archive["y_train"]
    indices = np.random.default_rng(42).permutation(len(images))[:7000]
    train_ids, validation_ids = indices[:6000], indices[6000:]
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    validation = (
        tf.data.Dataset.from_tensor_slices(
            (images[validation_ids], labels[validation_ids])
        )
        .batch(128)
        .with_options(options)
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["PROJECT_NAME"])
    output = ROOT / "reports/training/toy_tuner"
    output.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "reports/training-runtime/toy_tuner" / commit
    config = {
        "dataset": "MNIST training partition",
        "source_url": DATA_URL,
        "source_sha256": DATA_SHA256,
        "seed": 42,
        "train_images": 6000,
        "validation_images": 1000,
        "selected_indices_sha256": hashlib.sha256(
            indices.astype("<i8").tobytes()
        ).hexdigest(),
        "learning_rate": [0.001, 0.01],
        "units": [32, 64],
        "max_trials": 4,
        "epochs": 3,
        "batch_size": 128,
        "test_evaluated": False,
    }
    write_json(output / "configuration.json", config)
    hp = kt.HyperParameters()
    hp.Choice("learning_rate", config["learning_rate"])
    hp.Choice("units", config["units"])

    class ToySearch(kt.RandomSearch):
        def load_model(self, trial):
            return tf.keras.models.load_model(
                Path(self.get_trial_dir(trial.trial_id)) / "best.keras"
            )

        def run_trial(self, trial, **kwargs):
            check_space(ROOT)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)
            values = trial.hyperparameters.values
            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(28, 28), dtype="uint8"),
                    tf.keras.layers.Rescaling(1 / 255),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(values["units"], activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(values["learning_rate"]),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            train = (
                tf.data.Dataset.from_tensor_slices(
                    (images[train_ids], labels[train_ids])
                )
                .shuffle(6000, seed=42)
                .batch(128)
                .with_options(options)
            )
            folder = output / f"trial_{trial.trial_id}"
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(self.get_trial_dir(trial.trial_id)) / "best.keras"

            class EpochLog(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, logs=None):
                    check_space(ROOT)

                def on_epoch_end(self, epoch, logs=None):
                    mlflow.log_metrics(
                        {key: float(value) for key, value in logs.items()},
                        step=epoch + 1,
                    )

            with mlflow.start_run(
                run_name=f"mnist_tuner_{trial.trial_id}",
                tags={
                    "phase": "keras_tuner_toy",
                    "dataset": "mnist_training_partition",
                    "git_commit": commit,
                    "mlflow.source.git.commit": commit,
                    "test_evaluated": "false",
                    "device": "cpu",
                    "trial_id": trial.trial_id,
                    "data_fingerprint": DATA_SHA256,
                },
            ) as run:
                mlflow.log_params(
                    {
                        **values,
                        "seed": 42,
                        "epochs": 3,
                        "batch_size": 128,
                        "train_images": 6000,
                        "validation_images": 1000,
                    }
                )
                history = model.fit(
                    train,
                    validation_data=validation,
                    epochs=3,
                    verbose=0,
                    shuffle=False,
                    callbacks=[
                        EpochLog(),
                        tf.keras.callbacks.ModelCheckpoint(
                            checkpoint, monitor="val_loss", save_best_only=True
                        ),
                        tf.keras.callbacks.CSVLogger(folder / "history.csv"),
                    ],
                ).history
                log_path = folder / "history.csv"
                log_path.write_bytes(log_path.read_bytes().replace(b"\r\n", b"\n"))
                best_epoch = int(np.argmin(history["val_loss"])) + 1
                record = {
                    "trial_id": trial.trial_id,
                    **values,
                    "validation_loss": float(history["val_loss"][best_epoch - 1]),
                    "validation_accuracy": float(
                        history["val_accuracy"][best_epoch - 1]
                    ),
                    "best_epoch": best_epoch,
                    "epochs_completed": 3,
                    "run_id": run.info.run_id,
                    "git_commit": commit,
                    "test_evaluated": False,
                }
                write_json(folder / "run.json", record)
                mlflow.log_metrics(
                    {
                        key: record[key]
                        for key in (
                            "validation_loss",
                            "validation_accuracy",
                            "best_epoch",
                        )
                    }
                )
                mlflow.log_artifacts(str(folder), artifact_path="training")
                mlflow.log_artifact(
                    str(output / "configuration.json"), artifact_path="source"
                )
                mlflow.log_artifact(str(checkpoint), artifact_path="keras")
                print(
                    json.dumps({"toy_trial_complete": trial.trial_id, **record}),
                    flush=True,
                )
            return {"val_loss": record["validation_loss"]}

    tuner = ToySearch(
        objective="val_loss",
        max_trials=4,
        seed=42,
        hyperparameters=hp,
        allow_new_entries=False,
        directory=str(runtime),
        project_name="search",
        overwrite=True,
    )
    tuner.search(verbose=0)
    if len(tuner.oracle.trials) != 4 or any(
        trial.status != "COMPLETED" for trial in tuner.oracle.trials.values()
    ):
        raise RuntimeError("The four toy trials did not complete")
    trials = [
        json.loads((output / f"trial_{trial_id}/run.json").read_text())
        for trial_id in sorted(tuner.oracle.trials)
    ]
    best_trial = tuner.oracle.get_best_trials(1)[0]
    selected = next(
        trial for trial in trials if trial["trial_id"] == best_trial.trial_id
    )
    model_dir = ROOT / "models/toy_mnist"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(tuner.get_trial_dir(best_trial.trial_id)) / "best.keras",
        model_dir / "best.keras",
    )
    write_json(
        output / "summary.json",
        {
            "best": selected,
            "trials": trials,
            "completed_trials": 4,
            "test_evaluated": False,
        },
    )
    with (output / "trials.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(trials[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(trials)
    check_space(ROOT)
    print(json.dumps({"toy_best": selected}), flush=True)


if __name__ == "__main__":
    main()
