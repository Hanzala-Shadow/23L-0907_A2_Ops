"""Train the first CPU CNN using train/validation data and manual MLflow logging."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def configure_cpu():
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "MPLCONFIGDIR": str(ROOT / "reports/training-runtime/matplotlib"),
        "KERAS_HOME": str(ROOT / "reports/training-runtime/keras"),
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "60",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    }.items():
        os.environ[key] = value


def check_space(path, reserve_gib=2.0, margin_gib=0.125):
    free = shutil.disk_usage(path).free
    if free < (reserve_gib + margin_gib) * GIB:
        raise RuntimeError("Stopping to preserve the 2 GiB disk reserve")
    return free


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_figures(
    output, history, best_epoch, labels, probabilities, batch, augmentation
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, title in zip(
        axes, ("loss", "accuracy"), ("Binary cross-entropy", "Accuracy")
    ):
        axis.plot(epochs, history[key], label="Training")
        axis.plot(epochs, history[f"val_{key}"], label="Validation")
        axis.axvline(
            best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}"
        )
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        probabilities >= 0.5,
        display_labels=["Real-labeled", "AI-generated"],
        ax=axis,
        colorbar=False,
    )
    axis.set_title("Validation confusion matrix (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(output / "validation_confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(
        labels, probabilities, name="CNN validation", ax=axis
    )
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_title("Validation ROC curve")
    fig.tight_layout()
    fig.savefig(output / "validation_roc.png", dpi=150)
    plt.close(fig)
    images, targets = batch
    augmented = augmentation(images / 255.0, training=True).numpy()
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for index in range(8):
        axes[0, index].imshow(images[index].astype(np.uint8))
        axes[0, index].set_title(
            "AI-generated" if targets[index] else "Real-labeled", fontsize=8
        )
        axes[1, index].imshow(np.clip(augmented[index], 0, 1))
        for row in range(2):
            axes[row, index].axis("off")
    fig.suptitle(
        "Training images: deterministic preprocessing (top), random augmentation (bottom)"
    )
    fig.tight_layout()
    fig.savefig(output / "augmented_batch.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.reserve_gib < 2:
        parser.error("The minimum disk reserve is 2 GiB")
    check_space(ROOT, args.reserve_gib, 0.25)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from scripts.prepare_splits import verify_source
    from visionguard.cnn import build_augmentation, build_cnn
    from visionguard.pipeline import make_dataset, read_split, verify_split_hash
    from visionguard.preprocessing import POLICY_VERSION

    params = json.loads((ROOT / "params.json").read_text())
    preprocessing, config = params["preprocessing"], params["cnn_baseline"]
    if preprocessing["policy"] != POLICY_VERSION:
        raise ValueError("Unknown preprocessing policy")
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com/"):
        raise ValueError("Set the DagsHub MLflow address in the local environment")
    if not all(
        os.environ.get(key)
        for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
    ):
        raise ValueError("DagsHub MLflow credentials are missing")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT
    ).strip()
    if dirty:
        raise RuntimeError(
            "Commit tracked training code and parameters before training"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    summary = json.loads((ROOT / "data/splits/summary.json").read_text())
    raw = ROOT / "data/raw"
    with (ROOT / "reports/data/image_manifest.csv").open(newline="") as source:
        manifest = list(csv.DictReader(source))
    if verify_source(manifest, raw) != summary["raw_dataset_fingerprint_sha256"]:
        raise ValueError("Raw dataset differs from the recorded data version")
    splits = {}
    for name in ("train", "validation"):
        path = ROOT / f"data/splits/{name}.csv"
        verify_split_hash(path, summary["splits"][name]["manifest_sha256"])
        splits[name] = read_split(path, raw)
    if {row["sha256"] for row in splits["train"]} & {
        row["sha256"] for row in splits["validation"]
    }:
        raise ValueError("Training and validation hashes overlap")
    train = make_dataset(
        splits["train"], raw, preprocessing, training=True, seed=config["seed"]
    )
    validation = make_dataset(splits["validation"], raw, preprocessing)
    output = ROOT / "reports/training/cnn_baseline"
    model_dir = ROOT / "models/cnn_baseline"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "best.keras").exists():
        raise RuntimeError(
            "Existing baseline model would be overwritten; version it first"
        )
    model = build_cnn(preprocessing, config)
    lines = []
    model.summary(print_fn=lambda value: lines.append(value))
    (output / "model_summary.txt").write_text("\n".join(lines).rstrip() + "\n")
    with (output / "layers.csv").open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["layer", "type", "output_shape", "parameters"])
        for layer in model.layers:
            writer.writerow(
                [
                    layer.name,
                    type(layer).__name__,
                    str(layer.output.shape),
                    layer.count_params(),
                ]
            )
    raw_version = next(
        row.split(":", 1)[1].strip()
        for row in (ROOT / "data/raw.dvc").read_text().splitlines()
        if "md5:" in row
    )
    write_json(
        output / "configuration.json",
        {
            "preprocessing": preprocessing,
            "cnn_baseline": config,
            "label_mapping": summary["labels"],
            "input_dtype": "uint8",
            "normalization": "1/255 inside model",
            "test_evaluated": False,
        },
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("PROJECT_NAME", "23L-0907_A2_Ops"))
    minimum_free = [check_space(ROOT, args.reserve_gib)]
    started = time.monotonic()

    class TrainingLog(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            minimum_free[0] = min(minimum_free[0], check_space(ROOT, args.reserve_gib))

        def on_epoch_end(self, epoch, logs=None):
            minimum_free[0] = min(minimum_free[0], check_space(ROOT, args.reserve_gib))
            values = {key: float(value) for key, value in logs.items()}
            values["learning_rate"] = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
            mlflow.log_metrics(values, step=epoch + 1)
            print(json.dumps({"epoch": epoch + 1, **values}), flush=True)

    with mlflow.start_run(
        run_name="cnn_baseline_cpu_v1",
        tags={
            "model_family": "custom_cnn",
            "logging_method": "manual_callback",
            "git_commit": commit,
            "mlflow.source.git.commit": commit,
            "dvc_raw_md5": raw_version,
            "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
            "seed": str(config["seed"]),
            "device": "cpu",
            "preprocessing_policy": POLICY_VERSION,
            "test_evaluated": "false",
        },
    ) as run:
        mlflow.log_params(
            {
                **{
                    f"preprocessing.{key}": value
                    for key, value in preprocessing.items()
                },
                **{f"cnn.{key}": value for key, value in config.items()},
                "optimizer": "Adam",
                "loss_function": "binary_crossentropy",
                "selection_metric": "val_loss",
                "threshold": 0.5,
                "parameters": model.count_params(),
                "train_images": len(splits["train"]),
                "validation_images": len(splits["validation"]),
                "intraop_threads": 2,
                "interop_threads": 1,
                "op_determinism": True,
                "oneDNN": False,
            }
        )
        callbacks = [
            TrainingLog(),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                model_dir / "best.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=config["lr_patience"],
                factor=0.5,
                min_lr=1e-6,
            ),
            tf.keras.callbacks.CSVLogger(output / "history.csv"),
        ]
        history = model.fit(
            train,
            validation_data=validation,
            epochs=config["epochs"],
            callbacks=callbacks,
            verbose=2,
        ).history
        history_path = output / "history.csv"
        history_path.write_bytes(history_path.read_bytes().replace(b"\r\n", b"\n"))
        check_space(ROOT, args.reserve_gib)
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best = tf.keras.models.load_model(model_dir / "best.keras")
        probabilities = best.predict(validation, verbose=0).reshape(-1)
        labels = np.asarray([int(row["label"]) for row in splits["validation"]])
        predicted = probabilities >= 0.5
        metrics = {
            "validation_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "validation_accuracy": float(accuracy_score(labels, predicted)),
            "validation_precision": float(
                precision_score(labels, predicted, zero_division=0)
            ),
            "validation_recall": float(
                recall_score(labels, predicted, zero_division=0)
            ),
            "validation_f1": float(f1_score(labels, predicted, zero_division=0)),
            "validation_roc_auc": float(roc_auc_score(labels, probabilities)),
            "best_epoch": best_epoch,
            "epochs_completed": len(history["loss"]),
            "training_seconds": time.monotonic() - started,
            "parameters": model.count_params(),
            "threshold": 0.5,
        }
        mlflow.log_metrics(metrics)
        write_json(output / "metrics.json", metrics)
        with (output / "validation_predictions.csv").open("w", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(
                ["relative_path", "label", "ai_probability", "predicted_label"]
            )
            for row, probability in zip(splits["validation"], probabilities):
                writer.writerow(
                    [
                        row["relative_path"],
                        row["label"],
                        float(probability),
                        int(probability >= 0.5),
                    ]
                )
        preview_rows = []
        for label in (0, 1):
            preview_rows.extend(
                [row for row in splits["train"] if int(row["label"]) == label][:4]
            )
        preview = make_dataset(preview_rows, raw, {**preprocessing, "batch_size": 8})
        pixels, targets = next(iter(preview))
        make_figures(
            output,
            history,
            best_epoch,
            labels,
            probabilities,
            (pixels.numpy(), targets.numpy().reshape(-1)),
            build_augmentation(config["augmentation"], config["seed"]),
        )
        sample = pixels[:2].numpy()
        expected = best(sample, training=False).numpy()
        mlflow.tensorflow.log_model(
            best,
            artifact_path="model",
            input_example=sample,
            pip_requirements=[f"tensorflow-cpu=={metadata.version('tensorflow-cpu')}"],
        )
        with tempfile.TemporaryDirectory(
            prefix="visionguard-model-check-"
        ) as temporary:
            restored = mlflow.tensorflow.load_model(
                f"runs:/{run.info.run_id}/model", dst_path=temporary
            )
            np.testing.assert_allclose(
                expected, restored(sample, training=False).numpy(), rtol=1e-5, atol=1e-6
            )
        check_space(ROOT, args.reserve_gib)
        model_sha = hashlib.sha256((model_dir / "best.keras").read_bytes()).hexdigest()
        record = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_url": f"{uri.removesuffix('.mlflow')}/experiments/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
            "git_commit": commit,
            "dvc_raw_md5": raw_version,
            "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
            "model_sha256": model_sha,
            "label_mapping": summary["labels"],
            "test_evaluated": False,
            "model_roundtrip": "passed",
            "manual_logging": True,
            "minimum_free_gib_observed": minimum_free[0] / GIB,
            "free_after_gib": shutil.disk_usage(ROOT).free / GIB,
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "train_class_counts": dict(
                Counter(row["class"] for row in splits["train"])
            ),
        }
        write_json(output / "run.json", record)
        write_json(
            model_dir / "preprocessing.json",
            json.loads((output / "configuration.json").read_text()),
        )
        (output / "packages.txt").write_text(
            subprocess.check_output(
                [
                    os.sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "freeze",
                ],
                text=True,
            )
        )
        mlflow.log_artifacts(str(output), artifact_path="training")
        mlflow.log_artifacts(str(ROOT / "data/splits"), artifact_path="splits")
        mlflow.log_artifact(str(model_dir / "best.keras"), artifact_path="keras")
        mlflow.log_artifact(str(ROOT / "params.json"), artifact_path="source")
        mlflow.log_artifact(str(ROOT / "dvc.lock"), artifact_path="source")
        print(json.dumps({"metrics": metrics, "run": record}, indent=2), flush=True)


if __name__ == "__main__":
    main()
