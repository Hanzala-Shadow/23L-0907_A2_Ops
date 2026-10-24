"""Validation reports for the two MobileNetV2 training phases."""

import csv
import hashlib
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from visionguard.transfer import BACKBONE_NAME


def save_structure(model, output):
    lines = []
    model.summary(print_fn=lines.append, expand_nested=True, show_trainable=True)
    (output / "model_summary.txt").write_text("\n".join(lines).rstrip() + "\n")
    backbone = model.get_layer(BACKBONE_NAME)
    with (output / "layers.csv").open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["scope", "layer", "type", "parameters", "trainable"])
        for layer in backbone.layers:
            writer.writerow(
                [
                    "backbone",
                    layer.name,
                    type(layer).__name__,
                    layer.count_params(),
                    bool(layer.trainable_weights),
                ]
            )
        for layer in model.layers:
            if layer.name != BACKBONE_NAME:
                writer.writerow(
                    [
                        "head_or_preprocessing",
                        layer.name,
                        type(layer).__name__,
                        layer.count_params(),
                        bool(layer.trainable_weights),
                    ]
                )


def validation_metrics(labels, probabilities):
    predicted = probabilities >= 0.5
    return {
        "validation_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "validation_accuracy": float(accuracy_score(labels, predicted)),
        "validation_precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "validation_recall": float(recall_score(labels, predicted, zero_division=0)),
        "validation_f1": float(f1_score(labels, predicted, zero_division=0)),
        "validation_roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def save_predictions(path, rows, probabilities):
    with path.open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["relative_path", "label", "ai_probability", "predicted_label"])
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                [
                    row["relative_path"],
                    row["label"],
                    float(probability),
                    int(probability >= 0.5),
                ]
            )


def make_phase_figures(output, history, best_epoch, labels, probabilities, phase):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    title = "MobileNetV2 " + phase
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, label in zip(
        axes, ("loss", "accuracy"), ("Binary cross-entropy", "Accuracy")
    ):
        axis.plot(epochs, history[metric], label="Training")
        axis.plot(epochs, history["val_" + metric], label="Validation")
        axis.axvline(
            best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}"
        )
        axis.set(xlabel="Epoch", ylabel=label, title=label)
        axis.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        probabilities >= 0.5,
        display_labels=["Real-labeled", "AI-generated"],
        colorbar=False,
        ax=axis,
    )
    axis.set_title(title + "\nValidation, threshold 0.5")
    fig.tight_layout()
    fig.savefig(output / "validation_confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(labels, probabilities, name="Validation", ax=axis)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_title(title + " validation ROC")
    fig.tight_layout()
    fig.savefig(output / "validation_roc.png", dpi=150)
    plt.close(fig)


def pretrained_source(root, image_size, alpha):
    filename = f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{float(alpha)}_{image_size}_no_top.h5"
    path = root / "reports/training-runtime/keras/models" / filename
    return {
        "source_url": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
        + path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def comparison_records(root):
    rows = []
    fingerprints = set()
    split_counts = []
    validation_cases = []
    for name in ("cnn_baseline", "mobilenetv2_frozen", "mobilenetv2_finetuned"):
        folder = root / "reports/training" / name
        metrics = json.loads((folder / "metrics.json").read_text())
        run = json.loads((folder / "run.json").read_text())
        if run["test_evaluated"]:
            raise ValueError("This comparison is restricted to validation results")
        fingerprints.add(run["data_fingerprint"])
        split_counts.append(run["split_counts"])
        with (folder / "validation_predictions.csv").open(newline="") as source:
            validation_cases.append(
                [(row["relative_path"], row["label"]) for row in csv.DictReader(source)]
            )
        rows.append(
            {
                "model": name,
                **metrics,
                "run_id": run["run_id"],
                "run_url": run["run_url"],
                "model_path": f"models/{name}/best.keras",
            }
        )
    if len(fingerprints) != 1 or any(
        count != split_counts[0] for count in split_counts
    ):
        raise ValueError("Model runs used different datasets or split sizes")
    if any(cases != validation_cases[0] for cases in validation_cases):
        raise ValueError("Model runs used different validation images or labels")
    return rows


def write_comparison(root):
    rows = comparison_records(root)
    selected = min(rows, key=lambda row: row["validation_loss"])
    output = root / "reports/training"
    columns = [
        "model",
        "validation_loss",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
        "best_epoch",
        "epochs_completed",
        "run_id",
        "model_path",
    ]
    with (output / "model_comparison.csv").open("w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "selection_metric": "validation_loss",
        "threshold": 0.5,
        "provisional_selected_model": selected["model"],
        "selected_model_path": selected["model_path"],
        "test_evaluated": False,
        "models": rows,
    }
    (output / "model_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
