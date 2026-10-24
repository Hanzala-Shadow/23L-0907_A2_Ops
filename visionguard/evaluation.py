"""Validation threshold selection and durable, single-pass test predictions."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def checked_arrays(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty matching vectors")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both binary classes are required")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return labels.astype(int), probabilities


def binary_metrics(
    labels, probabilities, threshold, false_positive_cost=2, false_negative_cost=1
):
    labels, probabilities = checked_arrays(labels, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    predicted = probabilities >= threshold
    tn, fp, fn, tp = (
        int(x) for x in confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    )
    return {
        "loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "weighted_errors": false_positive_cost * fp + false_negative_cost * fn,
        "threshold": float(threshold),
        "images": len(labels),
    }


def choose_threshold(
    labels, probabilities, false_positive_cost=2, false_negative_cost=1
):
    labels, probabilities = checked_arrays(labels, probabilities)
    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("Error costs must be positive")
    unique = np.unique(probabilities)
    thresholds = np.unique(np.r_[0.0, 0.5, 1.0, (unique[:-1] + unique[1:]) / 2])
    curve = [
        binary_metrics(
            labels, probabilities, t, false_positive_cost, false_negative_cost
        )
        for t in thresholds
    ]
    selected = min(
        curve,
        key=lambda row: (
            row["weighted_errors"],
            row["fp"],
            abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )
    return selected, curve


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def predict_once(directory, identity, predict):
    """Reuse completed predictions; never repeat an interrupted/uncertain pass."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = directory / "started.json"
    complete = directory / "predictions.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["identity"] != identity:
            raise RuntimeError(
                "Final test identity changed; refusing another evaluation"
            )
        encoded = json.dumps(saved["probabilities"], allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != saved["probabilities_sha256"]:
            raise RuntimeError("Saved test predictions failed their integrity check")
        return np.asarray(saved["probabilities"], dtype=float), False
    try:
        with started.open("x") as target:
            json.dump(identity, target, sort_keys=True)
            target.flush()
            os.fsync(target.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            "A test pass started without durable results; refusing to repeat it"
        ) from error
    probabilities = np.asarray(predict(), dtype=float).reshape(-1)
    if len(probabilities) != identity["images"] or not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError("Invalid final predictions; test pass will not be repeated")
    values = probabilities.tolist()
    atomic_json(
        complete,
        {
            "identity": identity,
            "probabilities": values,
            "probabilities_sha256": hashlib.sha256(
                json.dumps(values, allow_nan=False).encode()
            ).hexdigest(),
            "inference_passes": 1,
        },
    )
    return probabilities, True
