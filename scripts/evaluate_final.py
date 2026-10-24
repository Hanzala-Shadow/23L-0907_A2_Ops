"""Evaluate the two locked models once, then report and log saved predictions."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.train_cnn import check_space, configure_cpu, write_json

ROOT = Path(__file__).resolve().parents[1]


def figures(folder, labels, probabilities, threshold, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        probabilities >= threshold,
        labels=[0, 1],
        display_labels=["Real-labeled", "AI-generated"],
        ax=axis,
        colorbar=False,
    )
    axis.set_title(f"{name} test\nThreshold {threshold:.4f}")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(labels, probabilities, name="Test", ax=axis)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_title(name + " test ROC")
    fig.tight_layout()
    fig.savefig(folder / "roc.png", dpi=150)
    plt.close(fig)


def error_grid(output, rows, probabilities, threshold):
    import matplotlib.pyplot as plt

    from visionguard.preprocessing import preprocess_image

    errors = [
        (row, float(p))
        for row, p in zip(rows, probabilities)
        if int(p >= threshold) != int(row["label"])
    ]
    chosen = errors[:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    examples = []
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(chosen):
            continue
        row, probability = chosen[index]
        axis.imshow(preprocess_image(ROOT / "data/raw" / row["relative_path"]))
        predicted = "AI" if probability >= threshold else "Real"
        actual = "AI" if int(row["label"]) else "Real"
        axis.set_title(
            f"{index + 1}. True {actual}, predicted {predicted}\np(AI)={probability:.3f}",
            fontsize=10,
        )
        examples.append(
            {
                "example": index + 1,
                **row,
                "ai_probability": probability,
                "predicted_label": int(probability >= threshold),
            }
        )
    fig.suptitle("Champion: first ten errors in the fixed test manifest")
    fig.tight_layout()
    fig.savefig(output / "misclassified_examples.png", dpi=140)
    plt.close(fig)
    write_json(output / "misclassified_examples.json", examples)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_space(ROOT, margin_gib=0.25)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import mlflow
    import numpy as np
    import tensorflow as tf

    from scripts.prepare_splits import verify_source
    from visionguard.evaluation import atomic_json, binary_metrics, predict_once
    from visionguard.pipeline import make_dataset, read_split, verify_split_hash
    from visionguard.transfer_reporting import save_predictions

    plan_path = ROOT / "reports/selection/plan.json"
    encoded = plan_path.read_bytes()
    if (
        subprocess.check_output(
            ["git", "show", "HEAD:reports/selection/plan.json"], cwd=ROOT
        )
        != encoded
    ):
        raise RuntimeError("Commit the locked selection plan before test evaluation")
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    if any(
        path != "dvc.lock" and not path.startswith("reports/evaluation/")
        for path in changed
    ):
        raise RuntimeError("Commit source changes before evaluation")
    plan, plan_sha = json.loads(encoded), hashlib.sha256(encoded).hexdigest()
    summary = json.loads((ROOT / "data/splits/summary.json").read_text())
    if summary["raw_dataset_fingerprint_sha256"] != plan["data_fingerprint"]:
        raise ValueError("Selection and evaluation data differ")
    with (ROOT / "reports/data/image_manifest.csv").open(newline="") as source:
        if (
            verify_source(list(csv.DictReader(source)), ROOT / "data/raw")
            != plan["data_fingerprint"]
        ):
            raise ValueError("Raw image fingerprint changed")
    for name, digest in plan["split_manifest_sha256"].items():
        verify_split_hash(ROOT / f"data/splits/{name}.csv", digest)
    rows = read_split(ROOT / "data/splits/test.csv", ROOT / "data/raw")
    labels = np.asarray([int(row["label"]) for row in rows])
    dataset = make_dataset(rows, ROOT / "data/raw", plan["preprocessing"])
    tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(42)
    output = ROOT / "reports/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / "reports/training-runtime/final-test"
    ledger.mkdir(parents=True, exist_ok=True)
    completed = {}
    for role in plan["policy"]["test_models"]:
        model_record = plan["models"][role]
        checkpoint = ROOT / model_record["model_path"]
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != model_record["model_sha256"]
        ):
            raise ValueError("Locked model checkpoint changed")
        identity = {
            "role": role,
            "plan_sha256": plan_sha,
            "model_sha256": model_record["model_sha256"],
            "test_manifest_sha256": plan["split_manifest_sha256"]["test"],
            "images": len(rows),
            "threshold": model_record["threshold"],
        }

        def predict(checkpoint=checkpoint):
            check_space(ROOT)
            model = tf.keras.models.load_model(checkpoint, compile=False)
            result = model.predict(dataset, verbose=0).reshape(-1)
            check_space(ROOT)
            return result

        probabilities, fresh = predict_once(ledger / role, identity, predict)
        folder = output / role
        folder.mkdir(parents=True, exist_ok=True)
        metrics = binary_metrics(
            labels,
            probabilities,
            model_record["threshold"],
            plan["policy"]["false_positive_cost"],
            plan["policy"]["false_negative_cost"],
        )
        write_json(folder / "metrics.json", metrics)
        save_predictions(folder / "predictions.csv", rows, probabilities)
        figures(
            folder,
            labels,
            probabilities,
            model_record["threshold"],
            model_record["model"],
        )
        receipt = json.loads((ledger / role / "predictions.json").read_text())
        receipt.pop("probabilities")
        write_json(folder / "receipt.json", receipt)
        completed[role] = {
            "model": model_record["model"],
            "run_id": model_record["run_id"],
            **metrics,
        }
        if role == "champion":
            error_grid(output, rows, probabilities, model_record["threshold"])
        print(
            json.dumps({"role": role, "new_inference_pass": fresh, **metrics}),
            flush=True,
        )
    record = {
        "plan_sha256": plan_sha,
        "models": completed,
        "test_evaluated": True,
        "selection_used_test": False,
        "inference_passes_per_model": 1,
        "evaluation_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_fingerprint": plan["data_fingerprint"],
        "test_manifest_sha256": plan["split_manifest_sha256"]["test"],
    }
    write_json(output / "summary.json", record)
    with (output / "metrics.csv").open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["role", *completed["champion"].keys()],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {"role": role, **metrics} for role, metrics in completed.items()
        )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["PROJECT_NAME"])
    run_path = ledger / "mlflow_run.json"
    run_id = json.loads(run_path.read_text())["run_id"] if run_path.exists() else None
    with mlflow.start_run(
        run_id=run_id,
        run_name="final_test_evaluation",
        tags={
            "phase": "final_test",
            "git_commit": record["evaluation_git_commit"],
            "selection_plan_sha256": plan_sha,
            "data_fingerprint": plan["data_fingerprint"],
            "test_evaluated": "true",
            "selection_used_test": "false",
        },
    ) as run:
        atomic_json(run_path, {"run_id": run.info.run_id})
        client = mlflow.MlflowClient()
        for role, metrics in completed.items():
            numeric = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            mlflow.log_metrics(
                {f"{role}.test_{key}": value for key, value in numeric.items()}
            )
            client.log_batch(
                metrics["run_id"],
                metrics=[
                    mlflow.entities.Metric(
                        "test_" + key,
                        value,
                        int(datetime.now(UTC).timestamp() * 1000),
                        0,
                    )
                    for key, value in numeric.items()
                ],
            )
            client.set_tag(
                metrics["run_id"], "final_test_evaluation_run_id", run.info.run_id
            )
            client.set_tag(metrics["run_id"], "test_evaluated", "true")
            client.log_artifacts(
                metrics["run_id"], str(output / role), artifact_path="final_test"
            )
        mlflow.log_artifacts(str(output), artifact_path="evaluation")
        mlflow.log_artifacts(str(ROOT / "reports/selection"), artifact_path="selection")
        mlflow.log_artifacts(str(ROOT / "data/splits"), artifact_path="splits")
        mlflow.log_artifact(
            str(ROOT / "requirements/locked.txt"), artifact_path="source"
        )
    write_json(
        output / "mlflow.json",
        {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id},
    )
    check_space(ROOT)


if __name__ == "__main__":
    main()
