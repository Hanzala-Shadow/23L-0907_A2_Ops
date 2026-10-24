"""Lock the model and decision thresholds using saved validation predictions."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.train_cnn import check_space, write_json
from visionguard.evaluation import binary_metrics, choose_threshold

ROOT = Path(__file__).resolve().parents[1]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_space(ROOT)
    output = ROOT / "reports/selection"
    if (output / "plan.json").exists():
        raise RuntimeError(
            "A selection plan is already locked; refusing to overwrite it"
        )
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT):
        raise RuntimeError("Commit source and policy before model selection")
    policy = json.loads((ROOT / "config/selection.json").read_text())
    summary = json.loads((ROOT / "data/splits/summary.json").read_text())
    validation_path = ROOT / "data/splits/validation.csv"
    digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    if digest != summary["splits"]["validation"]["manifest_sha256"]:
        raise ValueError("Validation manifest changed")
    with validation_path.open(newline="") as source:
        expected = [(r["relative_path"], r["label"]) for r in csv.DictReader(source)]
    candidates = []
    for name in ("cnn_baseline", "mobilenetv2_frozen", "mobilenetv2_finetuned"):
        folder = ROOT / "reports/training" / name
        candidates.append(
            {
                "model": name,
                **json.loads((folder / "run.json").read_text()),
                **json.loads((folder / "metrics.json").read_text()),
                "predictions": str(
                    folder.relative_to(ROOT) / "validation_predictions.csv"
                ),
                "model_path": f"models/{name}/best.keras",
            }
        )
    search = json.loads(
        (ROOT / "reports/training/keras_tuner/summary.json").read_text()
    )
    for trial in search["trials"]:
        candidates.append(
            {
                "model": "mobilenetv2_tuned",
                **trial,
                "predictions": f"reports/training/keras_tuner/trial_{trial['trial_id']}/validation_predictions.csv",
                "model_path": "models/mobilenetv2_tuned/best.keras",
            }
        )
    for candidate in candidates:
        if (
            candidate["test_evaluated"]
            or candidate["data_fingerprint"]
            != summary["raw_dataset_fingerprint_sha256"]
        ):
            raise ValueError(
                "Model comparison includes incompatible data or test results"
            )
        with (ROOT / candidate["predictions"]).open(newline="") as source:
            rows = list(csv.DictReader(source))
        if [(r["relative_path"], r["label"]) for r in rows] != expected:
            raise ValueError("Candidate validation membership differs")
        labels = [int(r["label"]) for r in rows]
        probabilities = [float(r["ai_probability"]) for r in rows]
        metrics = binary_metrics(labels, probabilities, 0.5)
        if abs(metrics["loss"] - candidate["validation_loss"]) > 1e-6:
            raise ValueError("Validation loss does not match saved predictions")
    selected = min(candidates, key=lambda row: (row["validation_loss"], row["run_id"]))
    if (
        selected["model"] == "mobilenetv2_tuned"
        and selected["run_id"] != search["best_run_id"]
    ):
        raise ValueError("Tuning summary and selected checkpoint disagree")
    output.mkdir(parents=True, exist_ok=True)
    locked = {}
    for role, candidate in (
        ("champion", selected),
        ("cnn_baseline", next(r for r in candidates if r["model"] == "cnn_baseline")),
        (
            "registry_reference",
            next(r for r in candidates if r["model"] == "mobilenetv2_frozen"),
        ),
    ):
        path = ROOT / candidate["model_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["model_sha256"]:
            raise ValueError("Checkpoint differs from its training record")
        prediction_path = ROOT / candidate["predictions"]
        with prediction_path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        best, curve = choose_threshold(
            [int(r["label"]) for r in rows],
            [float(r["ai_probability"]) for r in rows],
            policy["false_positive_cost"],
            policy["false_negative_cost"],
        )
        with (output / f"{role}_thresholds.csv").open("w", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=list(curve[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(curve)
        locked[role] = {
            "model": candidate["model"],
            "model_path": candidate["model_path"],
            "model_sha256": candidate["model_sha256"],
            "run_id": candidate["run_id"],
            "training_git_commit": candidate["git_commit"],
            "validation_loss": candidate["validation_loss"],
            "validation_predictions_sha256": hashlib.sha256(
                prediction_path.read_bytes()
            ).hexdigest(),
            "threshold": best["threshold"],
            "validation_metrics": best,
        }
    plan = {
        "policy": policy,
        "models": locked,
        "selection_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
        "split_manifest_sha256": {
            name: value["manifest_sha256"] for name, value in summary["splits"].items()
        },
        "preprocessing": json.loads((ROOT / "params.json").read_text())[
            "preprocessing"
        ],
        "labels": summary["labels"],
        "candidate_count": len(candidates),
        "test_evaluated": False,
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "candidates.json",
        [
            {
                key: r[key]
                for key in ("model", "run_id", "validation_loss", "validation_accuracy")
            }
            for r in candidates
        ],
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
