"""Compare saved validation results without evaluating the test split."""

from pathlib import Path

from scripts.train_cnn import configure_cpu


def main():
    configure_cpu()
    from visionguard.transfer_reporting import write_comparison

    root = Path(__file__).resolve().parents[1]
    result = write_comparison(root)
    lines = [
        "# Validation model comparison",
        "",
        "All three checkpoints use the same train/validation split and threshold 0.5.",
        "The test split has not been evaluated. Selection is provisional and uses validation loss.",
        "",
        "| Model | Loss | Accuracy | Precision | Recall | F1 | ROC-AUC | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["models"]:
        lines.append(
            f"| [{row['model']}]({row['run_url']}) | {row['validation_loss']:.4f} | {row['validation_accuracy']:.2%} | {row['validation_precision']:.4f} | {row['validation_recall']:.4f} | {row['validation_f1']:.4f} | {row['validation_roc_auc']:.4f} | {row['best_epoch']} |"
        )
    lines.extend(
        [
            "",
            f"Lowest validation loss: **{result['provisional_selected_model']}**.",
            "",
            "This is an initial comparison. Hyperparameter trials, threshold selection and final test evaluation remain outstanding. No model has been promoted to a serving champion.",
            "",
            "The CNN used manual MLflow logging. MobileNetV2 used TensorFlow autologging for epoch metrics and explicit logging for project metadata, exact checkpoint metrics, plots and selected model artifacts.",
            "",
        ]
    )
    (root / "reports/training/model_comparison.md").write_text("\n".join(lines))
    print("Provisional validation selection:", result["provisional_selected_model"])


if __name__ == "__main__":
    main()
