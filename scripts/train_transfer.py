"""Train the frozen or last-block fine-tuned MobileNetV2 phase on CPU."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from importlib import metadata
from pathlib import Path

from scripts.train_cnn import check_space, configure_cpu, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("frozen", "finetuned"), required=True)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.reserve_gib < 2:
        parser.error("The minimum reserve is 2 GiB")
    check_space(ROOT, args.reserve_gib, 0.25)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    import tensorflow as tf

    from scripts.prepare_splits import verify_source
    from visionguard.pipeline import make_dataset, read_split, verify_split_hash
    from visionguard.preprocessing import POLICY_VERSION
    from visionguard.transfer import (
        BACKBONE_NAME,
        backbone_fingerprints,
        build_transfer_model,
        compile_transfer_model,
        unfreeze_last_block,
        verify_backbone_changes,
    )
    from visionguard.transfer_reporting import (
        make_phase_figures,
        pretrained_source,
        save_predictions,
        save_structure,
        validation_metrics,
    )

    params = json.loads((ROOT / "params.json").read_text())
    preprocessing, config = params["preprocessing"], params["mobilenetv2"]
    phase_config = config[args.phase]
    if preprocessing["policy"] != POLICY_VERSION:
        raise ValueError("Unknown preprocessing policy")
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    if set(changed) - {"dvc.lock"}:
        raise RuntimeError("Commit tracked code and parameters before training")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com/") or not all(
        os.environ.get(key)
        for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
    ):
        raise ValueError("Configure DagsHub MLflow in the ignored local environment")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    summary = json.loads((ROOT / "data/splits/summary.json").read_text())
    with (ROOT / "reports/data/image_manifest.csv").open(newline="") as source:
        manifest = list(csv.DictReader(source))
    raw = ROOT / "data/raw"
    if verify_source(manifest, raw) != summary["raw_dataset_fingerprint_sha256"]:
        raise ValueError("Dataset fingerprint differs from the recorded version")
    splits = {}
    for name in ("train", "validation"):
        path = ROOT / f"data/splits/{name}.csv"
        verify_split_hash(path, summary["splits"][name]["manifest_sha256"])
        splits[name] = read_split(path, raw)
    if {row["sha256"] for row in splits["train"]} & {
        row["sha256"] for row in splits["validation"]
    }:
        raise ValueError("Training and validation overlap")
    train = make_dataset(
        splits["train"], raw, preprocessing, training=True, seed=config["seed"]
    )
    validation = make_dataset(splits["validation"], raw, preprocessing)
    name = f"mobilenetv2_{args.phase}"
    output = ROOT / "reports/training" / name
    model_dir = ROOT / "models" / name
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "best.keras").exists():
        raise RuntimeError("Use DVC versioning before replacing an existing checkpoint")
    initialization = {}
    if args.phase == "frozen":
        model = build_transfer_model(preprocessing, config)
    else:
        initial_path = ROOT / "models/mobilenetv2_frozen/best.keras"
        initial_run = json.loads(
            (ROOT / "reports/training/mobilenetv2_frozen/run.json").read_text()
        )
        initialization = {
            "initialized_from_run_id": initial_run["run_id"],
            "initialized_from_model_sha256": hashlib.sha256(
                initial_path.read_bytes()
            ).hexdigest(),
        }
        model = tf.keras.models.load_model(initial_path)
        unfreeze_last_block(model, phase_config["unfreeze_prefix"])
    compile_transfer_model(model, phase_config["learning_rate"])
    before = backbone_fingerprints(model)
    save_structure(model, output)
    trainable_backbone = [
        layer.name
        for layer in model.get_layer(BACKBONE_NAME).layers
        if layer.trainable_weights
    ]
    structure = {
        "trainable_backbone_layers": trainable_backbone,
        "trainable_parameters": sum(
            int(tf.size(weight)) for weight in model.trainable_weights
        ),
        "total_parameters": model.count_params(),
        "batchnorm_training_mode": False,
        "final_projection_trainable": False,
    }
    write_json(output / "trainable_layers.json", structure)
    configuration = {
        "phase": args.phase,
        "preprocessing": preprocessing,
        "mobilenetv2": config,
        "label_mapping": summary["labels"],
        "input_dtype": "uint8",
        "normalization": "pixel / 127.5 - 1 inside model",
        "test_evaluated": False,
        "pretrained_weights": pretrained_source(
            ROOT, preprocessing["image_size"], config["alpha"]
        ),
        **initialization,
    }
    write_json(output / "configuration.json", configuration)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ["PROJECT_NAME"])
    minimum_free = [check_space(ROOT, args.reserve_gib)]
    started = time.monotonic()

    class ResourceLog(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            minimum_free[0] = min(minimum_free[0], check_space(ROOT, args.reserve_gib))

        def on_epoch_end(self, epoch, logs=None):
            minimum_free[0] = min(minimum_free[0], check_space(ROOT, args.reserve_gib))
            print(
                json.dumps(
                    {
                        "phase": args.phase,
                        "epoch": epoch + 1,
                        **{key: float(value) for key, value in logs.items()},
                    }
                ),
                flush=True,
            )

    raw_md5 = next(
        line.split(":", 1)[1].strip()
        for line in (ROOT / "data/raw.dvc").read_text().splitlines()
        if "md5:" in line
    )
    tags = {
        "model_family": "mobilenetv2",
        "phase": args.phase,
        "logging_method": "tensorflow_autolog",
        "git_commit": commit,
        "mlflow.source.git.commit": commit,
        "dvc_raw_md5": raw_md5,
        "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
        "seed": str(config["seed"]),
        "device": "cpu",
        "test_evaluated": "false",
        "preprocessing_policy": POLICY_VERSION,
        **initialization,
    }
    with mlflow.start_run(run_name=name + "_cpu_v1", tags=tags) as run:
        mlflow.log_params(
            {
                **{f"phase.{key}": value for key, value in phase_config.items()},
                **{
                    f"transfer.{key}": value
                    for key, value in config.items()
                    if key not in ("frozen", "finetuned")
                },
                **{
                    f"preprocessing.{key}": value
                    for key, value in preprocessing.items()
                },
                **structure,
                "train_images": len(splits["train"]),
                "validation_images": len(splits["validation"]),
                "selection_metric": "val_loss",
                "threshold": 0.5,
                "intraop_threads": 2,
                "interop_threads": 1,
                "op_determinism": True,
                "oneDNN": False,
            }
        )
        mlflow.tensorflow.autolog(
            log_models=False, log_datasets=False, checkpoint=False, log_every_epoch=True
        )
        callbacks = [
            ResourceLog(),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                model_dir / "best.keras", monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=phase_config["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=phase_config["lr_patience"],
                factor=0.5,
                min_lr=1e-7,
            ),
            tf.keras.callbacks.CSVLogger(output / "history.csv"),
        ]
        try:
            history = model.fit(
                train,
                validation_data=validation,
                epochs=phase_config["epochs"],
                callbacks=callbacks,
                shuffle=False,
                verbose=2,
            ).history
        finally:
            mlflow.tensorflow.autolog(disable=True)
        history_path = output / "history.csv"
        history_path.write_bytes(history_path.read_bytes().replace(b"\r\n", b"\n"))
        check_space(ROOT, args.reserve_gib)
        best = tf.keras.models.load_model(model_dir / "best.keras")
        changed_layers = verify_backbone_changes(best, before, args.phase)
        write_json(
            output / "backbone_verification.json",
            {
                "phase": args.phase,
                "changed_backbone_layers": changed_layers,
                "frozen_weights_and_batchnorm_unchanged": True,
            },
        )
        probabilities = best.predict(validation, verbose=0).reshape(-1)
        labels = np.asarray([int(row["label"]) for row in splits["validation"]])
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        metrics = {
            **validation_metrics(labels, probabilities),
            "best_epoch": best_epoch,
            "epochs_completed": len(history["loss"]),
            "training_seconds": time.monotonic() - started,
            "parameters": model.count_params(),
            "trainable_parameters": structure["trainable_parameters"],
            "threshold": 0.5,
        }
        write_json(output / "metrics.json", metrics)
        mlflow.log_metrics(metrics)
        save_predictions(
            output / "validation_predictions.csv", splits["validation"], probabilities
        )
        make_phase_figures(
            output, history, best_epoch, labels, probabilities, args.phase
        )
        sample = next(iter(validation))[0][:2].numpy()
        expected = best(sample, training=False).numpy()
        mlflow.tensorflow.log_model(
            best,
            artifact_path="model",
            input_example=sample,
            pip_requirements=[f"tensorflow-cpu=={metadata.version('tensorflow-cpu')}"],
        )
        with tempfile.TemporaryDirectory(
            prefix="visionguard-transfer-model-"
        ) as temporary:
            restored = mlflow.tensorflow.load_model(
                f"runs:/{run.info.run_id}/model", dst_path=temporary
            )
            np.testing.assert_allclose(
                expected, restored(sample, training=False).numpy(), rtol=1e-5, atol=1e-6
            )
        check_space(ROOT, args.reserve_gib)
        record = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_url": f"{uri.removesuffix('.mlflow')}/experiments/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
            "git_commit": commit,
            "dvc_raw_md5": raw_md5,
            "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
            "model_sha256": hashlib.sha256(
                (model_dir / "best.keras").read_bytes()
            ).hexdigest(),
            "test_evaluated": False,
            "model_roundtrip": "passed",
            "logging_method": "tensorflow_autolog",
            "minimum_free_gib_observed": minimum_free[0] / 1024**3,
            "free_after_gib": shutil.disk_usage(ROOT).free / 1024**3,
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            **initialization,
        }
        write_json(output / "run.json", record)
        write_json(model_dir / "preprocessing.json", configuration)
        (output / "packages.txt").write_bytes(
            subprocess.check_output(
                [
                    os.sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "freeze",
                ]
            )
        )
        mlflow.log_artifacts(str(output), artifact_path="training")
        mlflow.log_artifacts(str(ROOT / "data/splits"), artifact_path="splits")
        mlflow.log_artifact(str(model_dir / "best.keras"), artifact_path="keras")
        mlflow.log_artifact(str(ROOT / "params.json"), artifact_path="source")
        mlflow.log_artifact(str(ROOT / "dvc.lock"), artifact_path="source")
        print(
            json.dumps(
                {"phase": args.phase, "metrics": metrics, "run": record}, indent=2
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
