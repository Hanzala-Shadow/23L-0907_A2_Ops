"""Run eight CPU Keras Tuner trials with one DagsHub MLflow run per trial."""

import argparse
import csv
import gc
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
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.reserve_gib < 2:
        parser.error("The minimum reserve is 2 GiB")
    check_space(ROOT, args.reserve_gib, 0.5)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import keras_tuner as kt
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    import tensorflow as tf

    from scripts.prepare_splits import verify_source
    from visionguard.pipeline import make_dataset, read_split, verify_split_hash
    from visionguard.preprocessing import POLICY_VERSION
    from visionguard.transfer import backbone_fingerprints
    from visionguard.transfer_reporting import (
        comparison_records,
        make_phase_figures,
        save_predictions,
        save_structure,
        validation_metrics,
    )
    from visionguard.tuning import (
        assert_complete_search,
        factor_effects,
        load_trial_model,
        search_space,
        verify_trial_weights,
    )

    params = json.loads((ROOT / "params.json").read_text())
    config, preprocessing = params["tuning"], params["preprocessing"]
    space = search_space(config)
    if preprocessing["policy"] != POLICY_VERSION:
        raise ValueError("Unknown preprocessing policy")
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    if set(changed) - {"dvc.lock"}:
        raise RuntimeError("Commit tracked code and parameters before tuning")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com/") or not all(
        os.environ.get(key)
        for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
    ):
        raise ValueError("Configure DagsHub MLflow in the local environment")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ["PROJECT_NAME"])
    mlflow.tensorflow.autolog(disable=True)
    client = mlflow.MlflowClient()
    summary = json.loads((ROOT / "data/splits/summary.json").read_text())
    with (ROOT / "reports/data/image_manifest.csv").open(newline="") as source:
        manifest = list(csv.DictReader(source))
    raw = ROOT / "data/raw"
    if verify_source(manifest, raw) != summary["raw_dataset_fingerprint_sha256"]:
        raise ValueError("Dataset differs from its recorded version")
    splits = {}
    for name in ("train", "validation"):
        path = ROOT / f"data/splits/{name}.csv"
        verify_split_hash(path, summary["splits"][name]["manifest_sha256"])
        splits[name] = read_split(path, raw)
    if {row["sha256"] for row in splits["train"]} & {
        row["sha256"] for row in splits["validation"]
    }:
        raise ValueError("Training and validation overlap")
    # Decode once into RAM. A new seeded shuffle is created for every trial.
    pixels = make_dataset(splits["train"], raw, preprocessing).unbatch()
    validation = make_dataset(splits["validation"], raw, preprocessing)
    labels = np.asarray([int(row["label"]) for row in splits["validation"]])
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
    if initialization["initialized_from_model_sha256"] != initial_run["model_sha256"]:
        raise ValueError("Frozen checkpoint differs from its recorded version")
    runtime = ROOT / "reports/training-runtime/keras_tuner" / commit
    runtime.mkdir(parents=True, exist_ok=True)
    output = ROOT / "reports/training/keras_tuner"
    model_dir = ROOT / "models/mobilenetv2_tuned"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    minimum_free = [check_space(ROOT, args.reserve_gib)]

    def guard():
        minimum_free[0] = min(minimum_free[0], check_space(ROOT, args.reserve_gib))

    class DagsHubSearch(kt.RandomSearch):
        def load_model(self, trial):
            return tf.keras.models.load_model(
                Path(self.get_trial_dir(trial.trial_id)) / "best.keras"
            )

        def run_trial(self, trial, **kwargs):
            guard()
            folder = Path(self.get_trial_dir(trial.trial_id))
            report = folder / "report"
            record_path = report / "run.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if client.get_run(record["run_id"]).info.status == "FINISHED":
                    return {"val_loss": record["validation_loss"]}
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(config["seed"])
            values = trial.hyperparameters.values
            model, selected = load_trial_model(initial_path, values)
            before = backbone_fingerprints(model)
            report.mkdir(parents=True, exist_ok=True)
            save_structure(model, report)
            configuration = {
                "hyperparameters": values,
                "search": config,
                "preprocessing": preprocessing,
                "label_mapping": summary["labels"],
                "input_dtype": "uint8",
                "normalization": "pixel / 127.5 - 1 inside model",
                "trainable_backbone_layers": selected,
                "batchnorm_training_mode": False,
                "test_evaluated": False,
                **initialization,
            }
            write_json(report / "configuration.json", configuration)
            train = (
                pixels.shuffle(
                    len(splits["train"]),
                    seed=config["seed"],
                    reshuffle_each_iteration=True,
                )
                .batch(preprocessing["batch_size"])
                .prefetch(1)
            )
            started = time.monotonic()

            class EpochLog(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, logs=None):
                    guard()

                def on_epoch_end(self, epoch, logs=None):
                    guard()
                    metrics = {key: float(value) for key, value in logs.items()}
                    mlflow.log_metrics(metrics, step=epoch + 1)
                    print(
                        json.dumps(
                            {"trial": trial.trial_id, "epoch": epoch + 1, **metrics}
                        ),
                        flush=True,
                    )

            tags = {
                "model_family": "mobilenetv2",
                "phase": "keras_tuner",
                "trial_id": trial.trial_id,
                "git_commit": commit,
                "mlflow.source.git.commit": commit,
                "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
                "dvc_raw_md5": initial_run["dvc_raw_md5"],
                "test_evaluated": "false",
                "device": "cpu",
                "logging_method": "manual",
                "preprocessing_policy": POLICY_VERSION,
                **initialization,
            }
            print(json.dumps({"trial_start": trial.trial_id, **values}), flush=True)
            with mlflow.start_run(
                run_name=f"mobilenetv2_tuner_{trial.trial_id}", tags=tags
            ) as run:
                mlflow.log_params(
                    {
                        **values,
                        "seed": config["seed"],
                        "epochs": config["epochs"],
                        "early_stopping_patience": config["early_stopping_patience"],
                        "lr_patience": config["lr_patience"],
                        "objective": "val_loss",
                        "train_images": len(splits["train"]),
                        "validation_images": len(splits["validation"]),
                        "batch_size": preprocessing["batch_size"],
                        "image_size": preprocessing["image_size"],
                        "threshold": 0.5,
                        "trainable_backbone_layers": selected,
                        "batchnorm_training_mode": False,
                        "intraop_threads": 2,
                        "interop_threads": 1,
                        "op_determinism": True,
                        "oneDNN": False,
                    }
                )
                history = model.fit(
                    train,
                    validation_data=validation,
                    epochs=config["epochs"],
                    shuffle=False,
                    verbose=0,
                    callbacks=[
                        EpochLog(),
                        tf.keras.callbacks.TerminateOnNaN(),
                        tf.keras.callbacks.ModelCheckpoint(
                            folder / "best.keras",
                            monitor="val_loss",
                            save_best_only=True,
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
                            min_lr=1e-7,
                        ),
                        tf.keras.callbacks.CSVLogger(report / "history.csv"),
                    ],
                ).history
                history_path = report / "history.csv"
                history_path.write_bytes(
                    history_path.read_bytes().replace(b"\r\n", b"\n")
                )
                guard()
                best = self.load_model(trial)
                changed_layers = verify_trial_weights(best, before, selected)
                write_json(
                    report / "backbone_verification.json",
                    {
                        "changed_backbone_layers": changed_layers,
                        "frozen_weights_and_batchnorm_unchanged": True,
                    },
                )
                probabilities = best.predict(validation, verbose=0).reshape(-1)
                best_epoch = int(np.argmin(history["val_loss"])) + 1
                metrics = {
                    **validation_metrics(labels, probabilities),
                    "best_epoch": best_epoch,
                    "epochs_completed": len(history["loss"]),
                    "training_seconds": time.monotonic() - started,
                    "best_epoch_training_loss": float(history["loss"][best_epoch - 1]),
                    "last_validation_loss": float(history["val_loss"][-1]),
                    "parameters": model.count_params(),
                    "trainable_parameters": sum(
                        int(tf.size(w)) for w in model.trainable_weights
                    ),
                    "threshold": 0.5,
                }
                write_json(report / "metrics.json", metrics)
                save_predictions(
                    report / "validation_predictions.csv",
                    splits["validation"],
                    probabilities,
                )
                make_phase_figures(
                    report,
                    history,
                    best_epoch,
                    labels,
                    probabilities,
                    f"trial {trial.trial_id}",
                )
                mlflow.log_metrics(metrics)
                record = {
                    **metrics,
                    **initialization,
                    "trial_id": trial.trial_id,
                    "hyperparameters": dict(values),
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "run_url": f"{uri.removesuffix('.mlflow')}/experiments/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
                    "git_commit": commit,
                    "data_fingerprint": summary["raw_dataset_fingerprint_sha256"],
                    "dvc_raw_md5": initial_run["dvc_raw_md5"],
                    "split_counts": {name: len(rows) for name, rows in splits.items()},
                    "model_sha256": hashlib.sha256(
                        (folder / "best.keras").read_bytes()
                    ).hexdigest(),
                    "test_evaluated": False,
                    "minimum_free_gib_observed": minimum_free[0] / 1024**3,
                }
                write_json(record_path, record)
                mlflow.log_artifacts(str(report), artifact_path="training")
                mlflow.log_artifacts(str(ROOT / "data/splits"), artifact_path="splits")
                mlflow.log_artifact(str(folder / "best.keras"), artifact_path="keras")
                mlflow.log_artifact(str(ROOT / "params.json"), artifact_path="source")
                print(
                    json.dumps(
                        {
                            "trial_complete": trial.trial_id,
                            "run_id": run.info.run_id,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
            return {"val_loss": metrics["validation_loss"]}

    tuner = DagsHubSearch(
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=config["max_trials"],
        hyperparameters=space,
        allow_new_entries=False,
        tune_new_entries=False,
        seed=config["seed"],
        directory=str(runtime),
        project_name="search",
        overwrite=False,
        max_retries_per_trial=1,
    )
    tuner.search(verbose=0)
    records = []
    for trial_id, trial in sorted(tuner.oracle.trials.items()):
        if trial.status != "COMPLETED":
            raise RuntimeError(f"Trial {trial_id} did not complete")
        folder = Path(tuner.get_trial_dir(trial_id))
        record = json.loads((folder / "report/run.json").read_text())
        if client.get_run(record["run_id"]).info.status != "FINISHED":
            raise RuntimeError("A trial was not logged successfully to DagsHub")
        records.append(record)
        shutil.copytree(
            folder / "report", output / f"trial_{trial_id}", dirs_exist_ok=True
        )
    assert_complete_search(records, config)
    best_trial = tuner.oracle.get_best_trials(1)[0]
    selected = next(
        record for record in records if record["trial_id"] == best_trial.trial_id
    )
    guard()
    shutil.copyfile(
        Path(tuner.get_trial_dir(best_trial.trial_id)) / "best.keras",
        model_dir / "best.keras",
    )
    shutil.copyfile(
        output / f"trial_{best_trial.trial_id}/configuration.json",
        model_dir / "preprocessing.json",
    )
    best = tuner.load_model(best_trial)
    with mlflow.start_run(run_id=selected["run_id"]):
        mlflow.set_tag("best_tuning_trial", "true")
        sample = next(iter(validation))[0][:2].numpy()
        expected = best(sample, training=False).numpy()
        mlflow.tensorflow.log_model(
            best,
            artifact_path="model",
            input_example=sample,
            pip_requirements=[f"tensorflow-cpu=={metadata.version('tensorflow-cpu')}"],
        )
        with tempfile.TemporaryDirectory(
            prefix="visionguard-tuning-model-"
        ) as temporary:
            restored = mlflow.tensorflow.load_model(
                f"runs:/{selected['run_id']}/model", dst_path=temporary
            )
            np.testing.assert_allclose(
                expected, restored(sample, training=False).numpy(), rtol=1e-5, atol=1e-6
            )
        effects = factor_effects(records, config)
        previous = comparison_records(ROOT)
        previous.append(
            {
                "model": "mobilenetv2_tuned",
                **selected,
                "model_path": "models/mobilenetv2_tuned/best.keras",
            }
        )
        provisional = min(previous, key=lambda row: row["validation_loss"])
        payload = {
            "search_method": "Keras Tuner RandomSearch, complete two-level factorial grid",
            "objective": "validation_loss",
            "completed_trials": len(records),
            "best_trial_id": selected["trial_id"],
            "best_run_id": selected["run_id"],
            "best_hyperparameters": selected["hyperparameters"],
            "best_validation_loss": selected["validation_loss"],
            "best_validation_accuracy": selected["validation_accuracy"],
            "factor_effects": effects,
            "largest_marginal_effect": max(
                effects, key=lambda name: abs(effects[name]["higher_minus_lower"])
            ),
            "provisional_selected_model": provisional["model"],
            "selected_model_path": provisional["model_path"],
            "test_evaluated": False,
            "threshold": 0.5,
            "minimum_free_gib_observed": minimum_free[0] / 1024**3,
            "free_after_gib": shutil.disk_usage(ROOT).free / 1024**3,
            "selected_mlflow_model_roundtrip": "passed",
            "models": previous,
            "trials": records,
        }
        write_json(output / "summary.json", payload)
        columns = [
            "trial_id",
            "learning_rate",
            "dropout",
            "unfrozen_layers",
            "validation_loss",
            "validation_accuracy",
            "validation_f1",
            "validation_roc_auc",
            "best_epoch",
            "epochs_completed",
            "run_id",
        ]
        with (output / "trials.csv").open("w", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows({**row, **row["hyperparameters"]} for row in records)
        for filename in ("oracle.json", "tuner0.json"):
            shutil.copyfile(Path(tuner.project_dir) / filename, output / filename)
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
        mlflow.log_artifacts(str(output), artifact_path="search")
    guard()
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in ("models", "trials")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
