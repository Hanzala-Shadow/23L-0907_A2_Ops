"""Register two MobileNetV2 versions and promote the validation-selected model."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from scripts.train_cnn import check_space, configure_cpu, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_space(ROOT, margin_gib=0.25)
    configure_cpu()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import mlflow

    plan_path = ROOT / "reports/selection/plan.json"
    plan = json.loads(plan_path.read_text())
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    evaluation = json.loads((ROOT / "reports/evaluation/summary.json").read_text())
    if evaluation["plan_sha256"] != plan_sha or evaluation["selection_used_test"]:
        raise ValueError("Final evaluation does not use the locked selection plan")
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = mlflow.MlflowClient()
    name = plan["policy"]["registry_name"]
    versions = []
    for role in ("registry_reference", "champion"):
        record = plan["models"][role]
        tags = {
            "decision_threshold": str(record["threshold"]),
            "preprocessing_policy": plan["preprocessing"]["policy"],
            "image_size": str(plan["preprocessing"]["image_size"]),
            "source_checkpoint_sha256": record["model_sha256"],
            "selection_plan_sha256": plan_sha,
            "selection_basis": "validation_loss_then_validation_threshold",
            "role": role,
            "false_positive_cost": "2",
            "false_negative_cost": "1",
        }
        existing = client.search_model_versions(f"name='{name}'")
        matching = [
            version for version in existing if version.run_id == record["run_id"]
        ]
        if matching:
            version = matching[0]
            for key, value in tags.items():
                client.set_model_version_tag(name, version.version, key, value)
        else:
            version = mlflow.register_model(
                f"runs:/{record['run_id']}/model",
                name,
                tags=tags,
                await_registration_for=60,
            )
        version = client.get_model_version(name, version.version)
        if version.status != "READY" or version.run_id != record["run_id"]:
            raise RuntimeError(
                "Registered version is not ready or has incorrect lineage"
            )
        client.update_model_version(
            name,
            version.version,
            description=f"{record['model']}; threshold {record['threshold']:.6f} selected on validation using cost 2*FP+FN.",
        )
        versions.append(
            {
                "role": role,
                "version": version.version,
                "run_id": version.run_id,
                "source": version.source,
                "threshold": record["threshold"],
                "status": version.status,
            }
        )
        client.log_artifact(
            record["run_id"], str(plan_path), artifact_path="decision_policy"
        )
    champion = versions[-1]
    client.set_registered_model_alias(
        name, plan["policy"]["registry_alias"], champion["version"]
    )
    resolved = client.get_model_version_by_alias(name, plan["policy"]["registry_alias"])
    if resolved.version != champion["version"]:
        raise RuntimeError("Champion alias did not resolve to the selected version")
    client.update_registered_model(
        name,
        description="VisionGuard MobileNetV2 versions. Champion and thresholds were chosen on validation before a single final test evaluation. Outputs are probabilities for the dataset's AI-generated class.",
    )
    bundle = ROOT / "models/champion"
    bundle.mkdir(parents=True, exist_ok=True)
    selected = plan["models"]["champion"]
    checkpoint = ROOT / selected["model_path"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selected["model_sha256"]:
        raise ValueError("Selected checkpoint changed")
    shutil.copyfile(checkpoint, bundle / "best.keras")
    contract = {
        "registry_name": name,
        "registry_version": champion["version"],
        "alias": "champion",
        "run_id": selected["run_id"],
        "threshold": selected["threshold"],
        "preprocessing": plan["preprocessing"],
        "labels": plan["labels"],
        "model_sha256": selected["model_sha256"],
        "selection_plan_sha256": plan_sha,
    }
    write_json(bundle / "policy.json", contract)
    output = ROOT / "reports/registry"
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "versions.json",
        {
            "name": name,
            "alias": "champion",
            "champion_version": champion["version"],
            "versions": versions,
            "selection_plan_sha256": plan_sha,
        },
    )
    check_space(ROOT)
    print(
        json.dumps(
            {
                "registered_model": name,
                "champion_version": champion["version"],
                "versions": versions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
