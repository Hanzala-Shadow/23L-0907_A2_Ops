"""Export the champion to float32 LiteRT and check validation parity."""

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visionguard.preprocessing import preprocess_image
from visionguard.serving import configure_cpu


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if shutil.disk_usage(ROOT).free < (2 * 1024**3 + 256 * 1024**2):
        raise SystemExit("Export needs the 2 GiB reserve plus 256 MiB headroom")
    output = ROOT / "models/cloud-api"
    if output.exists():
        raise SystemExit(
            "Cloud bundle already exists; preserve it or remove it explicitly"
        )
    configure_cpu()
    import numpy as np
    import tensorflow as tf

    source = ROOT / "models/champion"
    policy = json.loads((source / "policy.json").read_text())
    if digest(source / "best.keras") != policy["model_sha256"]:
        raise SystemExit("Champion checkpoint hash differs from its saved policy")
    model = tf.keras.models.load_model(source / "best.keras", compile=False)
    with tempfile.TemporaryDirectory(prefix="visionguard-export-") as temp:
        archive = tf.keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(
            name="serve",
            fn=lambda pixels: model(pixels, training=False),
            input_signature=[tf.TensorSpec((1, 128, 128, 3), tf.uint8)],
        )
        archive.write_out(temp, verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(temp)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converted = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=converted, num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    errors, changed = [], 0
    with (ROOT / "data/splits/validation.csv").open() as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        pixels = preprocess_image(
            ROOT / "data/raw" / row["relative_path"],
            policy["preprocessing"]["image_size"],
        )[None]
        original = float(model(pixels, training=False).numpy().reshape(-1)[0])
        interpreter.set_tensor(input_detail["index"], pixels)
        interpreter.invoke()
        exported = float(interpreter.get_tensor(output_detail["index"]).reshape(-1)[0])
        errors.append(abs(original - exported))
        changed += (original >= policy["threshold"]) != (
            exported >= policy["threshold"]
        )
    if max(errors) > 1e-4 or changed:
        raise SystemExit(
            f"Export parity failed: max error {max(errors)}, changed {changed}"
        )
    output.mkdir()
    (output / "model.tflite").write_bytes(converted)
    policy["source_model_sha256"] = policy.pop("model_sha256")
    policy["model_sha256"] = digest(output / "model.tflite")
    policy["format"] = "tflite-float32"
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    report = {
        "format": policy["format"],
        "source_model_sha256": policy["source_model_sha256"],
        "model_sha256": policy["model_sha256"],
        "policy_sha256": digest(output / "policy.json"),
        "model_bytes": len(converted),
        "threshold": policy["threshold"],
        "validation_images": len(rows),
        "maximum_absolute_probability_error": max(errors),
        "mean_absolute_probability_error": float(np.mean(errors)),
        "changed_validation_decisions": int(changed),
        "test_set_used": False,
        "tensorflow_version": tf.__version__,
        "free_disk_gib_after_export": shutil.disk_usage(ROOT).free / 1024**3,
    }
    target = ROOT / "reports/cloud"
    target.mkdir(exist_ok=True)
    (target / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    bundle = ROOT / "reports/training-runtime/cloud/champion-api.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.tflite", "policy.json"):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, (output / name).read_bytes())
    manifest = {
        key: report[key]
        for key in (
            "source_model_sha256",
            "model_sha256",
            "policy_sha256",
            "format",
            "threshold",
        )
    }
    manifest.update(bundle_sha256=digest(bundle), bundle_bytes=bundle.stat().st_size)
    (ROOT / "cloud/model-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
