"""Exercise the real LiteRT API using validation images and record peak memory."""

import csv
import io
import json
import resource
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visionguard.cloud import create_app
from visionguard.uploads import FORMATS


def main():
    app = create_app({"TESTING": True, "MODEL_API_KEY": "runtime-check-secret-" * 3})
    client = app.test_client()
    headers = {"X-API-Key": app.config["MODEL_API_KEY"]}
    with (
        ROOT / "reports/training/keras_tuner/trial_7/validation_predictions.csv"
    ).open() as file:
        rows = list(csv.DictReader(file))
    maximum_error, changed, classified, rejected, mismatches = 0.0, 0, 0, 0, 0
    for row in rows:
        path = ROOT / "data/raw" / row["relative_path"]
        response = client.post(
            "/predict",
            headers=headers,
            data={"image": (io.BytesIO(path.read_bytes()), path.name)},
        )
        with Image.open(path) as image:
            over_limit = (
                path.stat().st_size > app.config["MAX_UPLOAD_BYTES"]
                or image.width * image.height > app.config["MAX_IMAGE_PIXELS"]
            )
            mismatch = image.format != FORMATS.get(path.suffix.lower())
        if over_limit:
            assert response.status_code == 413
            rejected += 1
            continue
        if mismatch:
            expected_status = 400 if path.suffix.lower() in FORMATS else 415
            assert response.status_code == expected_status
            mismatches += 1
            continue
        if response.status_code != 200:
            raise RuntimeError(
                f"Validation upload failed with status {response.status_code}"
            )
        result = response.json
        classified += 1
        expected = float(row["ai_probability"])
        maximum_error = max(maximum_error, abs(expected - result["ai_probability"]))
        changed += (expected >= result["threshold"]) != (
            result["label"] == "ai_generated"
        )
    assert maximum_error <= 1e-4 and changed == 0
    assert client.post("/predict").status_code == 401
    assert client.post("/predict", headers=headers).status_code == 400
    assert client.get("/health").status_code == 200
    report = {
        "runtime": "ai-edge-litert==2.3.0",
        "validation_uploads": len(rows),
        "classified_validation_images": classified,
        "rejected_over_limit_images": rejected,
        "rejected_format_mismatches": mismatches,
        "maximum_absolute_probability_error": maximum_error,
        "changed_validation_decisions": int(changed),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "free_disk_gib": shutil.disk_usage(ROOT).free / 1024**3,
        "test_set_used": False,
        "tensorflow_imported": "tensorflow" in sys.modules,
    }
    assert report["peak_process_rss_mib"] < 400
    assert report["free_disk_gib"] >= 2
    assert not report["tensorflow_imported"]
    (ROOT / "reports/cloud/runtime.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
