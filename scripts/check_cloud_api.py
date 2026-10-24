"""Verify the deployed API with validation samples and safe rejection checks."""

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import dotenv_values, set_key
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = dotenv_values(ROOT / ".env.cloud")
    base_url = settings.get("MODEL_API_URL", "").rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname or "").endswith((".onrender.com", ".vercel.app"))
        or parsed.path
        or parsed.query
        or parsed.username
    ):
        raise SystemExit(
            "Set MODEL_API_URL to the HTTPS cloud service origin in .env.cloud"
        )
    headers = {"X-API-Key": settings["MODEL_API_KEY"]}
    manifest = json.loads((ROOT / "cloud/model-manifest.json").read_text())
    session = requests.Session()
    health = session.get(base_url + "/health", timeout=(10, 60), allow_redirects=False)
    health.raise_for_status()
    health_data = health.json()
    assert health_data["status"] == "ok"
    assert health_data["model"]["sha256"] == manifest["model_sha256"]
    assert health_data["model"]["threshold"] == manifest["threshold"]
    with (
        ROOT / "reports/training/keras_tuner/trial_7/validation_predictions.csv"
    ).open() as file:
        rows = list(csv.DictReader(file))
    cases = []
    for expected_label in (0, 1):
        for row in rows:
            expected = float(row["ai_probability"])
            if (
                int(row["label"]) != expected_label
                or int(expected >= manifest["threshold"]) != expected_label
            ):
                continue
            path = ROOT / "data/raw" / row["relative_path"]
            if path.stat().st_size > 4 * 1024**2:
                continue
            with Image.open(path) as image:
                if image.width * image.height > 4_000_000 or image.format not in (
                    "JPEG",
                    "PNG",
                ):
                    continue
                extension = ".png" if image.format == "PNG" else ".jpg"
            response = session.post(
                base_url + "/predict",
                headers=headers,
                files={"image": ("validation" + extension, path.read_bytes())},
                timeout=(10, 60),
                allow_redirects=False,
            )
            response.raise_for_status()
            result = response.json()
            error = abs(expected - result["ai_probability"])
            assert error < 1e-4
            assert result["label"] == ("ai_generated" if expected_label else "real")
            assert result["model"]["sha256"] == manifest["model_sha256"]
            assert result["threshold"] == manifest["threshold"]
            cases.append(
                {
                    "validation_class": expected_label,
                    "response": result,
                    "absolute_probability_error": error,
                    "http_status": response.status_code,
                }
            )
            break
        else:
            raise RuntimeError("No eligible validation sample for remote verification")
    checks = {
        "unauthenticated": session.post(
            base_url + "/predict", timeout=30, allow_redirects=False
        ).status_code,
        "missing_file": session.post(
            base_url + "/predict", headers=headers, timeout=30, allow_redirects=False
        ).status_code,
        "unsupported_file": session.post(
            base_url + "/predict",
            headers=headers,
            files={"image": ("sample.txt", b"invalid")},
            timeout=30,
            allow_redirects=False,
        ).status_code,
    }
    assert checks == {
        "unauthenticated": 401,
        "missing_file": 400,
        "unsupported_file": 415,
    }
    image = io.BytesIO()
    Image.new("RGB", (2001, 2000), "white").save(image, format="PNG")
    response = session.post(
        base_url + "/predict",
        headers=headers,
        files={"image": ("too-large.png", image.getvalue())},
        timeout=30,
        allow_redirects=False,
    )
    assert response.status_code == 413
    checks["pixel_limit"] = 413
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "provider": "Vercel" if parsed.hostname.endswith(".vercel.app") else "Render",
        "plan": "hobby" if parsed.hostname.endswith(".vercel.app") else "free",
        "base_url": base_url,
        "health": health_data,
        "prediction_checks": cases,
        "rejection_checks": checks,
        "test_set_used": False,
        "local_history_used": False,
    }
    (ROOT / "reports/cloud/deployment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    set_key(ROOT / ".env.cloud", "MODEL_API_URL", base_url)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
