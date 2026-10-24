"""Verify the cloud API using generated images; dataset files stay local."""

import importlib.util
import io
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def synthetic_images():
    images = {}
    for color in ("red", "blue", "white"):
        output = io.BytesIO()
        Image.new("RGB", (160, 96), color).save(output, format="PNG")
        images[color] = output.getvalue()
    return images


def local_reference():
    sys.path.insert(0, str(ROOT))
    from visionguard.cloud import load_cloud_bundle

    predictor, _ = load_cloud_bundle(
        ROOT / "models/cloud-api", ROOT / "cloud/model-manifest.json"
    )
    return {
        name: predictor.predict(io.BytesIO(data))
        for name, data in synthetic_images().items()
    }


def main():
    import requests
    from dotenv import dotenv_values, set_key

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
    fallback = ROOT / "reports/training-runtime/cloud/venv/bin/python"
    runtime = os.environ.get(
        "CLOUD_RUNTIME_PYTHON",
        sys.executable if importlib.util.find_spec("ai_edge_litert") else str(fallback),
    )
    reference = subprocess.run(
        [runtime, str(Path(__file__).resolve()), "--local-reference"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(reference.stdout)
    cases = []
    for name, data in synthetic_images().items():
        response = session.post(
            base_url + "/predict",
            headers=headers,
            files={"image": ("synthetic-" + name + ".png", data)},
            timeout=(10, 60),
            allow_redirects=False,
        )
        response.raise_for_status()
        result = response.json()
        error = abs(expected[name]["ai_probability"] - result["ai_probability"])
        assert error < 1e-4
        assert result["label"] == expected[name]["label"]
        assert result["model"]["sha256"] == manifest["model_sha256"]
        assert result["threshold"] == manifest["threshold"]
        cases.append(
            {
                "synthetic_image": name,
                "response": result,
                "absolute_probability_error": error,
                "http_status": response.status_code,
            }
        )
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
        "dataset_images_uploaded": False,
        "test_set_used": False,
        "local_history_used": False,
    }
    (ROOT / "reports/cloud/deployment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    set_key(ROOT / ".env.cloud", "MODEL_API_URL", base_url)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--local-reference"]:
        print(json.dumps(local_reference()))
    else:
        main()
