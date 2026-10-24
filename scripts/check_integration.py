"""Verify local Flask to cloud inference and PostgreSQL with generated images."""

import hashlib
import io
import json
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import requests
from dotenv import dotenv_values
from PIL import Image
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_cloud_api import synthetic_images
from scripts.ci_smoke import UploadForm
from visionguard.database import HistoryStore
from visionguard.remote import RemoteModelService
from visionguard.web import create_app


def verify():
    if shutil.disk_usage(ROOT).free < 2.125 * 1024**3:
        raise SystemExit("Keep 2 GiB plus the upload guard buffer before verification")
    settings = dotenv_values(ROOT / ".env")
    before = json.loads(
        (ROOT / "reports/training-runtime/integration/before.json").read_text()
    )
    manifest = json.loads((ROOT / "cloud/model-manifest.json").read_text())
    session = requests.Session()
    base = "http://127.0.0.1:5000"
    health_response = session.get(base + "/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    assert health["model"]["backend"] == "vercel" and not health["model"]["loaded"]
    assert health["model"]["threshold"] == manifest["threshold"]
    form = UploadForm()
    page = session.get(base + "/", timeout=10)
    page.raise_for_status()
    form.feed(page.text)
    assert form.token and "sent securely to the cloud" in page.text
    reference = json.loads(
        subprocess.check_output(
            [
                str(ROOT / "reports/training-runtime/cloud/venv/bin/python"),
                str(ROOT / "scripts/check_cloud_api.py"),
                "--local-reference",
            ],
            cwd=ROOT,
            text=True,
        )
    )
    checks = []

    def upload(filename, data, expected_status):
        response = session.post(
            base + "/predict",
            data={"csrf_token": form.token},
            files={"image": (filename, data)},
            headers={"Accept": "application/json"},
            timeout=(5, 60),
        )
        assert response.status_code == expected_status, (filename, response.status_code)
        payload = response.json()
        identifier = payload["id"] if expected_status == 201 else payload["result_id"]
        result = session.get(base + "/result/" + identifier, timeout=10)
        result.raise_for_status()
        checks.append(
            {
                "filename": filename,
                "http_status": response.status_code,
                "result_id": identifier,
            }
        )
        return payload

    for color, data in synthetic_images().items():
        row = upload("integration-" + color + ".png", data, 201)
        assert abs(row["ai_probability"] - reference[color]["ai_probability"]) < 1e-4
        assert row["label"] == reference[color]["label"]
        assert row["model_sha256"] == manifest["model_sha256"]
        assert row["threshold"] == manifest["threshold"]
    upload("integration-invalid.txt", b"synthetic invalid upload", 415)
    image = io.BytesIO()
    Image.new("RGB", (2001, 2000), "white").save(image, format="PNG")
    upload("integration-pixel-limit.png", image.getvalue(), 413)

    # Faults are isolated to a test application sharing PostgreSQL; the live
    # container and deployed API are not interrupted or reconfigured.
    fault_checks = []
    for name, error, expected_status, code in [
        ("timeout", requests.ReadTimeout("simulated timeout"), 504, "cloud_timeout"),
        (
            "network",
            requests.ConnectionError("simulated outage"),
            503,
            "cloud_unavailable",
        ),
    ]:
        service = RemoteModelService(
            settings["MODEL_API_URL"],
            settings["MODEL_API_KEY"],
            ROOT / "models/champion/policy.json",
            ROOT / "cloud/model-manifest.json",
        )
        store = HistoryStore(settings["DATABASE_URL"])
        app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": settings["FLASK_SECRET_KEY"],
                "DATABASE_URL": settings["DATABASE_URL"],
                "INFERENCE_BACKEND": "remote",
            },
            model_service=service,
            store=store,
        )
        client = app.test_client()
        client.get("/")
        with client.session_transaction() as state:
            token = state["csrf_token"]
        with patch.object(requests.Session, "request", side_effect=error):
            response = client.post(
                "/predict",
                data={
                    "csrf_token": token,
                    "image": (
                        io.BytesIO(synthetic_images()["red"]),
                        "integration-" + name + ".png",
                    ),
                },
                headers={"Accept": "application/json"},
            )
        assert response.status_code == expected_status
        identifier = response.json["result_id"]
        with store.engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        "SELECT status, error_code, ai_probability, label FROM predictions WHERE id=:id"
                    ),
                    {"id": uuid.UUID(identifier)},
                )
                .mappings()
                .one()
            )
            assert row["status"] == "error" and row["error_code"] == code
            assert row["label"] is None and row["ai_probability"] is None
        assert client.get("/history").status_code == 200
        fault_checks.append(
            {
                "case": name,
                "http_status": expected_status,
                "error_code": code,
                "result_id": identifier,
            }
        )
        store.engine.dispose()

    engine = create_engine(settings["DATABASE_URL"], hide_parameters=True)
    with engine.connect() as connection:
        rows = json.loads(
            json.dumps(
                [
                    dict(r)
                    for r in connection.execute(
                        text("SELECT * FROM predictions ORDER BY id")
                    ).mappings()
                ],
                default=str,
            )
        )
    engine.dispose()
    by_id = {row["id"]: row for row in rows}
    assert all(by_id.get(row["id"]) == row for row in before["rows"])
    for name, key in [("best.keras", "model_sha256"), ("policy.json", "policy_sha256")]:
        assert (
            hashlib.sha256((ROOT / "models/champion" / name).read_bytes()).hexdigest()
            == before[key]
        )
    history = session.get(
        base + "/history", headers={"Accept": "application/json"}, timeout=10
    )
    history.raise_for_status()
    assert history.json()["total"] == len(rows)
    free = shutil.disk_usage(ROOT).free / 1024**3
    assert free >= 2
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "local_url": base,
        "inference_backend": "vercel",
        "cloud_url": settings["MODEL_API_URL"],
        "health": health,
        "live_requests": checks,
        "isolated_postgresql_fault_checks": fault_checks,
        "original_rows_preserved": len(before["rows"]),
        "history_rows_after_checks": len(rows),
        "native_champion_unchanged": True,
        "free_disk_gib": free,
        "dataset_images_uploaded": False,
        "test_set_used": False,
    }
    output = ROOT / "reports/integration"
    output.mkdir(exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
