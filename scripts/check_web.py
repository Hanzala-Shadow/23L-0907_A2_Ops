"""Check the running web app and PostgreSQL using validation images only."""

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import create_engine, select

from scripts.train_cnn import check_space, write_json
from visionguard.database import Prediction

ROOT = Path(__file__).resolve().parents[1]


class UploadForm(HTMLParser):
    token = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "input" and values.get("name") == "csrf_token":
            self.token = values["value"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--output-dir", default="reports/application")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check previously recorded rows without uploading again",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urlparse(base).hostname not in ("127.0.0.1", "localhost"):
        parser.error("This check is limited to the local demo")
    check_space(ROOT)
    load_dotenv(ROOT / ".env")
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    health = session.get(base + "/health", timeout=10)
    health.raise_for_status()
    assert health.json()["database"] == {"backend": "postgresql", "status": "ok"}
    assert health.json()["model"]["version"] == "2" and health.json()["debug"] is False
    if args.verify_only:
        previous = json.loads((output / "verification.json").read_text())
        records = previous["requests"]
    else:
        home = session.get(base + "/", timeout=10)
        home.raise_for_status()
        form = UploadForm()
        form.feed(home.text)
        assert form.token
        manifest = ROOT / "data/splits/validation.csv"
        summary = json.loads((ROOT / "data/splits/summary.json").read_text())
        assert (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            == summary["splits"]["validation"]["manifest_sha256"]
        )
        rows = list(csv.DictReader(manifest.open()))
        expected = {
            row["relative_path"]: float(row["ai_probability"])
            for row in csv.DictReader(
                (
                    ROOT
                    / "reports/training/keras_tuner/trial_7/validation_predictions.csv"
                ).open()
            )
        }
        records = []
        for label in ("0", "1"):
            row = next(row for row in rows if row["label"] == label)
            path = ROOT / "data/raw" / row["relative_path"]
            with Image.open(path) as image:
                assert image.width * image.height <= 16_000_000
            assert path.stat().st_size <= 8 * 1024**2
            with path.open("rb") as source:
                response = session.post(
                    base + "/predict",
                    data={"csrf_token": form.token},
                    files={
                        "image": ("../../validation-" + label + path.suffix, source)
                    },
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
            assert response.status_code == 201, response.text[:250]
            result = response.json()
            assert abs(result["ai_probability"] - expected[row["relative_path"]]) < 1e-5
            assert ".." not in result["filename"] and "/" not in result["filename"]
            assert (
                result["threshold"] == 0.6196606457233429
                and result["model_version"] == "2"
            )
            assert (
                session.get(base + "/result/" + result["id"], timeout=10).status_code
                == 200
            )
            records.append(
                {
                    "id": result["id"],
                    "kind": "validation_image",
                    "source": row["relative_path"],
                    "status": 201,
                    "ai_probability": result["ai_probability"],
                    "label": result["label"],
                }
            )
        for filename, data, status in (
            ("unsupported.txt", b"text", 415),
            ("corrupt.png", b"not an image", 400),
            ("oversized.png", b"x" * (9 * 1024**2), 413),
        ):
            response = session.post(
                base + "/predict",
                data={"csrf_token": form.token},
                files={"image": (filename, io.BytesIO(data))},
                headers={"Accept": "application/json"},
                timeout=30,
            )
            assert response.status_code == status, response.text[:250]
            assert response.json()["result_id"]
            records.append(
                {
                    "id": response.json()["result_id"],
                    "kind": "rejected_upload",
                    "status": status,
                }
            )
    engine = create_engine(os.environ["DATABASE_URL"], hide_parameters=True)
    with engine.connect() as connection:
        for record in records:
            import uuid

            stored = (
                connection.execute(
                    select(Prediction).where(Prediction.id == uuid.UUID(record["id"]))
                )
                .mappings()
                .one()
            )
            assert (
                stored["model_version"] == "2"
                and stored["threshold"] == 0.6196606457233429
            )
            assert stored["status"] == (
                "success" if record["status"] == 201 else "rejected"
            )
    engine.dispose()
    history = session.get(
        base + "/history", headers={"Accept": "application/json"}, timeout=10
    )
    history.raise_for_status()
    history_ids = {row["id"] for row in history.json()["items"]}
    assert {record["id"] for record in records} <= history_ids
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "base_url": base,
        "health": health.json(),
        "requests": records,
        "database_rows_verified": len(records),
        "test_images_used": False,
        "persistence_check": args.verify_only,
        "free_after_gib": check_space(ROOT) / 1024**3,
    }
    write_json(
        output / ("persistence.json" if args.verify_only else "verification.json"),
        report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
