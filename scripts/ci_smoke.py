"""Exercise a disposable Compose stack using synthetic images only."""

import argparse
import binascii
import json
import math
import struct
import uuid
import zlib
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests


class UploadForm(HTMLParser):
    token = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "input" and values.get("name") == "csrf_token":
            self.token = values["value"]


def png(color):
    def chunk(kind, value):
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", binascii.crc32(kind + value) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 16, 8, 8, 2, 0, 0, 0)
    pixels = (b"\0" + bytes(color) * 16) * 8
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def multipart(token, filename, data):
    boundary = "ci-" + uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="csrf_token"\r\n\r\n{token}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, "multipart/form-data; boundary=" + boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/training-runtime/ci/docker-smoke.json"),
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urlparse(base).hostname not in ("localhost", "127.0.0.1"):
        parser.error("CI checks are limited to a local disposable stack")
    client = requests.Session()

    def request(path, data=None, headers=None):
        with client.request(
            "POST" if data is not None else "GET",
            base + path,
            data=data,
            headers=headers,
            timeout=20,
        ) as response:
            return response.status_code, response.content

    status, data = request("/health")
    health = json.loads(data)
    assert status == 200 and health["model"]["version"] == "synthetic"
    assert health["model"]["threshold"] == 0.62 and health["debug"] is False
    assert health["database"] == {"backend": "postgresql", "status": "ok"}
    assert health["disk"]["free_gib"] >= 2
    if args.verify_only:
        records = json.loads(args.output.read_text())["requests"]
    else:
        status, page = request("/")
        assert status == 200
        form = UploadForm()
        form.feed(page.decode())
        assert form.token
        records = []
        cases = [
            ("red.png", png((255, 0, 0)), 201, "real", 1 / (1 + math.exp(2))),
            ("blue.png", png((0, 0, 255)), 201, "ai_generated", 1 / (1 + math.exp(-2))),
            ("wrong.txt", b"text", 415, None, None),
            ("corrupt.png", b"not an image", 400, None, None),
            ("large.png", b"x" * (9 * 1024**2), 413, None, None),
        ]
        for filename, image, expected_status, label, probability in cases:
            body, content_type = multipart(form.token, filename, image)
            status, data = request(
                "/predict",
                body,
                {"Accept": "application/json", "Content-Type": content_type},
            )
            assert status == expected_status, (filename, status)
            result = json.loads(data)
            identifier = result["id"] if status == 201 else result["result_id"]
            if status == 201:
                assert result["label"] == label and result["threshold"] == 0.62
                assert abs(result["ai_probability"] - probability) < 1e-5
            assert request("/result/" + identifier)[0] == 200
            records.append({"id": identifier, "status": status, "label": label})
    status, data = request("/history", headers={"Accept": "application/json"})
    history = json.loads(data)
    assert status == 200 and history["total"] == 5
    saved = {row["id"]: row for row in history["items"]}
    for record in records:
        row = saved[record["id"]]
        assert row["model_version"] == "synthetic" and row["threshold"] == 0.62
        assert row["status"] == ("success" if record["status"] == 201 else "rejected")
        assert row["label"] == record["label"]
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "fixture": "synthetic",
        "requests": records,
        "history_rows": 5,
        "recreation_verified": args.verify_only,
        "health": health,
        "test_images_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
