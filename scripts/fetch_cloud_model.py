"""Fetch the pinned release bundle during a cloud build and verify every file."""

import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MAX_BUNDLE_BYTES = 24 * 1024**2
PUBLIC_BUNDLE_URL = (
    "https://github.com/Hanzala-Shadow/23L-0907_A2_Ops/"
    "releases/download/cloud-v1/champion-api.zip"
)


def allowed_bundle_url(url):
    parsed = urlsplit(url)
    return url == PUBLIC_BUNDLE_URL or (
        parsed.scheme == "https"
        and parsed.netloc == "release-assets.githubusercontent.com"
        and not parsed.fragment
    )


def unpack_bundle(data, manifest, destination):
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Cloud bundle exceeds its size limit")
    if hashlib.sha256(data).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Cloud bundle checksum failed")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Cloud model directory already exists")
    expected = {
        "model.tflite": manifest["model_sha256"],
        "policy.json": manifest["policy_sha256"],
    }
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sorted(archive.namelist()) != sorted(expected):
            raise ValueError("Cloud bundle contains unexpected files")
        for name, checksum in expected.items():
            if archive.getinfo(name).file_size > MAX_BUNDLE_BYTES:
                raise ValueError("Cloud model file exceeds its size limit")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError("Cloud model file checksum failed")
            files[name] = content
    destination.mkdir(parents=True)
    for name, content in files.items():
        (destination / name).write_bytes(content)


def main():
    url = os.environ.get("MODEL_BUNDLE_URL", "")
    if not allowed_bundle_url(url):
        raise SystemExit("MODEL_BUNDLE_URL must use the pinned GitHub release asset")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read(MAX_BUNDLE_BYTES + 1)
        manifest = json.loads((ROOT / "cloud/model-manifest.json").read_text())
        unpack_bundle(data, manifest, ROOT / "models/cloud-api")
    except (ValueError, OSError, urllib.error.URLError, zipfile.BadZipFile):
        # Signed URLs are credentials. Never include the URL or exception in logs.
        raise SystemExit(
            "Model download failed; check release availability and the pinned manifest"
        ) from None
    print("Cloud model bundle downloaded and verified")


if __name__ == "__main__":
    sys.exit(main())
