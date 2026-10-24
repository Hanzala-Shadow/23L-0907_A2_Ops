"""Prepare a short-lived, model-only release download for a manual deployment."""

import json
import os
import secrets
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import dotenv_values, set_key

REPO = "Hanzala-Shadow/23L-0907_A2_Ops"
ROOT = Path(__file__).resolve().parents[1]


def main():
    token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.get(
        f"https://api.github.com/repos/{REPO}/releases/tags/cloud-v1",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    release = response.json()
    asset = next(a for a in release["assets"] if a["name"] == "champion-api.zip")
    manifest = json.loads((ROOT / "cloud/model-manifest.json").read_text())
    if asset.get("digest") != "sha256:" + manifest["bundle_sha256"]:
        raise SystemExit(
            "Private release asset digest differs from the deployment manifest"
        )
    response = requests.get(
        asset["url"],
        headers={
            **headers,
            "Accept": "application/octet-stream",
            "Cache-Control": "no-cache",
        },
        params={"refresh": secrets.token_hex(8)},
        allow_redirects=False,
        timeout=30,
    )
    url = response.headers.get("Location", "")
    if (
        response.status_code != 302
        or urlsplit(url).hostname != "release-assets.githubusercontent.com"
    ):
        raise SystemExit("GitHub did not return a signed release download URL")
    path = ROOT / ".env.cloud"
    if not path.exists():
        path.touch(mode=0o600)
    os.chmod(path, 0o600)
    values = dotenv_values(path)
    set_key(path, "MODEL_BUNDLE_URL", url)
    if not values.get("MODEL_API_KEY"):
        set_key(path, "MODEL_API_KEY", secrets.token_urlsafe(32))
    print("Saved signed model download URL and API key in .env.cloud (mode 0600)")
    print("The URL expires in about one hour; refresh it before each cloud build")


if __name__ == "__main__":
    main()
