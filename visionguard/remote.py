"""Bounded server-to-server inference against the pinned cloud champion."""

import io
import json
import math
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image

from visionguard import PROJECT_NAME, PUBLIC_API_ORIGIN

API_ORIGIN = PUBLIC_API_ORIGIN
LEGACY_API_ORIGIN = "https://23l-0907-a2-ops-api.vercel.app"
MAX_RESPONSE_BYTES = 16 * 1024


class RemoteInferenceError(RuntimeError):
    def __init__(self, code, message, status=503):
        super().__init__(message)
        self.code, self.status = code, status


def bad_response():
    return RemoteInferenceError(
        "cloud_invalid_response",
        "The cloud model returned an unexpected result. Please try again later.",
        502,
    )


class RemoteModelService:
    backend = "remote"
    load_count = 0

    def __init__(
        self, url, key, policy_path, manifest_path, connect_timeout=3, read_timeout=45
    ):
        parsed = urlsplit(url)
        if (
            url.rstrip("/") not in {API_ORIGIN, LEGACY_API_ORIGIN}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("MODEL_API_URL must be the deployed Vercel HTTPS origin")
        if (
            not isinstance(key, str)
            or not 32 <= len(key) <= 256
            or not key.isascii()
            or any(c.isspace() for c in key)
        ):
            raise ValueError("Set a valid MODEL_API_KEY for remote inference")
        self.connect_timeout, self.read_timeout = (
            float(connect_timeout),
            float(read_timeout),
        )
        if (
            not math.isfinite(self.connect_timeout)
            or not 0 < self.connect_timeout <= 10
        ):
            raise ValueError("Cloud connect timeout must be in (0, 10] seconds")
        if not math.isfinite(self.read_timeout) or not 0 < self.read_timeout <= 60:
            raise ValueError("Cloud read timeout must be in (0, 60] seconds")
        original = json.loads(Path(policy_path).read_text())
        manifest = json.loads(Path(manifest_path).read_text())
        if (
            original["model_sha256"] != manifest["source_model_sha256"]
            or original["threshold"] != manifest["threshold"]
            or original["preprocessing"]["policy"] != "rgb-white-letterbox-v1"
            or original["preprocessing"]["image_size"] != 128
            or original["labels"] != {"real": 0, "ai_generated": 1}
            or manifest["format"] != "tflite-float32"
        ):
            raise ValueError(
                "The cloud manifest does not match the saved champion policy"
            )
        threshold = float(original["threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The saved cloud threshold is invalid")
        self.policy = {
            **original,
            "source_model_sha256": original["model_sha256"],
            "model_sha256": manifest["model_sha256"],
            "format": manifest["format"],
        }
        self.url, self._key = url.rstrip("/"), key
        self._sessions = threading.local()
        self._state_lock = threading.Lock()
        self._health, self._health_until, self._auth_failed = None, 0, False

    def _session(self):
        if not hasattr(self._sessions, "session"):
            session = requests.Session()
            session.trust_env = False
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            self._sessions.session = session
        return self._sessions.session

    def _request(self, method, path, **kwargs):
        timeout = kwargs.pop("timeout", (self.connect_timeout, self.read_timeout))
        try:
            with self._session().request(
                method,
                self.url + path,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
                **kwargs,
            ) as response:
                status = response.status_code
                if status in (401, 403):
                    self._auth_failed = True
                    raise RemoteInferenceError(
                        "cloud_authentication",
                        "The cloud model connection needs attention. Please try again later.",
                    )
                if status == 429 or status == 503:
                    raise RemoteInferenceError(
                        "cloud_busy",
                        "The cloud model is temporarily busy. Please try again shortly.",
                    )
                if status >= 500:
                    raise RemoteInferenceError(
                        "cloud_unavailable",
                        "The cloud model is unavailable. Please try again later.",
                    )
                if status in (400, 413, 415):
                    raise RemoteInferenceError(
                        "cloud_rejected_upload",
                        "The cloud model could not accept this image. Try a smaller still image.",
                        status,
                    )
                if (
                    status != 200
                    or response.headers.get("Content-Type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                    != "application/json"
                ):
                    raise bad_response()
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=4096):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise bad_response()
                    chunks.append(chunk)
                try:
                    result = json.loads(b"".join(chunks))
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise bad_response() from error
                if not isinstance(result, dict):
                    raise bad_response()
                return result
        except requests.Timeout as error:
            raise RemoteInferenceError(
                "cloud_timeout",
                "The cloud model took too long to respond. Please try again.",
                504,
            ) from error
        except requests.RequestException as error:
            raise RemoteInferenceError(
                "cloud_unavailable",
                "The cloud model could not be reached. Check the connection and try again.",
            ) from error

    def _validate_model(self, value):
        expected = {
            "version": self.policy["registry_version"],
            "threshold": self.policy["threshold"],
            "format": self.policy["format"],
            "sha256": self.policy["model_sha256"],
            "source_sha256": self.policy["source_model_sha256"],
        }
        if (
            not isinstance(value, dict)
            or value.get("name") not in {PROJECT_NAME, self.policy["registry_name"]}
            or any(value.get(k) != v for k, v in expected.items())
        ):
            raise bad_response()

    def _remember(self, status, code=None):
        with self._state_lock:
            self._health = {"status": status, "error_code": code}
            self._health_until = time.monotonic() + (30 if status == "ok" else 5)

    def health(self):
        if self._auth_failed:
            return {"status": "unavailable", "error_code": "cloud_authentication"}
        with self._state_lock:
            if self._health is not None and time.monotonic() < self._health_until:
                return dict(self._health)
        try:
            result = self._request("GET", "/health", timeout=(1, 1.5))
            self._validate_model(result.get("model"))
            if result.get("status") != "ok":
                raise bad_response()
            self._remember("ok")
        except RemoteInferenceError as error:
            self._remember("unavailable", error.code)
        with self._state_lock:
            return dict(self._health)

    def predict(self, source):
        data = source.read(4 * 1024**2 + 1)
        if len(data) > 4 * 1024**2:
            raise RemoteInferenceError(
                "cloud_rejected_upload", "Choose an image under 4 MiB.", 413
            )
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            extension, mime = {
                "JPEG": ("jpg", "image/jpeg"),
                "PNG": ("png", "image/png"),
                "WEBP": ("webp", "image/webp"),
            }[image.format]
        filename = "upload." + extension
        try:
            result = self._request(
                "POST",
                "/predict",
                headers={"X-API-Key": self._key},
                files={"image": (filename, data, mime)},
            )
            self._validate_model(result.get("model"))
            probability, confidence, threshold = (
                result.get(k) for k in ("ai_probability", "confidence", "threshold")
            )
            if any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or not 0 <= x <= 1
                for x in (probability, confidence, threshold)
            ):
                raise bad_response()
            label = (
                "ai_generated" if probability >= self.policy["threshold"] else "real"
            )
            expected_confidence = (
                probability if label == "ai_generated" else 1 - probability
            )
            if (
                result.get("label") != label
                or threshold != self.policy["threshold"]
                or abs(confidence - expected_confidence) > 1e-6
            ):
                raise bad_response()
            if result.get("image") != {
                "filename": filename,
                "width": width,
                "height": height,
            }:
                raise bad_response()
            self._remember("ok")
            return {
                "label": label,
                "ai_probability": probability,
                "confidence": confidence,
                "threshold": threshold,
            }
        except RemoteInferenceError as error:
            self._remember("unavailable", error.code)
            raise
