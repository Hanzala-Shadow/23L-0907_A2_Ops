"""Stateless image classification API for a small CPU instance."""

import hashlib
import hmac
import io
import json
import math
import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from werkzeug.exceptions import HTTPException

from visionguard import PROJECT_NAME, PUBLIC_API_ORIGIN
from visionguard.inference import ImagePredictor
from visionguard.preprocessing import POLICY_VERSION
from visionguard.uploads import MemoryRequest, UploadError, validate_upload


class LiteModel:
    def __init__(self, interpreter, size):
        self.interpreter = interpreter
        interpreter.allocate_tensors()
        inputs, outputs = (
            interpreter.get_input_details(),
            interpreter.get_output_details(),
        )
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or list(inputs[0]["shape"]) != [1, size, size, 3]
            or inputs[0]["dtype"] != np.uint8
            or list(outputs[0]["shape"]) != [1, 1]
            or outputs[0]["dtype"] != np.float32
        ):
            raise ValueError("Cloud model tensor contract is incompatible")
        self.input_index, self.output_index = inputs[0]["index"], outputs[0]["index"]

    def __call__(self, pixels, training=False):
        self.interpreter.set_tensor(self.input_index, pixels)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)


def load_cloud_bundle(directory, manifest_path, interpreter_class=None):
    directory = Path(directory)
    manifest = json.loads(Path(manifest_path).read_text())
    policy_bytes = (directory / "policy.json").read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != manifest["policy_sha256"]:
        raise ValueError("Cloud policy failed its SHA-256 check")
    policy = json.loads(policy_bytes)
    if (
        policy["format"] != "tflite-float32"
        or policy["preprocessing"]["policy"] != POLICY_VERSION
        or policy["labels"] != {"real": 0, "ai_generated": 1}
        or policy["source_model_sha256"] != manifest["source_model_sha256"]
    ):
        raise ValueError("Cloud policy is incompatible")
    threshold = float(policy["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Cloud threshold is invalid")
    checkpoint = directory / "model.tflite"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != policy["model_sha256"] or checksum != manifest["model_sha256"]:
        raise ValueError("Cloud model failed its SHA-256 check")
    if interpreter_class is None:
        from ai_edge_litert.interpreter import Interpreter

        interpreter_class = Interpreter
    size = policy["preprocessing"]["image_size"]
    model = LiteModel(
        interpreter_class(model_path=str(checkpoint), num_threads=1), size
    )
    score = float(model(np.zeros((1, size, size, 3), dtype=np.uint8))[0, 0])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Cloud model startup check failed")
    return ImagePredictor(model, threshold, size), policy


def create_app(config=None, bundle_loader=load_cloud_bundle):
    app = Flask(__name__)
    app.request_class = MemoryRequest
    app.config.update(
        MODEL_API_KEY=os.environ.get("MODEL_API_KEY", ""),
        CLOUD_MODEL_DIR=os.environ.get("CLOUD_MODEL_DIR", "models/cloud-api"),
        CLOUD_MANIFEST_PATH=os.environ.get(
            "CLOUD_MANIFEST_PATH", "cloud/model-manifest.json"
        ),
        MAX_CONTENT_LENGTH=4 * 1024**2 + 64 * 1024,
        MAX_FORM_MEMORY_SIZE=256 * 1024,
        MAX_FORM_PARTS=4,
        MAX_UPLOAD_BYTES=4 * 1024**2,
        MAX_IMAGE_PIXELS=4_000_000,
    )
    app.config.update(config or {})
    secret = app.config["MODEL_API_KEY"]
    if not isinstance(secret, str) or len(secret) < 32:
        raise ValueError("MODEL_API_KEY must contain at least 32 characters")
    predictor, policy = bundle_loader(
        app.config["CLOUD_MODEL_DIR"], app.config["CLOUD_MANIFEST_PATH"]
    )
    gate = threading.BoundedSemaphore(1)
    app.extensions["predictor"] = predictor
    metadata = {
        "name": PROJECT_NAME,
        "version": policy["registry_version"],
        "threshold": policy["threshold"],
        "format": policy["format"],
        "sha256": policy["model_sha256"],
        "source_sha256": policy["source_model_sha256"],
    }

    @app.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return {"error": error.name.lower().replace(" ", "_")}, error.code

    @app.get("/")
    def index():
        return render_template(
            "api.html", project_name=PROJECT_NAME, api_origin=PUBLIC_API_ORIGIN
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": metadata}

    @app.post("/predict")
    def predict():
        token = request.headers.get("X-API-Key", "")
        if len(token) > 256 or not hmac.compare_digest(token.encode(), secret.encode()):
            return {"error": "unauthorized"}, 401
        if not gate.acquire(blocking=False):
            return (
                {"error": "busy", "message": "Retry after the current prediction."},
                503,
                {"Retry-After": "2"},
            )
        try:
            filename, data, width, height = validate_upload(
                request.files.get("image"),
                app.config["MAX_UPLOAD_BYTES"],
                app.config["MAX_IMAGE_PIXELS"],
            )
            result = predictor.predict(io.BytesIO(data))
            return {
                **result,
                "model": metadata,
                "image": {"filename": filename, "width": width, "height": height},
            }
        except UploadError as error:
            return {"error": error.code, "message": str(error)}, error.status
        except HTTPException:
            raise
        except (RuntimeError, ValueError, OSError):
            app.logger.error("Cloud inference failed")
            return {"error": "inference_failed"}, 500
        finally:
            gate.release()

    return app
