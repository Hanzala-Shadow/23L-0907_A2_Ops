"""Load the local champion bundle once for a web-server process."""

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from visionguard.preprocessing import POLICY_VERSION


class InferenceError(RuntimeError):
    pass


def configure_cpu():
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }.items():
        os.environ.setdefault(name, value)


@dataclass
class ModelService:
    predictor: object
    policy: dict
    lock: threading.Lock = field(default_factory=threading.Lock)
    load_count: int = 1

    def predict(self, source):
        with self.lock:
            try:
                return self.predictor.predict(source)
            except Exception as error:
                raise InferenceError("The image could not be classified") from error


def load_bundle(model_path, policy_path, configured_threshold=None):
    configure_cpu()
    import numpy as np
    import tensorflow as tf

    from visionguard.inference import ImagePredictor

    model_path, policy_path = Path(model_path), Path(policy_path)
    policy = json.loads(policy_path.read_text())
    if policy["preprocessing"]["policy"] != POLICY_VERSION:
        raise ValueError("The champion preprocessing policy is incompatible")
    if policy["labels"] != {"real": 0, "ai_generated": 1}:
        raise ValueError("The champion label mapping is incompatible")
    threshold = float(policy["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The champion threshold is invalid")
    if configured_threshold is not None:
        configured = float(configured_threshold)
        if not math.isfinite(configured) or abs(configured - threshold) > 1e-12:
            raise ValueError(
                "DECISION_THRESHOLD differs from the saved champion policy"
            )
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != policy["model_sha256"]:
        raise ValueError("The champion checkpoint failed its SHA-256 check")
    model = tf.keras.models.load_model(model_path, compile=False)
    size = policy["preprocessing"]["image_size"]
    if (
        model.input_shape != (None, size, size, 3)
        or str(model.inputs[0].dtype) != "uint8"
    ):
        raise ValueError("The champion input contract is incompatible")
    predictor = ImagePredictor(model, threshold, size)
    sample = np.zeros((1, size, size, 3), dtype=np.uint8)
    probability = float(model(sample, training=False).numpy().reshape(-1)[0])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The champion startup check returned an invalid score")
    return ModelService(predictor, policy)
