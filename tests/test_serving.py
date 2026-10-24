"""Verify the saved model contract and one model load across HTTP requests."""

import hashlib
import io
import json

import pytest
import tensorflow as tf
from PIL import Image

from visionguard.serving import load_bundle
from visionguard.web import create_app


@pytest.fixture
def bundle(tmp_path):
    inputs = tf.keras.Input(shape=(16, 16, 3), dtype="uint8")
    pixels = tf.keras.layers.Rescaling(1 / 255)(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(pixels)
    model = tf.keras.Model(
        inputs, tf.keras.layers.Dense(1, activation="sigmoid")(features)
    )
    checkpoint = tmp_path / "best.keras"
    model.save(checkpoint)
    policy = {
        "registry_name": "23L-0907_A2_Ops",
        "registry_version": "2",
        "run_id": "r" * 32,
        "threshold": 0.62,
        "preprocessing": {"policy": "rgb-white-letterbox-v1", "image_size": 16},
        "labels": {"real": 0, "ai_generated": 1},
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return checkpoint, path


def test_model_loads_once_at_app_startup(bundle, monkeypatch):
    checkpoint, policy = bundle
    original = tf.keras.models.load_model
    loads = []

    def tracked(*args, **kwargs):
        loads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tf.keras.models, "load_model", tracked)
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "s" * 48,
            "DATABASE_URL": "sqlite://",
            "MODEL_PATH": str(checkpoint),
            "MODEL_POLICY_PATH": str(policy),
            "DECISION_THRESHOLD": "0.62",
        }
    )
    client = app.test_client()
    client.get("/")
    with client.session_transaction() as session:
        token = session["csrf_token"]
    for _ in range(2):
        source = io.BytesIO()
        Image.new("RGB", (16, 8), "blue").save(source, format="PNG")
        source.seek(0)
        response = client.post(
            "/predict",
            data={"csrf_token": token, "image": (source, "test.png")},
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 201 and response.json["threshold"] == 0.62
        assert client.get("/health").status_code == 200
    assert loads == [1]
    app.extensions["history"].engine.dispose()


@pytest.mark.parametrize("threshold", [0.5, float("nan")])
def test_threshold_override_cannot_change_locked_policy(bundle, threshold):
    with pytest.raises(ValueError, match="DECISION_THRESHOLD"):
        load_bundle(*bundle, configured_threshold=threshold)


def test_changed_checkpoint_is_rejected_before_loading(bundle, monkeypatch):
    checkpoint, policy = bundle
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(
        tf.keras.models,
        "load_model",
        lambda *args, **kwargs: pytest.fail("must not load changed weights"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(checkpoint, policy)
