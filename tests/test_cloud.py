import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts.fetch_cloud_model import (
    PUBLIC_BUNDLE_URL,
    allowed_bundle_url,
    unpack_bundle,
)
from visionguard import PROJECT_NAME
from visionguard.cloud import create_app, load_cloud_bundle

KEY = "test-secret-value-that-is-at-least-32-characters"


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "red").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def bundle(tmp_path):
    policy = {
        "format": "tflite-float32",
        "registry_name": "fixture",
        "registry_version": "2",
        "threshold": 0.6,
        "labels": {"real": 0, "ai_generated": 1},
        "preprocessing": {"policy": "rgb-white-letterbox-v1", "image_size": 128},
        "source_model_sha256": "a" * 64,
        "model_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "model.tflite").write_bytes(b"fixture")
    data = json.dumps(policy).encode()
    (directory / "policy.json").write_bytes(data)
    manifest = {
        "source_model_sha256": policy["source_model_sha256"],
        "model_sha256": policy["model_sha256"],
        "policy_sha256": hashlib.sha256(data).hexdigest(),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return directory, path, policy, manifest


class Interpreter:
    loads = 0

    def __init__(self, **kwargs):
        type(self).loads += 1

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"shape": [1, 128, 128, 3], "dtype": np.uint8, "index": 0}]

    def get_output_details(self):
        return [{"shape": [1, 1], "dtype": np.float32, "index": 1}]

    def set_tensor(self, index, data):
        assert data.dtype == np.uint8 and data.shape == (1, 128, 128, 3)

    def invoke(self):
        pass

    def get_tensor(self, index):
        return np.array([[0.7]], dtype=np.float32)


def make_app(bundle):
    directory, path, _, _ = bundle
    return create_app(
        {
            "TESTING": True,
            "MODEL_API_KEY": KEY,
            "CLOUD_MODEL_DIR": directory,
            "CLOUD_MANIFEST_PATH": path,
        },
        bundle_loader=lambda *args: load_cloud_bundle(
            *args, interpreter_class=Interpreter
        ),
    )


def test_api_loads_once_and_uses_saved_policy(bundle):
    Interpreter.loads = 0
    app = make_app(bundle)
    client = app.test_client()
    assert client.get("/health").json["model"]["version"] == "2"
    for _ in range(2):
        response = client.post(
            "/predict",
            headers={"X-API-Key": KEY},
            data={"image": (io.BytesIO(image_bytes()), "image.png")},
        )
        assert response.status_code == 200
        assert response.json["label"] == "ai_generated"
        assert response.json["threshold"] == 0.6
        assert response.headers["Cache-Control"] == "no-store"
    assert Interpreter.loads == 1


def test_public_pages_and_predictions_use_product_identity(bundle):
    client = make_app(bundle).test_client()
    landing = client.get("/")
    assert landing.status_code == 200 and PROJECT_NAME.encode() in landing.data
    assert b"X-API-Key" in landing.data
    health = client.get("/health")
    prediction = client.post(
        "/predict",
        headers={"X-API-Key": KEY},
        data={"image": (io.BytesIO(image_bytes()), "image.png")},
    )
    for response in (health, prediction):
        assert response.json["model"]["name"] == PROJECT_NAME
        assert b"fixture" not in response.data and KEY.encode() not in response.data
    for response in (landing, health, prediction):
        assert b"23L-0907" not in response.data and b"MLOps" not in response.data


@pytest.mark.parametrize(
    "url,allowed",
    [
        (PUBLIC_BUNDLE_URL, True),
        ("https://release-assets.githubusercontent.com/file?token=value", True),
        (PUBLIC_BUNDLE_URL.replace("cloud-v1", "other"), False),
        (PUBLIC_BUNDLE_URL + "?redirect=elsewhere", False),
        ("https://github.com/other/repo/releases/download/x/model.zip", False),
        ("https://release-assets.githubusercontent.com.evil.test/file", False),
        ("https://user@release-assets.githubusercontent.com/file", False),
        ("http://release-assets.githubusercontent.com/file", False),
    ],
)
def test_release_download_origin_is_restricted(url, allowed):
    assert allowed_bundle_url(url) is allowed


@pytest.mark.parametrize("key", [None, "incorrect", "x" * 257, "non-ascii-\u00e9"])
def test_api_requires_key(bundle, key):
    client = make_app(bundle).test_client()
    response = client.post("/predict", headers={"X-API-Key": key} if key else {})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "data,code",
    [
        ({}, 400),
        ({"image": (io.BytesIO(b"bad"), "x.png")}, 400),
        ({"image": (io.BytesIO(image_bytes()), "x.jpg")}, 400),
        ({"image": (io.BytesIO(b"text"), "x.txt")}, 415),
    ],
)
def test_api_rejects_invalid_uploads(bundle, data, code):
    response = (
        make_app(bundle)
        .test_client()
        .post("/predict", headers={"X-API-Key": KEY}, data=data)
    )
    assert response.status_code == code


def test_bundle_checksum_prevents_tampering(bundle):
    directory, path, _, _ = bundle
    (directory / "model.tflite").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_cloud_bundle(directory, path, interpreter_class=Interpreter)


def test_api_accepts_multipart_image_above_parser_buffer(bundle):
    data = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(data).save(output, format="PNG")
    assert len(output.getvalue()) > 256 * 1024
    response = (
        make_app(bundle)
        .test_client()
        .post(
            "/predict",
            headers={"X-API-Key": KEY},
            data={"image": (io.BytesIO(output.getvalue()), "noise.png")},
        )
    )
    assert response.status_code == 200


def test_api_pixel_limit_before_inference(bundle):
    app = make_app(bundle)
    app.config["MAX_IMAGE_PIXELS"] = 100
    response = app.test_client().post(
        "/predict",
        headers={"X-API-Key": KEY},
        data={"image": (io.BytesIO(image_bytes()), "image.png")},
    )
    assert response.status_code == 413 and response.json["error"] == "too_many_pixels"


def test_bundle_rejects_changed_policy(bundle):
    directory, path, _, _ = bundle
    (directory / "policy.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        load_cloud_bundle(directory, path, interpreter_class=Interpreter)


def test_download_unpack_checks_exact_members(bundle, tmp_path):
    directory, _, _, manifest = bundle
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in ["model.tflite", "policy.json"]:
            archive.writestr(name, (directory / name).read_bytes())
    data = output.getvalue()
    manifest["bundle_sha256"] = hashlib.sha256(data).hexdigest()
    unpack_bundle(data, manifest, tmp_path / "restored")
    assert (tmp_path / "restored/model.tflite").read_bytes() == b"fixture"
    with pytest.raises(ValueError, match="already exists"):
        unpack_bundle(data, manifest, tmp_path / "restored")
    with pytest.raises(ValueError, match="checksum"):
        unpack_bundle(data + b"changed", manifest, tmp_path / "changed")


def test_api_rejects_weak_key(bundle):
    with pytest.raises(ValueError, match="32 characters"):
        create_app({"MODEL_API_KEY": "short"})


def test_real_litert_runtime_roundtrip(bundle, tmp_path):
    import tensorflow as tf

    directory, manifest_path, policy, manifest = bundle
    inputs = tf.keras.Input((128, 128, 3), dtype="uint8")
    x = tf.keras.layers.Rescaling(1 / 255)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.layers[-1].set_weights(
        [np.ones((3, 1), dtype=np.float32), np.zeros(1, dtype=np.float32)]
    )
    archive = tf.keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(
        name="serve",
        fn=lambda pixels: model(pixels, training=False),
        input_signature=[tf.TensorSpec((1, 128, 128, 3), tf.uint8)],
    )
    archive.write_out(tmp_path / "saved", verbose=False)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp_path / "saved"))
    data = converter.convert()
    (directory / "model.tflite").write_bytes(data)
    policy["model_sha256"] = hashlib.sha256(data).hexdigest()
    encoded = json.dumps(policy).encode()
    (directory / "policy.json").write_bytes(encoded)
    manifest.update(
        model_sha256=policy["model_sha256"],
        policy_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest))
    app = create_app(
        {
            "TESTING": True,
            "MODEL_API_KEY": KEY,
            "CLOUD_MODEL_DIR": directory,
            "CLOUD_MANIFEST_PATH": manifest_path,
        }
    )
    response = app.test_client().post(
        "/predict",
        headers={"X-API-Key": KEY},
        data={"image": (io.BytesIO(image_bytes()), "red.png")},
    )
    assert response.status_code == 200
    assert response.json["label"] == "ai_generated"
    assert 0.7 < response.json["ai_probability"] < 1
