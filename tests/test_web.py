"""Exercise the HTTP flow with synthetic images and isolated history storage."""

import io
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from flask import request
from PIL import Image
from sqlalchemy.exc import OperationalError

from visionguard import PROJECT_NAME
from visionguard.database import HistoryStore
from visionguard.serving import ModelService
from visionguard.web import create_app


@pytest.fixture
def app():
    class Predictor:
        calls = 0

        def predict(self, source):
            self.calls += 1
            return {
                "label": "ai_generated",
                "ai_probability": 0.8,
                "confidence": 0.8,
                "threshold": 0.62,
            }

    policy = {
        "registry_name": "23L-0907_A2_Ops",
        "registry_version": "2",
        "run_id": "r" * 32,
        "model_sha256": "a" * 64,
        "threshold": 0.62,
        "preprocessing": {"policy": "rgb-white-letterbox-v1", "image_size": 128},
    }
    store = HistoryStore("sqlite://", testing=True)
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "x" * 48,
            "DATABASE_URL": "sqlite://",
            "INFERENCE_BACKEND": "local",
            "MAX_UPLOAD_BYTES": 1024 * 1024,
            "MAX_IMAGE_PIXELS": 1_000_000,
            "HISTORY_PAGE_SIZE": 2,
        },
        model_service=ModelService(Predictor(), policy),
        store=store,
    )
    yield application
    store.engine.dispose()


@pytest.fixture
def client(app):
    client = app.test_client()
    client.get("/")
    with client.session_transaction() as session:
        client.csrf = session["csrf_token"]
    return client


def image_file(format="PNG", size=(24, 12), mode="RGB"):
    source = io.BytesIO()
    Image.new(mode, size, "green").save(source, format=format)
    source.seek(0)
    return source


def upload(client, source=None, filename="sample.png", json=True):
    return client.post(
        "/predict",
        data={
            "csrf_token": client.csrf,
            "image": (source if source is not None else image_file(), filename),
        },
        headers={"Accept": "application/json" if json else "text/html"},
    )


def test_success_result_history_and_one_model_instance(app, client):
    service = app.extensions["champion"]
    first = upload(client, filename="../../sample.png")
    second = upload(client)
    assert first.status_code == second.status_code == 201
    assert first.json["filename"] == "sample.png"
    assert first.json["threshold"] == 0.62 and first.json["model_version"] == "2"
    assert first.json["label"] == "ai_generated" and first.json["confidence"] == 0.8
    assert service.load_count == 1 and service.predictor.calls == 2
    result = client.get("/result/" + first.json["id"])
    assert result.status_code == 200 and b"Likely AI-generated" in result.data
    assert b"cannot verify" in result.data and b"not a guarantee" in result.data
    history = client.get("/history", headers={"Accept": "application/json"})
    assert history.json["total"] == 2 and len(history.json["items"]) == 2
    stored = app.extensions["history"].get(uuid.UUID(first.json["id"]))
    assert stored.image_sha256 and stored.image_bytes > 0 and stored.image_width == 24
    assert stored.created_at is not None and stored.processing_ms >= 0
    assert stored.model_name == "23L-0907_A2_Ops" and stored.model_run_id == "r" * 32
    assert first.json["model_name"] == PROJECT_NAME
    assert "model_run_id" not in first.json
    assert history.json["items"][0]["model_name"] == PROJECT_NAME
    for page in (
        client.get("/"),
        result,
        client.get("/history"),
        client.get("/health"),
    ):
        assert PROJECT_NAME.encode() in page.data
        assert b"23L-0907" not in page.data and b"MLOps" not in page.data


@pytest.mark.parametrize(
    "format,filename,mode",
    [("JPEG", "a.jpg", "RGB"), ("PNG", "a.png", "RGBA"), ("WEBP", "a.webp", "RGB")],
)
def test_supported_image_formats(client, format, filename, mode):
    assert (
        upload(client, image_file(format=format, mode=mode), filename).status_code
        == 201
    )


@pytest.mark.parametrize(
    "filename,data,status,code",
    [
        ("wrong.txt", b"text", 415, "unsupported_type"),
        ("fake.png", b"not an image", 400, "invalid_image"),
        ("empty.png", b"", 400, "empty_file"),
        ("big.png", b"a" * (1024 * 1024 + 1), 413, "file_too_large"),
    ],
)
def test_rejected_uploads_are_logged_without_inference(
    app, client, filename, data, status, code
):
    response = upload(client, io.BytesIO(data), filename)
    assert response.status_code == status and response.json["result_id"]
    row = app.extensions["history"].get(uuid.UUID(response.json["result_id"]))
    assert row.status == "rejected" and row.error_code == code
    assert row.label is row.ai_probability is row.confidence is None
    assert app.extensions["champion"].predictor.calls == 0


def test_friendly_error_and_body_limit(app, client):
    response = upload(client, io.BytesIO(b"broken"), "broken.jpg", json=False)
    assert response.status_code == 400 and b"could not be decoded" in response.data
    oversized = upload(client, io.BytesIO(b"a" * (2 * 1024 * 1024)), "large.png")
    assert oversized.status_code == 413
    assert (
        app.extensions["history"].get(uuid.UUID(oversized.json["result_id"])).error_code
        == "file_too_large"
    )


def test_missing_image_csrf_format_mismatch_and_pixel_limit(app, client):
    no_image = client.post("/predict", data={"csrf_token": client.csrf})
    assert no_image.status_code == 400
    no_csrf = client.post("/predict", data={"image": (image_file(), "a.png")})
    assert no_csrf.status_code == 403
    mismatch = upload(client, image_file(), "wrong.jpg")
    assert mismatch.status_code == 400
    app.config["MAX_IMAGE_PIXELS"] = 10
    assert upload(client).status_code == 413
    assert app.extensions["champion"].predictor.calls == 0


def test_multipart_files_stay_in_memory(app):
    generator = np.random.default_rng(42)
    image = Image.fromarray(generator.integers(0, 256, (450, 450, 3), dtype=np.uint8))
    source = io.BytesIO()
    image.save(source, format="PNG")
    assert source.tell() > 500_000
    source.seek(0)
    with app.test_request_context(
        "/predict", method="POST", data={"image": (source, "large.png")}
    ):
        assert isinstance(request.files["image"].stream, io.BytesIO)


def test_pagination_missing_results_and_html_escaping(app, client):
    for filename in ("one.png", "two.png", "<script>alert.png"):
        assert upload(client, filename=filename).status_code == 201
    first = client.get("/history", headers={"Accept": "application/json"}).json
    second = client.get("/history?page=2", headers={"Accept": "application/json"}).json
    assert (
        first["total"] == 3 and len(first["items"]) == 2 and len(second["items"]) == 1
    )
    page = client.get("/history").data
    assert b"<script>alert" not in page
    assert client.get("/history?page=0").status_code == 400
    assert client.get("/history?page=no").status_code == 400
    assert client.get("/result/" + str(uuid.uuid4())).status_code == 404


def test_browser_post_redirect_get_and_security_headers(client):
    response = upload(client, json=False)
    assert response.status_code == 303 and "/result/" in response.headers["Location"]
    result = client.get(response.headers["Location"])
    assert result.status_code == 200 and result.headers["Cache-Control"] == "no-store"
    assert result.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in result.headers["Content-Security-Policy"]


def test_database_outage_fails_closed_and_degrades_health(app, client, monkeypatch):
    store = app.extensions["history"]

    def broken(*args, **kwargs):
        raise OperationalError("statement", {}, Exception("private database details"))

    monkeypatch.setattr(store, "add", broken)
    monkeypatch.setattr(store, "healthy", broken)
    assert upload(client).status_code == 503
    response = client.get("/health")
    assert (
        response.status_code == 503
        and response.json["database"]["status"] == "unavailable"
    )
    assert b"private database details" not in response.data


def test_prediction_failure_is_logged(app, client, monkeypatch):
    def broken(source):
        raise RuntimeError("Internal model details")

    monkeypatch.setattr(app.extensions["champion"].predictor, "predict", broken)
    response = upload(client)
    assert response.status_code == 500 and "Internal" not in response.json["error"]
    row = app.extensions["history"].get(uuid.UUID(response.json["result_id"]))
    assert row.status == "error" and row.error_code == "prediction_failed"


def test_disk_reserve_blocks_upload_without_model_call(app, client, monkeypatch):
    monkeypatch.setattr(
        "visionguard.web.shutil.disk_usage",
        lambda path: SimpleNamespace(free=2 * 1024**3),
    )
    assert upload(client).status_code == 507
    assert app.extensions["champion"].predictor.calls == 0
    assert client.get("/health").status_code == 503


def test_health_reports_loaded_policy_without_reloading(app, client):
    for _ in range(2):
        response = client.get("/health")
        assert response.status_code == 200 and response.json["debug"] is False
        assert response.json["model"]["threshold"] == 0.62
    assert app.extensions["champion"].load_count == 1
    assert app.extensions["champion"].predictor.calls == 0


def test_production_database_requires_postgresql():
    with pytest.raises(ValueError, match="postgresql"):
        HistoryStore("sqlite://")
