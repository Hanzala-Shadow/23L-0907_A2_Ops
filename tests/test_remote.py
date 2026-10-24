"""Cloud client and Flask boundaries use synthetic inputs and a fake transport."""

import io
import json
import uuid

import pytest
import requests
from PIL import Image

from visionguard.database import HistoryStore
from visionguard.remote import API_ORIGIN, RemoteInferenceError, RemoteModelService
from visionguard.web import create_app

KEY = "test-api-secret-" * 3


def pixels():
    source = io.BytesIO()
    Image.new("RGB", (24, 12), "red").save(source, format="PNG")
    return source.getvalue()


class Response:
    def __init__(
        self, data=None, status=200, body=None, content_type="application/json"
    ):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.content = json.dumps(data).encode() if body is None else body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class Transport:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def service(tmp_path):
    policy = {
        "registry_name": "23L-0907_A2_Ops",
        "registry_version": "2",
        "run_id": "r" * 32,
        "model_sha256": "a" * 64,
        "threshold": 0.62,
        "preprocessing": {"policy": "rgb-white-letterbox-v1", "image_size": 128},
        "labels": {"real": 0, "ai_generated": 1},
    }
    manifest = {
        "source_model_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "threshold": 0.62,
        "format": "tflite-float32",
    }
    p, m = tmp_path / "policy.json", tmp_path / "manifest.json"
    p.write_text(json.dumps(policy))
    m.write_text(json.dumps(manifest))
    return RemoteModelService(API_ORIGIN, KEY, p, m)


def identity(service):
    p = service.policy
    return {
        "name": p["registry_name"],
        "version": p["registry_version"],
        "threshold": p["threshold"],
        "format": p["format"],
        "sha256": p["model_sha256"],
        "source_sha256": p["source_model_sha256"],
    }


def result(service, probability=0.8):
    is_ai = probability >= service.policy["threshold"]
    return {
        "model": identity(service),
        "threshold": service.policy["threshold"],
        "ai_probability": probability,
        "label": "ai_generated" if is_ai else "real",
        "confidence": probability if is_ai else 1 - probability,
        "image": {"filename": "upload.png", "width": 24, "height": 12},
    }


def use_transport(service, monkeypatch, responses):
    transport = Transport(responses)
    monkeypatch.setattr(service, "_session", lambda: transport)
    return transport


def test_success_bounds_time_and_sends_only_image_with_generic_filename(
    service, monkeypatch
):
    transport = use_transport(
        service,
        monkeypatch,
        [Response(result(service)), Response(result(service, 0.61))],
    )
    assert service.predict(io.BytesIO(pixels()))["label"] == "ai_generated"
    assert service.predict(io.BytesIO(pixels()))["label"] == "real"
    assert service.policy["model_sha256"] == "b" * 64
    assert service.load_count == 0
    method, url, options = transport.calls[0]
    assert method == "POST" and url == API_ORIGIN + "/predict"
    assert options["timeout"] == (3, 45) and options["allow_redirects"] is False
    assert options["stream"] is True
    assert options["headers"] == {"X-API-Key": KEY}
    assert options["files"] == {"image": ("upload.png", pixels(), "image/png")}


@pytest.mark.parametrize(
    "error,code,status",
    [
        (requests.ReadTimeout("private details"), "cloud_timeout", 504),
        (requests.ConnectTimeout(), "cloud_timeout", 504),
        (requests.ConnectionError("private details"), "cloud_unavailable", 503),
        (requests.exceptions.SSLError(), "cloud_unavailable", 503),
    ],
)
def test_transport_failures_do_not_retry(service, monkeypatch, error, code, status):
    transport = use_transport(service, monkeypatch, [error])
    with pytest.raises(RemoteInferenceError) as captured:
        service.predict(io.BytesIO(pixels()))
    assert captured.value.code == code and captured.value.status == status
    assert "private" not in str(captured.value) and len(transport.calls) == 1


@pytest.mark.parametrize(
    "upstream,code,status",
    [
        (401, "cloud_authentication", 503),
        (403, "cloud_authentication", 503),
        (429, "cloud_busy", 503),
        (503, "cloud_busy", 503),
        (500, "cloud_unavailable", 503),
        (502, "cloud_unavailable", 503),
        (302, "cloud_invalid_response", 502),
        (404, "cloud_invalid_response", 502),
        (400, "cloud_rejected_upload", 400),
        (413, "cloud_rejected_upload", 413),
        (415, "cloud_rejected_upload", 415),
    ],
)
def test_http_errors_never_expose_upstream_body(
    service, monkeypatch, upstream, code, status
):
    use_transport(service, monkeypatch, [Response(status=upstream, body=KEY.encode())])
    with pytest.raises(RemoteInferenceError) as captured:
        service.predict(io.BytesIO(pixels()))
    assert captured.value.code == code and captured.value.status == status
    assert KEY not in str(captured.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "wrong"),
        ("version", "3"),
        ("threshold", 0.5),
        ("format", "different"),
        ("sha256", "c" * 64),
        ("source_sha256", "d" * 64),
    ],
)
def test_remote_model_must_match_pinned_policy(service, monkeypatch, field, value):
    response = result(service)
    response["model"][field] = value
    use_transport(service, monkeypatch, [Response(response)])
    with pytest.raises(RemoteInferenceError, match="unexpected"):
        service.predict(io.BytesIO(pixels()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("ai_probability", float("nan")),
        ("ai_probability", float("inf")),
        ("ai_probability", True),
        ("ai_probability", 1.1),
        ("ai_probability", "0.8"),
        ("confidence", 0.1),
        ("threshold", 0.5),
        ("label", "real"),
        ("image", {"filename": "upload.png", "width": 99, "height": 12}),
    ],
)
def test_invalid_results_fail_closed(service, monkeypatch, field, value):
    response = result(service)
    response[field] = value
    use_transport(service, monkeypatch, [Response(response)])
    with pytest.raises(RemoteInferenceError):
        service.predict(io.BytesIO(pixels()))


@pytest.mark.parametrize(
    "response",
    [
        Response(body=b"not json"),
        Response(body=b"[]"),
        Response(body=b"x" * 16385),
        Response(body=b"{}", content_type="text/html"),
    ],
)
def test_invalid_or_oversized_json_is_bounded(service, monkeypatch, response):
    use_transport(service, monkeypatch, [response])
    with pytest.raises(RemoteInferenceError):
        service.predict(io.BytesIO(pixels()))


def test_health_is_cached_and_auth_failure_degrades_it(service, monkeypatch):
    transport = use_transport(
        service,
        monkeypatch,
        [Response({"status": "ok", "model": identity(service)}), Response(status=401)],
    )
    assert service.health()["status"] == service.health()["status"] == "ok"
    assert len(transport.calls) == 1 and transport.calls[0][2]["timeout"] == (1, 1.5)
    with pytest.raises(RemoteInferenceError):
        service.predict(io.BytesIO(pixels()))
    assert service.health() == {
        "status": "unavailable",
        "error_code": "cloud_authentication",
    }
    assert len(transport.calls) == 2


def app_client(service):
    store = HistoryStore("sqlite://", testing=True)
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "x" * 48,
            "DATABASE_URL": "sqlite://",
            "INFERENCE_BACKEND": "remote",
        },
        model_service=service,
        store=store,
    )
    client = app.test_client()
    page = client.get("/")
    assert b"sent securely to the cloud" in page.data
    assert b"4 MiB and 4 megapixels" in page.data
    with client.session_transaction() as session:
        token = session["csrf_token"]
    return app, client, store, token


def upload(client, token):
    return client.post(
        "/predict",
        data={"csrf_token": token, "image": (io.BytesIO(pixels()), "private-name.png")},
        headers={"Accept": "application/json"},
    )


def test_flask_cloud_result_is_saved_with_cloud_provenance(service, monkeypatch):
    use_transport(service, monkeypatch, [Response(result(service))])
    app, client, store, token = app_client(service)
    response = upload(client, token)
    assert response.status_code == 201
    row = store.get(uuid.UUID(response.json["id"]))
    assert row.filename == "private-name.png" and row.model_sha256 == "b" * 64
    assert row.model_run_id == "r" * 32 and row.threshold == 0.62
    assert client.get("/result/" + str(row.id)).status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/health").json["model"]["backend"] == "vercel"
    assert app.extensions["champion"].load_count == 0
    store.engine.dispose()


@pytest.mark.parametrize(
    "error,status,code",
    [
        (requests.ReadTimeout(KEY), 504, "cloud_timeout"),
        (requests.ConnectionError(KEY), 503, "cloud_unavailable"),
        (Response(status=401, body=KEY.encode()), 503, "cloud_authentication"),
        (Response(body=b"invalid"), 502, "cloud_invalid_response"),
    ],
)
def test_flask_cloud_failures_have_no_success_record(
    service, monkeypatch, caplog, error, status, code
):
    use_transport(service, monkeypatch, [error])
    _, client, store, token = app_client(service)
    response = upload(client, token)
    assert response.status_code == status
    row = store.get(uuid.UUID(response.json["result_id"]))
    assert row.status == "error" and row.error_code == code
    assert row.label is row.ai_probability is row.confidence is None
    assert KEY not in response.get_data(as_text=True) and KEY not in caplog.text
    assert client.get("/history").status_code == 200
    assert client.get("/health").status_code == 503
    store.engine.dispose()


@pytest.mark.parametrize(
    "url,key,connect,read",
    [
        ("http://23l-0907-a2-ops-api.vercel.app", KEY, 3, 45),
        ("https://untrusted.vercel.app", KEY, 3, 45),
        (API_ORIGIN + "?key=x", KEY, 3, 45),
        (API_ORIGIN, "", 3, 45),
        (API_ORIGIN, KEY, 0, 45),
        (API_ORIGIN, KEY, 3, float("inf")),
        (API_ORIGIN, KEY, 3, 61),
    ],
)
def test_invalid_client_settings_are_rejected(
    service, tmp_path, url, key, connect, read
):
    with pytest.raises(ValueError):
        RemoteModelService(
            url, key, tmp_path / "absent", tmp_path / "absent", connect, read
        )
