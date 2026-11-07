"""VisionGuard Flask application and PostgreSQL-backed classification routes."""

import hashlib
import io
import math
import os
import secrets
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import RequestEntityTooLarge

from visionguard import PROJECT_NAME
from visionguard.database import HistoryStore
from visionguard.remote import RemoteInferenceError, RemoteModelService
from visionguard.uploads import MemoryRequest, UploadError, safe_name, validate_upload

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def create_app(config=None, model_service=None, store=None):
    load_dotenv(ROOT / ".env", override=False)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.jinja_env.globals["project_name"] = PROJECT_NAME
    app.request_class = MemoryRequest
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY"),
        DATABASE_URL=os.environ.get("DATABASE_URL"),
        MODEL_PATH=os.environ.get("MODEL_PATH", "models/champion/best.keras"),
        MODEL_POLICY_PATH=os.environ.get(
            "MODEL_POLICY_PATH", "models/champion/policy.json"
        ),
        DECISION_THRESHOLD=os.environ.get("DECISION_THRESHOLD"),
        INFERENCE_BACKEND=os.environ.get("INFERENCE_BACKEND", "local"),
        MODEL_API_URL=os.environ.get("MODEL_API_URL", ""),
        MODEL_API_KEY=os.environ.get("MODEL_API_KEY", ""),
        MODEL_API_CONNECT_TIMEOUT=float(
            os.environ.get("MODEL_API_CONNECT_TIMEOUT", "3")
        ),
        MODEL_API_READ_TIMEOUT=float(os.environ.get("MODEL_API_READ_TIMEOUT", "45")),
        MODEL_API_MANIFEST_PATH=os.environ.get(
            "MODEL_API_MANIFEST_PATH", "cloud/model-manifest.json"
        ),
        MAX_UPLOAD_BYTES=int(os.environ.get("MAX_UPLOAD_BYTES", "8388608")),
        MAX_IMAGE_PIXELS=int(os.environ.get("MAX_IMAGE_PIXELS", "16000000")),
        DISK_RESERVE_GIB=float(os.environ.get("DISK_RESERVE_GIB", "2")),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower()
        == "true",
        MAX_FORM_MEMORY_SIZE=256 * 1024,
        MAX_FORM_PARTS=5,
        DEBUG=False,
        HISTORY_PAGE_SIZE=20,
    )
    if config:
        app.config.update(config)
    if app.config["INFERENCE_BACKEND"] not in ("local", "remote"):
        raise ValueError("INFERENCE_BACKEND must be local or remote")
    remote = app.config["INFERENCE_BACKEND"] == "remote"
    if remote:
        app.config["MAX_UPLOAD_BYTES"] = min(
            app.config["MAX_UPLOAD_BYTES"], 4 * 1024**2
        )
        app.config["MAX_IMAGE_PIXELS"] = min(app.config["MAX_IMAGE_PIXELS"], 4_000_000)
    if not app.config["SECRET_KEY"] or len(app.config["SECRET_KEY"]) < 32:
        raise ValueError("Set a persistent FLASK_SECRET_KEY of at least 32 characters")
    if not app.config["DATABASE_URL"]:
        raise ValueError("Set DATABASE_URL for PostgreSQL")
    if (
        not math.isfinite(app.config["DISK_RESERVE_GIB"])
        or app.config["DISK_RESERVE_GIB"] < 2
    ):
        raise ValueError("The minimum disk reserve is 2 GiB")
    if app.config["MAX_UPLOAD_BYTES"] <= 0 or app.config["MAX_IMAGE_PIXELS"] <= 0:
        raise ValueError("Upload limits must be positive")
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"] + 64 * 1024
    app.debug = False
    store = (
        store
        if store is not None
        else HistoryStore(app.config["DATABASE_URL"], app.testing)
    )
    store.initialize()
    if model_service is None:
        if remote:
            model_service = RemoteModelService(
                app.config["MODEL_API_URL"],
                app.config["MODEL_API_KEY"],
                ROOT / app.config["MODEL_POLICY_PATH"],
                ROOT / app.config["MODEL_API_MANIFEST_PATH"],
                app.config["MODEL_API_CONNECT_TIMEOUT"],
                app.config["MODEL_API_READ_TIMEOUT"],
            )
            configured = app.config["DECISION_THRESHOLD"]
            if (
                configured is not None
                and float(configured) != model_service.policy["threshold"]
            ):
                raise ValueError("DECISION_THRESHOLD differs from the saved policy")
        else:
            from visionguard.serving import load_bundle

            model_service = load_bundle(
                ROOT / app.config["MODEL_PATH"],
                ROOT / app.config["MODEL_POLICY_PATH"],
                app.config["DECISION_THRESHOLD"],
            )
    app.extensions.update(history=store, champion=model_service)

    def json_requested():
        return (
            request.accept_mimetypes["application/json"]
            > request.accept_mimetypes["text/html"]
        )

    def error_page(message, status, record=None):
        if json_requested():
            return jsonify(
                error=message, result_id=str(record.id) if record else None
            ), status
        return render_template(
            "error.html", message=message, status=status, record=record
        ), status

    def metadata():
        policy = model_service.policy
        return {
            "model_name": policy["registry_name"],
            "model_version": policy["registry_version"],
            "model_run_id": policy["run_id"],
            "model_sha256": policy["model_sha256"],
            "preprocessing_policy": policy["preprocessing"]["policy"],
            "threshold": policy["threshold"],
        }

    def public_record(row):
        result = row.as_dict()
        result["model_name"] = PROJECT_NAME
        result.pop("model_run_id", None)
        return result

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if not request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def upload():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return render_template(
            "upload.html",
            max_mb=app.config["MAX_UPLOAD_BYTES"] / 1024**2,
            max_mp=app.config["MAX_IMAGE_PIXELS"] / 1_000_000,
            remote=remote,
        )

    @app.post("/predict")
    def predict():
        started = time.monotonic()
        filename, details = None, {}
        try:
            free = shutil.disk_usage(ROOT).free
            if free < (app.config["DISK_RESERVE_GIB"] + 0.125) * GIB:
                return error_page(
                    "Storage is temporarily limited. Please try again later.", 507
                )
            token = request.form.get("csrf_token", "")
            if not token or not secrets.compare_digest(
                token, session.get("csrf_token", "")
            ):
                raise UploadError(
                    "Your form has expired. Open the upload page and try again.",
                    "invalid_form",
                    403,
                )
            uploads = request.files.getlist("image")
            if len(uploads) != 1:
                raise UploadError("Choose exactly one image.", "missing_file")
            filename = safe_name(uploads[0].filename)
            filename, data, width, height = validate_upload(
                uploads[0],
                app.config["MAX_UPLOAD_BYTES"],
                app.config["MAX_IMAGE_PIXELS"],
            )
            details = {
                "image_bytes": len(data),
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "image_width": width,
                "image_height": height,
            }
            result = model_service.predict(io.BytesIO(data))
            if abs(result["threshold"] - model_service.policy["threshold"]) > 1e-12:
                raise RuntimeError(
                    "Inference threshold does not match the loaded policy"
                )
            if (
                shutil.disk_usage(ROOT).free
                < (app.config["DISK_RESERVE_GIB"] + 0.125) * GIB
            ):
                return error_page(
                    "Storage is temporarily limited. Please try again later.", 507
                )
            row = store.add(
                status="success",
                filename=filename,
                **details,
                **metadata(),
                label=result["label"],
                ai_probability=result["ai_probability"],
                confidence=result["confidence"],
                processing_ms=(time.monotonic() - started) * 1000,
            )
        except (UploadError, RequestEntityTooLarge) as error:
            status = error.status if isinstance(error, UploadError) else 413
            code = error.code if isinstance(error, UploadError) else "file_too_large"
            message = (
                str(error)
                if isinstance(error, UploadError)
                else "The upload is too large. Choose a smaller image."
            )
            try:
                row = store.add(
                    status="rejected",
                    filename=filename,
                    **metadata(),
                    error_code=code,
                    error_message=message,
                    processing_ms=(time.monotonic() - started) * 1000,
                )
            except SQLAlchemyError as database_error:
                app.logger.error(
                    "Rejected upload could not be logged (%s)",
                    type(database_error).__name__,
                )
                return error_page(
                    "History is unavailable. Please try again later.", 503
                )
            return error_page(message, status, row)
        except RemoteInferenceError as error:
            app.logger.warning("Cloud prediction failed (%s)", error.code)
            try:
                row = store.add(
                    status="rejected"
                    if error.code == "cloud_rejected_upload"
                    else "error",
                    filename=filename,
                    **details,
                    **metadata(),
                    error_code=error.code,
                    error_message=str(error),
                    processing_ms=(time.monotonic() - started) * 1000,
                )
            except SQLAlchemyError:
                return error_page(
                    "History is unavailable. Please try again later.", 503
                )
            response = app.make_response(error_page(str(error), error.status, row))
            if error.status == 503:
                response.headers["Retry-After"] = "5"
            return response
        except SQLAlchemyError as error:
            app.logger.error("Prediction could not be saved (%s)", type(error).__name__)
            return error_page(
                "Your result could not be saved. Please try again when history is available.",
                503,
            )
        except (RuntimeError, ValueError, OSError) as error:
            app.logger.error("Prediction failed (%s)", type(error).__name__)
            try:
                row = store.add(
                    status="error",
                    filename=filename,
                    **details,
                    **metadata(),
                    error_code="prediction_failed",
                    error_message="The image could not be classified. Please try again.",
                    processing_ms=(time.monotonic() - started) * 1000,
                )
            except SQLAlchemyError:
                row = None
            return error_page(
                "The image could not be classified. Please try again.", 500, row
            )
        if json_requested():
            return jsonify(public_record(row)), 201
        return redirect(url_for("result", result_id=row.id), code=303)

    @app.get("/result/<uuid:result_id>")
    def result(result_id):
        try:
            row = store.get(result_id)
        except SQLAlchemyError as error:
            app.logger.error("History lookup failed (%s)", type(error).__name__)
            return error_page("History is unavailable. Please try again later.", 503)
        if row is None:
            abort(404)
        return (
            jsonify(public_record(row))
            if json_requested()
            else render_template("result.html", record=row)
        )

    @app.get("/history")
    def history():
        try:
            page = int(request.args.get("page", "1"))
            if not 1 <= page <= 1_000_000:
                raise ValueError
        except ValueError:
            return error_page("Choose a valid history page.", 400)
        try:
            rows, total = store.page(page, app.config["HISTORY_PAGE_SIZE"])
        except SQLAlchemyError as error:
            app.logger.error("History query failed (%s)", type(error).__name__)
            return error_page("History is unavailable. Please try again later.", 503)
        if json_requested():
            return jsonify(
                items=[public_record(row) for row in rows], page=page, total=total
            )
        return render_template(
            "history.html",
            records=rows,
            page=page,
            total=total,
            pages=max(1, math.ceil(total / app.config["HISTORY_PAGE_SIZE"])),
        )

    @app.get("/health")
    def health():
        try:
            database_ok = store.healthy()
        except SQLAlchemyError:
            database_ok = False
        free = shutil.disk_usage(ROOT).free / GIB
        cloud = model_service.health() if remote else None
        model_ok = cloud is None or cloud["status"] == "ok"
        ready = (
            database_ok and model_ok and free >= app.config["DISK_RESERVE_GIB"] + 0.125
        )
        return jsonify(
            status="ok" if ready else "degraded",
            database={
                "status": "ok" if database_ok else "unavailable",
                "backend": store.engine.dialect.name,
            },
            model={
                "loaded": not remote,
                "backend": "vercel" if remote else "local",
                "status": "ok" if model_ok else "unavailable",
                "error_code": cloud.get("error_code") if cloud else None,
                "name": PROJECT_NAME,
                "version": model_service.policy["registry_version"],
                "threshold": model_service.policy["threshold"],
            },
            disk={
                "free_gib": round(free, 3),
                "reserve_gib": app.config["DISK_RESERVE_GIB"],
            },
            debug=app.debug,
        ), 200 if ready else 503

    @app.errorhandler(404)
    def missing(error):
        return error_page("This page or saved result could not be found.", 404)

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(error):
        return error_page("The upload is too large. Choose a smaller image.", 413)

    @app.errorhandler(500)
    def unexpected(error):
        return error_page("Something went wrong. Please try again.", 500)

    return app
