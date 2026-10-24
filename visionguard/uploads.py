"""Bounded, in-memory upload validation before model inference."""

import io
import warnings
from pathlib import Path

from flask import Request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


class MemoryRequest(Request):
    def _get_file_stream(
        self, total_content_length, content_type, filename=None, content_length=None
    ):
        return io.BytesIO()


class UploadError(ValueError):
    def __init__(self, message, code, status=400):
        super().__init__(message)
        self.code, self.status = code, status


def safe_name(value):
    return secure_filename(value or "")[:180] or None


def validate_upload(upload, max_bytes, max_pixels):
    if upload is None or not upload.filename:
        raise UploadError("Choose an image before submitting.", "missing_file")
    filename = safe_name(upload.filename)
    suffix = Path(filename or "").suffix.lower()
    if suffix not in FORMATS:
        raise UploadError("Use a JPEG, PNG or WebP image.", "unsupported_type", 415)
    data = upload.stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadError(
            "The image is too large. Choose a smaller file.", "file_too_large", 413
        )
    if not data:
        raise UploadError("The image file is empty.", "empty_file")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if width * height > max_pixels:
                    raise UploadError(
                        "The image has too many pixels. Resize it and try again.",
                        "too_many_pixels",
                        413,
                    )
                if image.format != FORMATS[suffix]:
                    raise UploadError(
                        "The file contents do not match its image extension.",
                        "format_mismatch",
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise UploadError(
                        "Choose a still image rather than an animation.",
                        "animated_image",
                    )
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise UploadError(
            "This file could not be decoded as an image.", "invalid_image"
        ) from error
    return filename, data, width, height
