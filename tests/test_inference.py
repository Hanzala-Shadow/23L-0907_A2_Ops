"""Check the image and decision contract without using held-out data."""

import io

import numpy as np
import pytest
from PIL import Image

from visionguard.inference import ImagePredictor


def test_inference_applies_locked_threshold_and_shared_uint8_input():
    calls = []

    def model(pixels, training):
        calls.append((pixels, training))
        return np.array([[0.6]])

    source = io.BytesIO()
    Image.new("RGBA", (16, 8), (255, 0, 0, 128)).save(source, format="PNG")
    source.seek(0)
    result = ImagePredictor(model, threshold=0.62, image_size=16).predict(source)
    assert result["label"] == "real"
    assert result["confidence"] == pytest.approx(0.4)
    assert calls[0][0].dtype == np.uint8 and calls[0][0].shape == (1, 16, 16, 3)
    assert calls[0][1] is False
    np.testing.assert_array_equal(calls[0][0][0, 0, 0], [255, 255, 255])


def test_exact_threshold_is_ai_and_invalid_outputs_are_rejected(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    assert (
        ImagePredictor(lambda *args, **kwargs: [[0.6]], 0.6).predict(path)["label"]
        == "ai_generated"
    )
    with pytest.raises(ValueError, match="invalid probability"):
        ImagePredictor(lambda *args, **kwargs: [[float("nan")]], 0.6).predict(path)
