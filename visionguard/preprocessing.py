"""Deterministic image conversion shared by training and future inference."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

POLICY_VERSION = "rgb-white-letterbox-v1"


def preprocess_image(source, image_size=128):
    """Return square RGB uint8 pixels without stretching or saving an image.

    ``source`` can be a path or a binary file object such as an uploaded image.
    EXIF orientation is applied before compositing and resizing. Transparency
    and padding use white; the saved CNN performs the 1/255 rescaling itself.
    """
    if not isinstance(image_size, int) or image_size < 8:
        raise ValueError("image_size must be an integer of at least 8 pixels")
    if isinstance(source, (str, Path)):
        source = Path(source)
    with Image.open(source) as original:
        oriented = ImageOps.exif_transpose(original)
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        square = ImageOps.pad(
            rgb,
            (image_size, image_size),
            method=Image.Resampling.BILINEAR,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )
        return np.asarray(square, dtype=np.uint8).copy()
