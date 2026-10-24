"""Check image policy, split safety and the CPU model's serving contract."""

import csv
import io
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from visionguard.preprocessing import preprocess_image


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_supported_modes_return_rgb_uint8_without_changing_source(tmp_path, mode):
    path = tmp_path / "image.png"
    Image.new(mode, (20, 10)).save(path)
    original = path.read_bytes()
    pixels = preprocess_image(path, 16)
    assert pixels.shape == (16, 16, 3)
    assert pixels.dtype == np.uint8
    assert path.read_bytes() == original
    np.testing.assert_array_equal(pixels[0, 0], [255, 255, 255])


def test_alpha_composites_onto_white_and_binary_upload_matches_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(path)
    actual = preprocess_image(io.BytesIO(path.read_bytes()), 8)
    np.testing.assert_array_equal(actual[0, 0], [255, 127, 127])
    np.testing.assert_array_equal(actual, preprocess_image(path, 8))


def test_letterbox_preserves_aspect_ratio(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (20, 10), "black").save(path)
    pixels = preprocess_image(path, 20)
    assert np.all(pixels[:5] == 255)
    assert np.all(pixels[5:15] == 0)
    assert np.all(pixels[15:] == 255)


def test_exif_orientation_precedes_letterbox(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 10), "black").save(path, exif=exif)
    pixels = preprocess_image(path, 20)
    assert np.all(pixels[:, :5] == 255)
    assert np.all(pixels[:, 5:15] == 0)
    assert np.all(pixels[:, 15:] == 255)


def test_bad_image_is_rejected():
    with pytest.raises(UnidentifiedImageError):
        preprocess_image(io.BytesIO(b"not an image"))


def write_split(path, relative="image.png", label="0"):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=["relative_path", "class", "label", "sha256"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": relative,
                "class": "real",
                "label": label,
                "sha256": "a" * 64,
            }
        )


def test_split_rejects_path_traversal_and_label_changes(tmp_path):
    from visionguard.pipeline import read_split

    path = tmp_path / "split.csv"
    write_split(path, "../outside.png")
    with pytest.raises(ValueError, match="Unsafe"):
        read_split(path, tmp_path)
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")
    write_split(path, label="1")
    with pytest.raises(ValueError, match="label"):
        read_split(path, tmp_path)


def test_tf_data_matches_shared_upload_preprocessing(tmp_path):
    from visionguard.pipeline import make_dataset, read_split

    path = tmp_path / "split.csv"
    Image.new("RGBA", (10, 20), (0, 255, 0, 128)).save(tmp_path / "image.png")
    write_split(path)
    config = {"image_size": 16, "batch_size": 1, "cache_in_memory": False}
    dataset = make_dataset(read_split(path, tmp_path), tmp_path, config)
    images, labels = next(iter(dataset))
    np.testing.assert_array_equal(
        images[0].numpy(), preprocess_image(tmp_path / "image.png", 16)
    )
    assert labels.numpy().tolist() == [[0.0]]


def test_cnn_training_and_saved_inference_contract(tmp_path):
    import tensorflow as tf

    from visionguard.cnn import build_cnn

    tf.keras.utils.set_random_seed(42)
    config = {
        "seed": 42,
        "filters": [4, 8],
        "dense_units": 8,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "augmentation": {"rotation": 0.05, "zoom": 0.1},
    }
    model = build_cnn({"image_size": 16}, config)
    pixels = np.random.default_rng(42).integers(
        0, 256, size=(4, 16, 16, 3), dtype=np.uint8
    )
    metrics = model.train_on_batch(
        pixels, np.array([[0], [1], [0], [1]], dtype="float32")
    )
    assert all(np.isfinite(value) for value in metrics)
    expected = model(pixels, training=False).numpy()
    np.testing.assert_array_equal(expected, model(pixels, training=False).numpy())
    augmentation = model.get_layer("training_augmentation")
    scaled = pixels.astype("float32") / 255
    np.testing.assert_array_equal(augmentation(scaled, training=False).numpy(), scaled)
    assert not np.array_equal(augmentation(scaled, training=True).numpy(), scaled)
    path = tmp_path / "model.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(
        expected, restored(pixels, training=False).numpy(), rtol=1e-5, atol=1e-6
    )
