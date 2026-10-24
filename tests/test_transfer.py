"""Test MobileNet preprocessing, frozen weights and fine-tuning boundaries."""

import numpy as np
import pytest
import tensorflow as tf

from visionguard.transfer import (
    BACKBONE_NAME,
    backbone_fingerprints,
    build_transfer_model,
    compile_transfer_model,
    unfreeze_last_block,
    verify_backbone_changes,
)


@pytest.fixture
def transfer():
    tf.keras.utils.set_random_seed(42)
    config = {
        "seed": 42,
        "alpha": 0.35,
        "dense_units": 8,
        "dropout": 0.3,
        "augmentation": {"rotation": 0.05, "zoom": 0.1},
    }
    model = build_transfer_model({"image_size": 32}, config, weights=None)
    compile_transfer_model(model, 0.001)
    return model


def test_scaling_matches_mobilenet_preprocess_input(transfer):
    pixels = np.arange(256, dtype="float32").reshape(1, 16, 16, 1)
    expected = tf.keras.applications.mobilenet_v2.preprocess_input(pixels.copy())
    np.testing.assert_allclose(
        transfer.get_layer("mobilenet_scaling")(pixels).numpy(), expected, atol=2e-7
    )


def test_frozen_backbone_and_save_load(transfer, tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    before = backbone_fingerprints(transfer)
    transfer.train_on_batch(pixels, np.array([[0], [1]], dtype="float32"))
    assert verify_backbone_changes(transfer, before, "frozen") == []
    assert not transfer.get_layer(BACKBONE_NAME).trainable_weights
    expected = transfer(pixels, training=False).numpy()
    path = tmp_path / "transfer.keras"
    transfer.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(
        restored(pixels, training=False).numpy(), expected, rtol=1e-5, atol=1e-6
    )


def test_only_last_block_can_be_unfrozen_and_batchnorm_stays_frozen(transfer):
    names = unfreeze_last_block(transfer)
    backbone = transfer.get_layer(BACKBONE_NAME)
    assert names and all(name.startswith("block_16_") for name in names)
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
    assert not backbone.get_layer("Conv_1").trainable
    compile_transfer_model(transfer, 0.00001)
    before = backbone_fingerprints(transfer)
    pixels = np.random.default_rng(42).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    transfer.train_on_batch(pixels, np.array([[0], [1]], dtype="float32"))
    after = backbone_fingerprints(transfer)
    assert all(
        before[name] == after[name]
        for name in before
        if not name.startswith("block_16_")
    )
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert before[layer.name] == after[layer.name]
    with pytest.raises(ValueError):
        unfreeze_last_block(transfer, "block_15_")


def test_comparison_rejects_different_validation_images(tmp_path):
    import csv
    import json

    from visionguard.transfer_reporting import comparison_records

    for name in ("cnn_baseline", "mobilenetv2_frozen", "mobilenetv2_finetuned"):
        folder = tmp_path / "reports/training" / name
        folder.mkdir(parents=True)
        (folder / "metrics.json").write_text(json.dumps({"validation_loss": 0.5}))
        (folder / "run.json").write_text(
            json.dumps(
                {
                    "test_evaluated": False,
                    "data_fingerprint": "same",
                    "split_counts": {"train": 2, "validation": 2},
                    "run_id": name,
                    "run_url": "https://example.com/run",
                }
            )
        )
        with (folder / "validation_predictions.csv").open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["relative_path", "label"])
            writer.writerow(
                ["other.png" if name == "mobilenetv2_finetuned" else "a.png", 0]
            )
            writer.writerow(["b.png", 1])
    with pytest.raises(ValueError, match="validation images"):
        comparison_records(tmp_path)
