"""Check trial isolation, frozen weights and full search-space coverage."""

import itertools

import keras_tuner as kt
import numpy as np
import pytest
import tensorflow as tf

from visionguard.transfer import (
    BACKBONE_NAME,
    backbone_fingerprints,
    build_transfer_model,
)
from visionguard.tuning import (
    assert_complete_search,
    factor_effects,
    load_trial_model,
    search_space,
    set_unfrozen_layers,
    verify_trial_weights,
)

CONFIG = {
    "learning_rate": [1e-5, 1e-4],
    "dropout": [0.2, 0.5],
    "unfrozen_layers": [1, 3],
    "max_trials": 8,
}


def test_random_search_completes_every_combination(tmp_path):
    records = []

    class Search(kt.RandomSearch):
        def run_trial(self, trial, **kwargs):
            record = {
                "hyperparameters": dict(trial.hyperparameters.values),
                "test_evaluated": False,
            }
            records.append(record)
            return {"val_loss": sum(trial.hyperparameters.values.values())}

    tuner = Search(
        objective="val_loss",
        max_trials=8,
        seed=42,
        hyperparameters=search_space(CONFIG),
        directory=str(tmp_path),
    )
    tuner.search(verbose=0)
    assert_complete_search(records, CONFIG)
    with pytest.raises(RuntimeError, match="eight"):
        assert_complete_search(records[:-1], CONFIG)
    records[0]["test_evaluated"] = True
    with pytest.raises(RuntimeError, match="test set"):
        assert_complete_search(records, CONFIG)


@pytest.mark.parametrize("count", [1, 3])
def test_trial_starts_from_same_weights_and_only_trains_selected_layers(
    tmp_path, count
):
    tf.keras.utils.set_random_seed(42)
    original = build_transfer_model(
        {"image_size": 32},
        {
            "alpha": 0.35,
            "seed": 42,
            "dense_units": 8,
            "dropout": 0.3,
            "augmentation": {"rotation": 0.05, "zoom": 0.1},
        },
        weights=None,
    )
    path = tmp_path / "initial.keras"
    original.save(path)
    pixels = np.random.default_rng(42).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    expected = original(pixels, training=False).numpy()
    model, selected = load_trial_model(
        path, {"learning_rate": 1e-4, "dropout": 0.5, "unfrozen_layers": count}
    )
    np.testing.assert_allclose(
        expected, model(pixels, training=False).numpy(), rtol=1e-6, atol=1e-7
    )
    assert len(selected) == count
    backbone = model.get_layer(BACKBONE_NAME)
    assert not backbone.get_layer("Conv_1").trainable
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
    before = backbone_fingerprints(model)
    model.train_on_batch(pixels, np.array([[0], [1], [0], [1]], dtype=np.float32))
    verify_trial_weights(model, before, selected)
    model.save(tmp_path / "trial.keras")
    restored = tf.keras.models.load_model(tmp_path / "trial.keras")
    assert restored.get_layer("dropout").rate == 0.5
    assert len(restored.get_layer(BACKBONE_NAME).trainable_weights) == count
    with pytest.raises(ValueError):
        set_unfrozen_layers(model, 4)


def test_factor_effects_average_over_other_factors():
    trials = []
    for rate, dropout, layers in itertools.product(
        CONFIG["learning_rate"], CONFIG["dropout"], CONFIG["unfrozen_layers"]
    ):
        trials.append(
            {
                "hyperparameters": {
                    "learning_rate": rate,
                    "dropout": dropout,
                    "unfrozen_layers": layers,
                },
                "validation_loss": 10 * rate + dropout + layers,
            }
        )
    effects = factor_effects(trials, CONFIG)
    assert effects["unfrozen_layers"]["higher_minus_lower"] == pytest.approx(2)
    assert effects["dropout"]["higher_minus_lower"] == pytest.approx(0.3)
    assert effects["learning_rate"]["higher_minus_lower"] == pytest.approx(0.0009)
