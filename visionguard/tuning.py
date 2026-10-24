"""Controlled MobileNetV2 search space and validation-only analysis."""

import itertools

import keras_tuner as kt
import numpy as np
import tensorflow as tf

from visionguard.transfer import (
    BACKBONE_NAME,
    backbone_fingerprints,
    compile_transfer_model,
)


def search_space(config):
    space = kt.HyperParameters()
    for name in ("learning_rate", "dropout", "unfrozen_layers"):
        values = config[name]
        if len(values) != 2 or len(set(values)) != 2:
            raise ValueError("Use two distinct levels per factor for eight trials")
        space.Choice(name, values, ordered=True)
    if config["max_trials"] != 8:
        raise ValueError("This balanced search requires exactly eight trials")
    return space


def set_unfrozen_layers(model, count):
    """Unfreeze the last N weighted convolutions in block 16, keeping BN fixed."""
    backbone = model.get_layer(BACKBONE_NAME)
    eligible = [
        layer.name
        for layer in backbone.layers
        if layer.name.startswith("block_16_")
        and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
    ]
    if count not in range(len(eligible) + 1):
        raise ValueError("Unfrozen layer count exceeds the final residual block")
    selected = eligible[-count:] if count else []
    backbone.trainable = bool(selected)
    for layer in backbone.layers:
        layer.trainable = layer.name in selected
    return selected


def load_trial_model(checkpoint, values):
    model = tf.keras.models.load_model(checkpoint, compile=False)
    rate = values["dropout"]
    if not 0 <= rate < 1:
        raise ValueError("Dropout must be in [0, 1)")
    model.get_layer("dropout").rate = rate
    selected = set_unfrozen_layers(model, values["unfrozen_layers"])
    compile_transfer_model(model, values["learning_rate"])
    return model, selected


def verify_trial_weights(model, before, selected):
    after = backbone_fingerprints(model)
    changed = [name for name in before if before[name] != after[name]]
    if set(changed) != set(selected):
        raise RuntimeError(
            "Trial changed frozen weights or failed to train selected layers"
        )
    return changed


def assert_complete_search(trials, config):
    expected = set(
        itertools.product(
            config["learning_rate"], config["dropout"], config["unfrozen_layers"]
        )
    )
    actual = {
        tuple(
            trial["hyperparameters"][key]
            for key in ("learning_rate", "dropout", "unfrozen_layers")
        )
        for trial in trials
    }
    if len(trials) != 8 or actual != expected:
        raise RuntimeError("Search did not complete all eight unique combinations")
    if any(trial["test_evaluated"] for trial in trials):
        raise RuntimeError("Tuning must not evaluate the test set")


def factor_effects(trials, config):
    """Marginal effects for the balanced grid; negative means lower loss."""
    effects = {}
    for name in ("learning_rate", "dropout", "unfrozen_layers"):
        levels = sorted(config[name])
        means = {
            str(level): float(
                np.mean(
                    [
                        trial["validation_loss"]
                        for trial in trials
                        if trial["hyperparameters"][name] == level
                    ]
                )
            )
            for level in levels
        }
        effects[name] = {
            "mean_validation_loss_by_level": means,
            "higher_minus_lower": means[str(levels[1])] - means[str(levels[0])],
        }
    return effects
