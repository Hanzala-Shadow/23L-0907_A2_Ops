"""MobileNetV2 transfer learning with an explicit fine-tuning boundary."""

import hashlib

import tensorflow as tf

from visionguard.cnn import build_augmentation

BACKBONE_NAME = "mobilenetv2_backbone"


def build_transfer_model(preprocessing, config, weights="imagenet"):
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        alpha=config["alpha"],
        input_shape=(preprocessing["image_size"], preprocessing["image_size"], 3),
        name=BACKBONE_NAME,
    )
    backbone.trainable = False
    inputs = tf.keras.Input(
        shape=(preprocessing["image_size"], preprocessing["image_size"], 3),
        dtype="uint8",
        name="rgb_pixels",
    )
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_scaling")(
        inputs
    )
    x = build_augmentation(config["augmentation"], config["seed"])(x)
    # Keep pretrained BatchNorm statistics in inference mode in both phases.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(config["dense_units"], activation="relu", name="head")(x)
    x = tf.keras.layers.Dropout(
        config["dropout"], seed=config["seed"] + 3, name="dropout"
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def unfreeze_last_block(model, prefix="block_16_"):
    if prefix != "block_16_":
        raise ValueError(
            "This experiment only fine-tunes the final inverted residual block"
        )
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith(prefix) and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    names = [layer.name for layer in backbone.layers if layer.trainable_weights]
    if not names:
        raise ValueError("The requested block contains no trainable weights")
    return names


def compile_transfer_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def backbone_fingerprints(model):
    result = {}
    for layer in model.get_layer(BACKBONE_NAME).layers:
        if not layer.weights:
            continue
        digest = hashlib.sha256()
        for weight in layer.weights:
            digest.update(weight.numpy().tobytes())
        result[layer.name] = digest.hexdigest()
    return result


def verify_backbone_changes(model, before, phase, prefix="block_16_"):
    after = backbone_fingerprints(model)
    changed = [name for name in before if before[name] != after[name]]
    if phase == "frozen" and changed:
        raise RuntimeError("Frozen backbone weights changed during head training")
    if phase == "finetuned":
        bn_names = {
            layer.name
            for layer in model.get_layer(BACKBONE_NAME).layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        }
        if any(not name.startswith(prefix) or name in bn_names for name in changed):
            raise RuntimeError(
                "Fine-tuning changed a frozen layer or BatchNorm statistics"
            )
        if not changed:
            raise RuntimeError("Fine-tuning did not update the selected block")
    return changed
