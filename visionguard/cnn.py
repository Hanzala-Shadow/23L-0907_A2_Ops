"""Small convolutional baseline with training-only random augmentation."""

import tensorflow as tf


def build_augmentation(config, seed):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed, name="horizontal_flip"),
            tf.keras.layers.RandomRotation(
                config["rotation"], seed=seed + 1, fill_mode="reflect", name="rotation"
            ),
            tf.keras.layers.RandomZoom(
                config["zoom"], seed=seed + 2, fill_mode="reflect", name="zoom"
            ),
        ],
        name="training_augmentation",
    )


def build_cnn(preprocessing, config):
    seed = config["seed"]
    inputs = tf.keras.Input(
        shape=(preprocessing["image_size"], preprocessing["image_size"], 3),
        dtype="uint8",
        name="rgb_pixels",
    )
    x = tf.keras.layers.Rescaling(1 / 255, name="rescale")(inputs)
    x = build_augmentation(config["augmentation"], seed)(x)
    for index, filters in enumerate(config["filters"], start=1):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"conv_{index}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(config["dense_units"], activation="relu", name="head")(x)
    x = tf.keras.layers.Dropout(config["dropout"], seed=seed + 3, name="dropout")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = tf.keras.Model(inputs, output, name="cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
