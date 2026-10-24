"""Read split manifests and stream consistently decoded images through tf.data."""

import csv
import hashlib
from pathlib import PurePosixPath

import numpy as np
import tensorflow as tf

from visionguard.preprocessing import preprocess_image

LABELS = {"real": 0, "ai_generated": 1}


def read_split(csv_path, raw_dir):
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The split is empty")
    paths, hashes = set(), set()
    for row in rows:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Unsafe image path in split")
        path = raw_dir.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(raw_dir.resolve()):
            raise ValueError("Image path leaves the raw data directory")
        if not path.is_file():
            raise ValueError(f"Missing source image: {relative}")
        if row["class"] not in LABELS or int(row["label"]) != LABELS[row["class"]]:
            raise ValueError("Unexpected class or numeric label")
        if row["relative_path"] in paths or row["sha256"] in hashes:
            raise ValueError("Duplicate path or hash within a split")
        paths.add(row["relative_path"])
        hashes.add(row["sha256"])
    return rows


def verify_split_hash(csv_path, expected):
    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Split manifest differs from its recorded version: {csv_path.name}"
        )


def make_dataset(rows, raw_dir, config, training=False, seed=42):
    paths = [str(raw_dir / row["relative_path"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.float32)
    image_size = config["image_size"]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        def load(value):
            return preprocess_image(np.asarray(value).item().decode(), image_size)

        pixels = tf.numpy_function(load, [path], tf.uint8)
        pixels.set_shape((image_size, image_size, 3))
        return pixels, tf.reshape(label, (1,))

    dataset = dataset.map(decode, num_parallel_calls=1, deterministic=True)
    if config["cache_in_memory"]:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(config["batch_size"], drop_remainder=False).prefetch(1)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options)
