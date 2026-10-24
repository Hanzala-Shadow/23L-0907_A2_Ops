# CPU CNN baseline

## Image contract and input pipeline

`visionguard/preprocessing.py` is the shared decoder for training and future
Flask uploads. It accepts a path or a binary file object. The policy applies
EXIF orientation, converts grayscale and palette images to RGB, composites
transparency onto white, then resizes with bilinear interpolation and centred
white padding. Images keep their aspect ratio; no crop or geometric stretching
is applied. The result is a 128 x 128 x 3 uint8 array. The model itself rescales
pixels by 1/255, so callers must not normalize them again.

128 pixels is a practical starting point for this two-core CPU. It reduces
convolution cost and memory use, but can discard fine synthesis artifacts.
Image size remains a parameter for later comparisons.

The tf.data pipeline reads the versioned CSV lists, decodes through the shared
Pillow function with `tf.numpy_function`, fixes the tensor shape, caches the
resized uint8 images in RAM, shuffles training rows, batches 16 images, and
prefetches one batch. Decoding uses one worker. The train and validation pixel
caches total about 39 MiB; no resized dataset is written to disk. The decoder is
Python-based so the same Pillow policy can be reused in Flask; the saved Keras
model accepts decoded pixels rather than filenames or encoded uploads.

Only training applies horizontal flips, rotations up to 18 degrees in either
direction, and random zoom up to 10%. Keras random layers are inside the model
and bypass augmentation during validation and inference. `augmented_batch.png`
shows training examples before and after augmentation.

Label 0 means real-labeled; label 1 means AI-generated. The script verifies the
raw SHA-256 inventory and recorded train/validation manifest hashes before
starting. Test images are not passed to the model at this stage.

## Model and training choices

The CNN has 28,145 parameters. Three blocks use 3 x 3 convolutions with 16, 32
and 64 filters, batch normalization, ReLU, and max pooling. Convolutions learn
local patterns, normalization stabilizes activations, and pooling reduces
spatial cost. Biases are omitted before batch normalization. Global average
pooling avoids a large flattened classifier. A 64-unit ReLU head, dropout 0.3,
and one sigmoid output produce the AI-generated probability.

Binary cross-entropy matches the binary target; Adam starts at 0.001. Metrics
include accuracy, ROC-AUC, precision and recall. The mild class imbalance is
left unweighted in this first baseline. The reference threshold is 0.5; threshold
selection and product error tradeoffs are deferred to model comparison.

Seeds are fixed at 42. TensorFlow uses two intra-op threads, one inter-op thread,
deterministic operations, no GPU, no XLA and no oneDNN optimizations. Exact
numerical reproduction on another hardware or library version is not guaranteed;
the package lock and data fingerprints record the environment.

Training allows up to 20 epochs. Checkpointing saves only the lowest validation
loss. Early stopping has patience 5; learning-rate reduction has patience 2 and
halves the rate down to 0.000001. Final validation predictions come from the
saved best checkpoint, not the final epoch. Curves mark the best epoch.

## Run and versioning

```bash
source .venv/bin/activate
python scripts/check_disk.py
PYTHONHASHSEED=42 dvc repro train_cnn
dvc push -j 2
```

Commit training code and parameters before starting. MLflow receives the source
Git commit, DVC raw-data checksum, source fingerprint, seed, hyperparameters,
per-epoch metrics and final validation metrics. This baseline uses a manual
Keras callback; a later transfer-learning experiment can demonstrate autologging
without duplicating metrics in the same run.

The run includes the split lists, package versions, configuration, model summary,
per-layer table, augmented batch, training curves, validation predictions,
confusion matrix, ROC curve, native Keras checkpoint and an MLflow model. A
remote model download must reproduce the saved checkpoint's predictions. The
model is a baseline, not a registry champion or a deployed detector.

DVC tracks `models/cnn_baseline`; Git tracks the code, parameters, pipeline lock
and small reports under `reports/training/cnn_baseline`. Raw execution logs and
local caches stay under ignored `reports/training-runtime`. Reports contain no
credentials or machine-specific environment file contents.

Training checks the 2 GiB free-space reserve on every batch and before artifact
work, with an additional safety margin. Keep the original dataset and the DVC
cache; no preprocessed image directory or epoch-by-epoch checkpoint archive is
created. The test split remains reserved for a later, single final evaluation.

References: [TensorFlow determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism),
[Keras training-only augmentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_rotation/)
and [MLflow TensorFlow tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html).
