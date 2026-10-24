# Keras Tuner search

The assignment's built-in dataset example is documented in
[the MNIST toy search](tuning-toy.md). Its four trials are separate from the eight
VisionGuard trials described here.

Run the search with the existing CPU environment and authenticated DagsHub MLflow
configuration:

```bash
source .venv/bin/activate
PYTHONHASHSEED=42 DVC_NO_ANALYTICS=true DAGSHUB_DISABLE_ANALYTICS=1 dvc repro tune_mobilenet
```

Commit code and parameters first. The run records that commit and verifies the
raw dataset and train/validation manifest hashes. Test images never enter a
training, prediction or evaluation dataset. Copying split manifests into MLflow
records membership without running inference on test images.

Keras Tuner `RandomSearch` runs all eight combinations of this finite space,
using seed 42 and validation cross-entropy as its objective:

| Setting | Values |
| --- | --- |
| Initial Adam learning rate | 0.00001, 0.0001 |
| Dropout | 0.2, 0.5 |
| Unfrozen backbone convolution layers | 1, 3 |

Every trial starts from the same previously trained frozen MobileNetV2 checkpoint
with a fresh optimizer. Changing dropout preserves initial inference predictions.
One unfrozen layer means `block_16_project`; three means `block_16_expand`,
`block_16_depthwise` and `block_16_project`. All BatchNorm layers and the final
`Conv_1` projection remain frozen, and backbone execution uses inference mode.
After training, weight hashes check that only the selected convolutions changed.

The shared RGB letterbox decoder caches pixels in RAM. Each trial creates a new
seeded training shuffle; the model's augmentation layers run only during training.
The epoch limit is six, early stopping patience is two and learning-rate reduction
patience is one. These small budgets suit this CPU and do not establish a global
optimum. Validation is reused to select hyperparameters, so the best score is an
optimistic selection estimate. Final test evaluation remains a separate step.

Each trial has its own MLflow run, parameters, epoch metrics, best Keras checkpoint,
validation predictions, training curves, confusion matrix, ROC curve and layer
verification. The selected trial also gets a native MLflow TensorFlow model and a
download/load prediction check. DVC versions the selected model and search reports.
The earlier model comparison remains available; the search summary compares the
best tuning trial with all three earlier runs before choosing a provisional model.
Classification uses the unchanged threshold of 0.5.

The results include mean best validation loss at each factor level, averaging over
the other two factors. The largest absolute difference identifies the strongest
marginal effect within this grid. Interactions, seed variation and the small
validation set limit that interpretation. Training curves show whether later
epochs worsen validation loss; training loss includes augmentation and dropout,
so its difference from validation loss is not a pure generalization estimate.

Free disk is checked before the search, every training batch, at epoch boundaries
and around artifact operations, with a 128 MiB buffer above the 2 GiB reserve.
Decoded images are never written to disk. Checkpoints for individual trials and
resume state stay in ignored `reports/training-runtime/keras_tuner/`; reports and
the selected DVC model are separate outputs. Resume state is scoped to the source
commit. Keep it until the search finishes; Keras Tuner reloads completed trials
after interruption. A completed trial is reused only when its MLflow run finished.

References: [RandomSearch API](https://keras.io/keras_tuner/api/tuners/random/)
and [custom tuner guide](https://keras.io/guides/keras_tuner/custom_tuner/).
