# MobileNetV2 transfer learning

Both phases use the existing 681/146 train/validation split, 128 x 128 RGB
letterboxing, batch size 16, seed 42 and training-only flip/rotation/zoom policy.
The test split is reserved for later final evaluation.

The ImageNet MobileNetV2 backbone has no original classifier. Global average
pooling, a 64-unit ReLU head, dropout 0.3 and one sigmoid output adapt it to the
binary task. The complete network has 2,340,033 parameters. MobileNetV2 expects
pixels in [-1, 1], unlike the custom CNN's [0, 1] input normalization. A saved
Rescaling layer performs `pixel / 127.5 - 1`; external callers supply the same
uint8 RGB arrays returned by the shared decoder.

## Training phases

The frozen phase trains 82,049 head parameters with Adam at 0.001 for up to 12
epochs. All pretrained weights and BatchNorm statistics remain frozen.

The fine-tuned phase starts from the frozen phase's checkpoint with the lowest
validation loss. Only `block_16_expand`, `block_16_depthwise` and
`block_16_project` become trainable. Earlier blocks, the final `Conv_1` projection
and every BatchNorm layer stay frozen. There are 551,489 trainable parameters.
The model is recompiled with a fresh Adam optimizer at 0.00001, 100 times below
the head-training rate, for up to 8 epochs. Calling the backbone with
`training=False` preserves its pretrained moving statistics while gradients
still update the selected convolution kernels.

Both phases checkpoint on validation loss, use early stopping with patience 3,
and halve the learning rate after 2 epochs without improvement. Fine-tuning
can worsen validation performance; both phase checkpoints are retained and
compared independently. The comparison uses the lowest validation loss for a
provisional choice. Accuracy, precision, recall, F1 and exact ROC-AUC are also
reported at the unchanged reference threshold of 0.5. Registry promotion and
threshold selection are later steps.

## Logging and storage

```bash
source .venv/bin/activate
python scripts/check_disk.py
PYTHONHASHSEED=42 dvc repro train_mobilenet_finetuned
dvc push -j 2
```

DVC runs frozen-head training before fine-tuning. Each phase has its own model
directory, report directory and MLflow run. MLflow TensorFlow autologging records
fit parameters, optimizer settings, epoch metrics and TensorBoard evidence.
Automatic model and checkpoint uploads are disabled; the script uploads the
explicitly selected best checkpoint afterward. This avoids saving the final
epoch as the chosen model. The resource callback checks disk space and prints
progress without manually logging the same epoch metrics a second time.

The custom CNN baseline used manual logging. Autologging adds framework and
optimizer information, but project-specific data fingerprints, split artifacts,
trainable-layer boundaries, exact validation metrics and model roundtrip checks
still require explicit code. CSV epoch indices and autolog steps are zero-based;
figures and best-epoch numbers are one-based.

Each run includes the source commit, data fingerprint, DVC checksum, seed,
pretrained weights URL and SHA-256, layer table, trainable-layer list, curves,
validation predictions, confusion matrix, ROC curve, split lists, package list,
native checkpoint and MLflow model. Weight fingerprints verify that frozen
layers and BatchNorm statistics do not change. Fine-tuning must update at least
one selected backbone layer. A model downloaded from MLflow must reproduce the
checkpoint's predictions.

Weights are downloaded from Keras's official Google-hosted source into the
ignored runtime cache, about 9 MiB. Images are cached only as resized uint8
arrays in RAM, about 39 MiB for train and validation. DVC versions the two small
model directories; no copied image dataset or epoch checkpoint archive is kept.
The CPU thread limits and 2 GiB disk reserve checks match the first CNN run.
A surrounding process monitor checks the reserve during logging and DVC work.

The CNN DVC dependencies name its source files explicitly so adding separate
transfer-learning modules does not require training the original baseline again.
Its recorded model and metric files are preserved.

References: [Keras transfer learning and BatchNorm guidance](https://keras.io/guides/transfer_learning/),
[MobileNetV2 preprocessing](https://keras.io/api/applications/mobilenet/).
