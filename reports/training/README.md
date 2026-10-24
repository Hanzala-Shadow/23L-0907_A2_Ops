# First CNN baseline

The CPU run `cnn_baseline_cpu_v1` completed on 24 October 2026. It used source
commit `fef79320155900edecea0c0b0fb4792965d5bab5`, 681 training images and 146
validation images. The 147-image test split was not evaluated.

[MLflow run on DagsHub](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/932b890056eb403aaefa21905070956d)

| Checkpoint validation metric | Value |
| --- | ---: |
| Binary cross-entropy | 0.7012 |
| Accuracy | 44.52% |
| AI-generated precision | 0.0000 |
| AI-generated recall | 0.0000 |
| AI-generated F1 | 0.0000 |
| Exact ROC-AUC | 0.5151 |
| Threshold | 0.5 |
| Selected epoch | 1 |
| Epochs completed | 6 |

Early stopping ended training after five epochs without a new minimum validation
loss. Training and validation took about 149 seconds; preprocessing, remote
logging, artifact verification and DVC bookkeeping brought the command to about
200 seconds. The network has 28,145 parameters.

The selected checkpoint predicted every validation image as real-labeled: 65
correct real predictions and 81 missed AI-generated images. Precision is logged
as zero under the `zero_division=0` convention because no positive predictions
were made. Always predicting the training-majority class, AI-generated, would
score 55.48% validation accuracy. This first CNN is therefore an unsuccessful
baseline and is not suitable for serving.

Training accuracy improved to 67.25% at epoch 6, while the selected checkpoint
has almost chance-level ranking. Batch-normalization running statistics and the
short early-stopping window are possible contributors to the training/inference
gap; this is a hypothesis for a controlled comparison, not an established cause.
Transfer learning and normalization/training-duration comparisons remain future
experiments. No threshold was tuned against the test set.

Per-epoch `val_auc` is Keras's threshold-based approximation for that epoch.
Final `validation_roc_auc` uses scikit-learn on the saved checkpoint's exact
probabilities. The last epoch and selected checkpoint are different models;
compare final checkpoint metrics when selecting a model.

The run includes the original split lists, package versions, configuration,
model summary, per-layer shapes and counts, augmented training batch, curves,
validation predictions, confusion matrix, ROC curve and model artifacts. A
model downloaded from MLflow reproduced local checkpoint predictions. DVC
versions the native checkpoint and preprocessing contract separately from Git.

Disk space stayed above 5.93 GiB during the full command, preserving the 2 GiB
reserve. No preprocessed dataset or sequence of per-epoch checkpoints was kept.

See [training details](../../docs/training.md) and the files in
[cnn_baseline](cnn_baseline/) for the recorded configuration and evidence.
