# Transfer-learning results

Two CPU runs completed on 24 October 2026 using source commit
`b1d5f71123345377024bd6cf533eed72b461f6b9`. Both used the same image policy and
681/146 train/validation membership as the CNN baseline. The 147-image test
split remains unevaluated.

The frozen MobileNetV2 checkpoint has the lowest validation loss, 0.4984, with
78.08% accuracy, precision 0.8101, recall 0.7901, F1 0.8000 and ROC-AUC 0.8418.
Its best epoch was 3; early stopping ended the phase after 6 epochs.

Fine-tuning the last inverted residual block produced loss 0.5104, accuracy
77.40%, precision 0.7927, recall 0.8025, F1 0.7975 and ROC-AUC 0.8427. Its best
epoch was 2; training stopped after 5 epochs. At threshold 0.5 it detected one
additional AI-generated image but incorrectly flagged two additional real-labeled
images. The tiny ROC-AUC increase did not improve loss, accuracy or F1. The
frozen model is the provisional choice; fine-tuning is retained as a separate
experiment rather than replacing the better checkpoint.

The frozen confusion matrix is TN=50, FP=15, FN=17, TP=64. The fine-tuned matrix
is TN=48, FP=17, FN=16, TP=65. These scores describe this validation sample and
are not an estimate from the held-out test set. Threshold selection, broader
hyperparameter comparison and final test evaluation remain outstanding.

Weight fingerprints confirmed that the frozen phase did not change the
pretrained backbone. Fine-tuning changed exactly `block_16_expand`,
`block_16_depthwise` and `block_16_project`. Earlier layers, the final projection
and all BatchNorm parameters and statistics remained unchanged.

Both runs used TensorFlow autologging for epoch metrics, fit settings and
optimizer information. Project metadata, exact checkpoint metrics, split lists,
figures and selected models were logged explicitly. Each native checkpoint was
uploaded as an MLflow model, downloaded and checked against local predictions.
The checkpoints and preprocessing contracts are also versioned with DVC.

Both phases, remote logging, model roundtrips and comparison took about 6.5
minutes in total. Disk space stayed above 5.86 GiB during that command,
preserving the 2 GiB reserve. No additional Python packages were installed;
the official pretrained weights occupied about 9 MiB in the ignored local cache.

See the [comparison table](model_comparison.md), [frozen run](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/30e0c29830cf47509601dd80fbe38b39),
[fine-tuned run](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/ae95c40886344577ade22fa344f0a7a9)
and [training method](../../docs/transfer-learning.md).
