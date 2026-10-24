# Keras Tuner results

Eight VisionGuard trials completed on CPU using Keras Tuner RandomSearch. Each
started from the same frozen MobileNetV2 checkpoint with a fresh Adam optimizer,
seed 42, shared 128-pixel RGB preprocessing and a newly seeded training shuffle.
The fixed split contains 681 training and 146 validation images. No model used
the test images for training, prediction or evaluation. Classification uses 0.5.

| Trial | Learning rate | Dropout | Unfrozen convolutions | Validation loss | Accuracy | Best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| [0](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/6b72ce8920c84fa6ad9b326cb2da5b7d) | 0.0001 | 0.2 | 1 | 0.513839 | 76.71% | 4 |
| [1](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/3a67561976894ca7ab51c928abbd659a) | 1e-05 | 0.5 | 1 | 0.503321 | 76.71% | 2 |
| [2](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/829d1e5888b14aa6845067c59e8f86fd) | 1e-05 | 0.5 | 3 | 0.504160 | 76.71% | 2 |
| [3](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/36b46870e7ab4eae89648f35882a49d6) | 0.0001 | 0.2 | 3 | 0.544091 | 76.03% | 4 |
| [4](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/d1b5ff9d31ac4d0aa1c8a4278b184029) | 1e-05 | 0.2 | 3 | 0.512578 | 74.66% | 2 |
| [5](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/4b4fe730d948462ca13d361a5f865d2c) | 1e-05 | 0.2 | 1 | 0.511039 | 75.34% | 2 |
| [6](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/94a24bdfe5344fce8ac0ae18ed2e1809) | 0.0001 | 0.5 | 1 | 0.496035 | 76.03% | 4 |
| [7](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/c61c4503fd4945818677efbf877d1581) | 0.0001 | 0.5 | 3 | 0.485135 | 78.77% | 3 |

The selected trial is 7: learning rate 0.0001, dropout 0.5 and three unfrozen
convolutions in the final inverted residual block. BatchNorm and all other
backbone weights stayed unchanged. Its checkpoint comes from epoch 3 of 5.
The DVC model is `models/mobilenetv2_tuned/best.keras`; its native MLflow model is
[`runs:/c61c4503fd4945818677efbf877d1581/model`](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/c61c4503fd4945818677efbf877d1581). Downloading and
loading that model reproduced local predictions.

| Model | Validation loss | Accuracy | F1 | ROC AUC |
| --- | --- | --- | --- | --- |
| cnn_baseline | 0.701214 | 44.52% | 0.0000 | 0.5151 |
| mobilenetv2_frozen | 0.498448 | 78.08% | 0.8000 | 0.8418 |
| mobilenetv2_finetuned | 0.510448 | 77.40% | 0.7975 | 0.8427 |
| mobilenetv2_tuned | 0.485135 | 78.77% | 0.8098 | 0.8663 |

The tuned model is the provisional choice by the predefined validation-loss
objective. Its confusion matrix is TN 49, FP 16, FN 15 and TP 66. Compared with
the frozen model, it detects two more AI images but falsely labels one more
real image as AI. The net accuracy gain is one image out of 146, or 0.68 percentage
points; the validation-loss improvement is about 0.0133. This small change needs
independent evaluation before any broader claim.

## Which setting mattered most

| Factor | Mean loss at lower level | Mean loss at higher level | Higher minus lower |
| --- | --- | --- | --- |
| learning_rate | 0.507775 | 0.509775 | +0.002001 |
| dropout | 0.520387 | 0.497163 | -0.023224 |
| unfrozen_layers | 0.506059 | 0.511491 | +0.005432 |

Dropout has the largest marginal effect in this balanced grid: 0.5 reduces
average best validation loss by 0.0232 compared with 0.2. Stronger regularization
appears useful for this small training set. These are averages over the other
two factors, not proof of a universal effect. Interactions matter: the winning
combination uses the higher learning rate and all three selected convolutions
even though their marginal averages are slightly worse. Only one seed and a
small search space were tested.

## Overfitting and selection limits

In the selected trial, training loss fell from 0.3081 at epoch 3 to 0.2401 at
epoch 5 while validation loss rose from 0.4851 to 0.5063. Early stopping restored
the epoch-3 checkpoint. Augmentation and dropout apply only during training, so
the training/validation loss difference is not a pure generalization estimate.

Trying eight settings and choosing the minimum reuses the same 146 validation
cases. Selection can favor random noise and overfit validation even when each
fit uses early stopping. These are selection results; final threshold choice
and one independent test evaluation remain later work.

## Reproduction and tracking

Run `dvc repro tune_mobilenet` in the pinned environment with DagsHub credentials
in the ignored local environment. DVC records the selected model and all trial
reports. The summary and CSV are in `reports/training/keras_tuner/`. Every trial
has a separate MLflow run with parameters, epoch metrics, checkpoint, validation
predictions, curves and checks of frozen backbone weights. Individual local
trial checkpoints and logs stay in the ignored runtime directory.

## Built-in dataset demonstration

The separate MNIST search used only 6,000 training and 1,000 validation images
from the built-in training partition. Four three-epoch trials tuned hidden units
and learning rate. It did not read the archive's test entries. These digit
classification scores are separate from VisionGuard.

| Trial | Learning rate | Units | Validation loss | Accuracy | Best epoch |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.01 | 32 | 0.314494 | 90.00% | 3 |
| 1 | 0.001 | 32 | 0.441613 | 88.40% | 3 |
| 2 | 0.01 | 64 | 0.246962 | 91.50% | 2 |
| 3 | 0.001 | 64 | 0.376497 | 89.60% | 3 |

The best toy settings were learning rate 0.01 and 64 units, selected at epoch 2.
All four toy runs are logged to DagsHub under `phase=keras_tuner_toy`.

The independent disk monitors observed at least 5.69 GiB free during the real
search and 5.67 GiB during the toy search, preserving the 2 GiB reserve.
