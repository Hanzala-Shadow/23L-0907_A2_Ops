# Validation model comparison

All three checkpoints use the same train/validation split and threshold 0.5.
The test split has not been evaluated. Selection is provisional and uses validation loss.

| Model | Loss | Accuracy | Precision | Recall | F1 | ROC-AUC | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [cnn_baseline](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/932b890056eb403aaefa21905070956d) | 0.7012 | 44.52% | 0.0000 | 0.0000 | 0.0000 | 0.5151 | 1 |
| [mobilenetv2_frozen](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/30e0c29830cf47509601dd80fbe38b39) | 0.4984 | 78.08% | 0.8101 | 0.7901 | 0.8000 | 0.8418 | 3 |
| [mobilenetv2_finetuned](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops/experiments/#/experiments/0/runs/ae95c40886344577ade22fa344f0a7a9) | 0.5104 | 77.40% | 0.7927 | 0.8025 | 0.7975 | 0.8427 | 2 |

Lowest validation loss: **mobilenetv2_frozen**.

This is an initial comparison. Hyperparameter trials, threshold selection and final test evaluation remain outstanding. No model has been promoted to a serving champion.

The CNN used manual MLflow logging. MobileNetV2 used TensorFlow autologging for epoch metrics and explicit logging for project metadata, exact checkpoint metrics, plots and selected model artifacts.
