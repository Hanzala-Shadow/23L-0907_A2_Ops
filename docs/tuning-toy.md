# Built-in dataset example

Task 2.7 also asks for a tiny network tuned on a built-in dataset. This example
uses MNIST, with 6,000 training images and 1,000 validation images selected from
its original training partition using NumPy seed 42. It reads only the archive's
`x_train` and `y_train` entries. Neither MNIST's test entries nor VisionGuard's
test images are used.

```bash
source .venv/bin/activate
PYTHONHASHSEED=42 DVC_NO_ANALYTICS=true DAGSHUB_DISABLE_ANALYTICS=1 dvc repro tune_toy
```

Keras Tuner RandomSearch runs four combinations: Adam learning rates 0.001 and
0.01, and hidden layer sizes 32 and 64. Each network rescales pixels, flattens the
image, uses one ReLU hidden layer and predicts ten digit classes. The budget is
three epochs per trial, with the best validation-loss checkpoint selected.

Each trial is logged to DagsHub under `phase=keras_tuner_toy`. Those runs are
separate from the eight VisionGuard trials and their scores are not comparable.
Results, best settings and the source archive SHA-256 are saved under
`reports/training/toy_tuner/`; DVC tracks the selected checkpoint. The approximately
11 MiB archive and individual trial checkpoints remain in the ignored runtime
directory. Batch-level disk checks preserve the 2 GiB reserve.

Dataset: [Keras MNIST](https://keras.io/api/datasets/mnist/), downloaded from the
same TensorFlow-hosted archive used by Keras. Keras documents MNIST's license as
Creative Commons Attribution-Share Alike 3.0.
