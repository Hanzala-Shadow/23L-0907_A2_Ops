# Final model selection and evaluation

`config/selection.json` records the decision rule before the test set is opened.
The model is selected by minimum recorded validation cross-entropy across the CNN,
both transfer-learning phases and all eight tuning trials. The selected checkpoint
is unchanged after selection; there is no retraining on validation or test images.

The threshold minimizes `2 * false positives + false negatives` on validation.
A false positive means calling a real-labeled image AI-generated. For this
demonstration, falsely discrediting authentic work is treated as twice as costly
as missing an AI image. This is a stated product assumption, not a measured cost.
Ties prefer fewer false positives, then proximity to 0.5, then the higher threshold.
Candidates include midpoints between distinct validation scores and 0, 0.5 and 1.
Predictions at or above the threshold are AI-generated.

```bash
source .venv/bin/activate
python -m scripts.select_model
git add reports/selection
git commit -m "Lock model and validation thresholds"
PYTHONHASHSEED=42 python -m scripts.evaluate_final
python -m scripts.register_model
```

Source and policy must be committed before selection. The resulting plan must
also be committed before final evaluation. The plan binds model hashes, thresholds,
validation prediction hashes, data versions and the preprocessing contract. The
two fixed models in the final evaluation are the custom CNN and the selected
MobileNetV2, satisfying the assignment's comparison without choosing from test
results. Each model receives one prediction pass over the fixed 147-image test set.
Accuracy, precision, recall, F1, ROC AUC, cross-entropy and confusion counts are
computed from those saved probabilities. ROC AUC uses probabilities, not labels.

Before inference, an exclusive durable marker is created in the ignored runtime
directory. Completed probabilities are saved atomically and checked by hash on
reuse. Retrying reporting or uploads reuses them. An interrupted pass without
durable probabilities is refused rather than silently repeated. Preserve this
ledger. Final receipts and probabilities also belong in the committed evaluation
report. Do not rerun selection or tune the threshold after viewing test results.

The frozen MobileNetV2 and selected tuned MobileNetV2 become separate versions of
`23L-0907_A2_Ops` in the DagsHub MLflow registry. `champion` identifies the selected
version. Version tags record the exact threshold, preprocessing policy, source
checkpoint hash and selection-plan hash. Registration reuses existing versions
with the same source run when retried. The frozen reference retains its own
validation-selected threshold and is not selected using test scores.

```bash
python -m scripts.predict_registry path/to/a/validation/image.png
```

This separate example resolves `champion` once, loads that immutable registry
version, applies the shared RGB letterbox preprocessing and uses the threshold
from that same version. Use a validation image for this demonstration, preserving
the single final test pass. The local serving bundle contains `best.keras` and
`policy.json` under `models/champion/` and is versioned with DVC. The future server
will load this bundle once at startup, avoiding DagsHub requests per image.

The reported confidence is the selected class's uncalibrated probability score;
it is not proof of image provenance. Because error costs differ, a real verdict
can have a probability score below 0.5. Dataset labels include photographs and
artwork and cannot establish authenticity for arbitrary uploads.

All operations check disk space with a buffer above the 2 GiB reserve. Native
model downloads for the alias example use a temporary directory that is removed
after the prediction. Model binaries, credentials, runtime logs and test-pass
markers are excluded from Git.

References: [DagsHub MLflow integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
and [MLflow registry aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/).
