# 23L-0907_A2_Ops

The final [assignment audit](docs/assignment-checklist.md) maps every rubric ID
to implementation and evidence. The submission report is
`output/pdf/23L-0907_VisionGuard.pdf`. The
[Postman collection](postman/visionguard.postman_collection.json) was checked
from a separate GitHub-hosted network; results are in `reports/labs/postman/`.

[![CI](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops/actions/workflows/ci.yml)

VisionGuard is an image classifier for AI-generated and real-labeled images. The repository
contains dataset download, audit, deduplication, split preparation, CPU training
pipelines, and a Flask image classifier with PostgreSQL history and a Docker
Compose deployment. An eight-trial Keras Tuner search varies learning
rate, dropout and the number of unfrozen backbone layers; see
[tuning instructions](docs/tuning.md) and [results](reports/training/tuning-results.md).

The final model, validation-derived threshold, single test evaluation and
champion registry version are documented in the
[final evaluation](reports/evaluation/README.md) and
[selection workflow](docs/model-selection.md).

The [Flask application guide](docs/web-app.md) covers local setup, upload
validation, saved results, history and health checks. Start PostgreSQL with
`docker compose -f compose.postgres.yaml up -d --wait`, then run
`gunicorn --config gunicorn.conf.py wsgi:app` in the configured environment.
The [Docker guide](docs/docker.md) covers the guarded image build and Compose
stack, reusing the champion model and persistent PostgreSQL history.
The [GitHub Actions guide](docs/ci.md) describes automated linting, tests, Docker
build checks and an isolated Compose integration check on hosted runners.

The public [GitHub repository](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops)
contains the code, audit and versioning metadata. The public
[DagsHub repository](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops)
mirrors GitHub and provides DVC storage and the MLflow endpoint. The code, report, experiments and versioned artifacts are publicly readable.
DagsHub storage clients may still require a free signed-in account.

## Submission

Submit the [15-page report](output/pdf/23L-0907_VisionGuard.pdf), this GitHub
repository, the [DagsHub project](https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops),
and the [live VisionGuard API](https://visionguard-image-api.vercel.app).
The separate [CI lab](https://github.com/Hanzala-Shadow/23L-0907_A2_Ops-ci-lab)
contains the Python 3.11/3.12 failing-then-passing exercise. The Postman
collection linked above is included in this repository. Share the prediction
API key privately with the evaluator; keep it out of the report and public links.
The report retains student details for grading; the live product uses only
VisionGuard branding. The earlier API address remains an alias for report links.

## Environment

Use Python 3.12 and Docker Compose. Training and inference use CPU TensorFlow.
Exact Python dependency versions are recorded in `requirements/locked.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --no-compile -r requirements/locked.txt
source .venv/bin/activate
python scripts/check_disk.py
python scripts/check_environment.py
```

The environment check's optional `--smoke` flag tests synthetic Keras training,
MLflow logging, model save/load, Flask and a local SQL query. SQLAlchemy is
constrained to 2.0 because MLflow 2.x is incompatible with SQLAlchemy 2.1.
MLflow's declared TensorFlow range ends at 2.19.0; the installed 2.19.1 patch
release passed these checks but emits a range warning.

Use `.env.example` as a reference for local `.env` settings. Preserve existing
credentials when adding service settings, and keep them out of Git. The runtime
dependency list and pinned serving lock are separate from development dependencies
so the serving image excludes project training and analysis tools.

## Dataset

Source: [AI Generated Images vs Real Images on Kaggle](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images),
version 1. The source lists the Community Data License Agreement - Sharing 1.0.

The raw dataset has 975 images: 539 AI-generated and 436 real-labeled. All images
decoded successfully. One exact duplicate pair occurs within the AI class,
leaving 974 unique images. The raw files remain unchanged; only split-list
membership excludes the redundant copy.

```bash
python scripts/download_dataset.py --reserve-gib 2
python scripts/audit_dataset.py --reserve-gib 2 --seed 42
```

The download command is for a new workspace without `data/raw`. It verifies ZIP
checksums and removes the archive after extraction. The audit records file size,
dimensions, colour mode, format, SHA-256, decode errors and duplicates.

See the [audit report](reports/data/audit.md),
[image manifest](reports/data/image_manifest.csv),
[class statistics](reports/data/class_summary.csv) and
[duplicate list](reports/data/duplicate_files.csv).

The real-labeled folder includes photographs and artwork. Also, 79 images use
palette, grayscale or RGBA modes. Training and serving must share an explicit
RGB conversion, transparency and aspect-ratio policy.

## Data versioning and splits

DVC tracks the original dataset and the generated split lists. Git tracks the
code, audit tables, parameters, DVC pointer and pipeline lockfile.

```bash
source .venv/bin/activate
python scripts/check_disk.py
dvc pull -j 2
dvc repro
dvc status
```

The default DVC remote is DagsHub's S3-compatible DVC storage. A fresh clone needs
DagsHub authentication before `dvc pull`; see the
[credential setup instructions](docs/data-versioning.md#storage). Credentials
belong in ignored local configuration. Git contains the storage address only.

Split proportions and seed are in `params.json`: 70% training, 15% validation,
15% test, seed 42. Each class is ranked by a seeded SHA-256 key. Integer counts
use largest-remainder rounding. The script verifies original file hashes,
removes duplicate membership deterministically, and checks split coverage and
overlap by both path and hash. Label 0 is real; label 1 is AI-generated.

| Split | AI-generated | Real-labeled | Total |
| --- | ---: | ---: | ---: |
| Training | 376 | 305 | 681 |
| Validation | 81 | 65 | 146 |
| Test | 81 | 66 | 147 |

Lists, exclusion records and their hashes are under `data/splits/`, restored by
DVC. The test split is reserved for final evaluation; model and threshold
selection must use training and validation data.

See [data versioning](docs/data-versioning.md) for storage and restore commands.

## CPU baseline training

The CNN uses 128 x 128 RGB inputs, aspect-preserving white padding, train-only
augmentation, three convolutional blocks, batch normalization and dropout.
Parameters are in `params.json`. The decoder accepts image paths and binary
uploads so future inference can reuse the training preprocessing.

```bash
source .venv/bin/activate
python scripts/check_disk.py
PYTHONHASHSEED=42 dvc repro train_cnn
dvc push -j 2
```

The training stage logs to DagsHub MLflow using ignored local credentials. It
saves the checkpoint with the lowest validation loss, training curves and
validation metrics. The test set remains reserved for final evaluation.
See [training details](docs/training.md) for architecture, data handling,
reproducibility and artifact locations.

## Transfer learning

MobileNetV2 uses ImageNet weights, first with a frozen backbone and then with
only its last inverted residual block fine-tuned. The saved models include their
own [-1, 1] input scaling. BatchNorm statistics stay frozen in both phases.

```bash
source .venv/bin/activate
python scripts/check_disk.py
PYTHONHASHSEED=42 dvc repro compare_models
dvc push -j 2
```

The two phases use TensorFlow autologging on DagsHub. They retain independent
checkpoints and validation reports so fine-tuning can be compared with the
frozen model and custom CNN. See [transfer-learning details](docs/transfer-learning.md).

The initial [validation comparison](reports/training/model_comparison.md) favours
the frozen MobileNetV2 checkpoint: 78.08% accuracy, F1 0.8000 and ROC-AUC 0.8418.
Fine-tuning did not improve validation loss. These are provisional validation
results; the test set has not been evaluated. See the
[result analysis](reports/training/transfer-results.md).

## Cloud website and model API

The [live VisionGuard website](https://visionguard-image-api.vercel.app) accepts
image uploads directly in the browser, with a preview, likely class, confidence
and an honest caveat. Website visitors do not need an API key. The separate
`/predict` API remains key-protected for server clients. Uploaded images are
processed in memory and not saved by the cloud service.


The standalone API uses a float32 LiteRT export of champion version 2 for
Vercel's free Hobby plan. It preserves the saved threshold and all validation
decisions. The local Flask app and PostgreSQL history continue running locally.
The [live health check](https://visionguard-image-api.vercel.app/health) exposes
readiness; predictions require an API key. See [cloud setup and HTTP contract](docs/cloud-api.md) and
[export verification](reports/cloud/export.json).

## Storage and checks

The local Flask app can use the deployed API with bounded timeouts and
PostgreSQL history. See [cloud integration](docs/cloud-integration.md) for the
small client container, failure handling, and preserved local storage.

Keep at least 2 GiB free on the Linux filesystem. Downloads and split generation
check this reserve with an additional safety margin. Check space before DVC
copies, training and Docker builds. Keep one Python environment, avoid pip
download caches, and do not copy images into split directories.

```bash
python scripts/check_disk.py
python -m pytest -q
ruff check scripts tests visionguard
python scripts/check_repository.py --history
```

The repository check rejects local configuration, credentials, raw data,
oversized Git files, decorative emoji and agent coauthor trailers. Local setup
logs, cache files and storage directories are ignored. Dataset figures and
audit tables are retained as experiment evidence.
