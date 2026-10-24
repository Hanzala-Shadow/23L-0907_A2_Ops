# VisionGuard

An image classifier for AI-generated and real-labeled images. The repository
currently contains dataset download, audit, deduplication and split preparation.
Model training, the Flask application and deployment are the next stages.

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

Copy `.env.example` to `.env` when configuring services. Keep credentials out of
Git. The runtime dependency list is separate from development dependencies so
the serving image will not include training and analysis tools.

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
dvc pull
python scripts/check_disk.py
dvc repro
dvc status
```

`dvc pull` requires the configured remote storage to be available. The current
remote is a project-local `.dvc-storage` directory used for push/pull verification;
it is ignored by Git and is not a shared or off-machine backup. A fresh clone
needs the same storage or a separately configured shared remote. GitHub and
DagsHub remotes have not been configured.

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

## Storage and checks

Keep at least 2 GiB free on the Linux filesystem. Downloads and split generation
check this reserve with an additional safety margin. Check space before DVC
copies, training and Docker builds. Keep one Python environment, avoid pip
download caches, and do not copy images into split directories.

```bash
python scripts/check_disk.py
pytest -q
ruff check scripts tests
python scripts/check_repository.py --history
```

The repository check rejects local configuration, credentials, raw data,
oversized Git files, decorative emoji and agent coauthor trailers. Local setup
logs, cache files and storage directories are ignored. Dataset figures and
audit tables are retained as experiment evidence.
