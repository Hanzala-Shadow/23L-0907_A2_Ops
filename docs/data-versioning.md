# Data versioning

## Tracked data

`data/raw.dvc` identifies the unchanged Kaggle version 1 dataset: 975 files and
503,783,378 bytes. Its directory MD5 is
`b2cec67839f742df4653f66d4bcbbafb.dir`.

The pointer contains the directory checksum, byte size, file count and relative
path. DVC stores the image contents as checksum-addressed cache objects. Git
stores this small pointer rather than hundreds of megabytes of images.

`dvc.yaml` defines the `prepare_splits` stage. Its dependencies are the raw
dataset, the audit manifest and the split script. Parameters come from
`params.json`; `dvc.lock` records the parameter values and dependency/output
checksums used by the stage.

## Duplicate policy and splits

The AI-generated files `AiArtData/AiArtData/image33.jpg` and
`AiArtData/AiArtData/image34.jpg` have identical SHA-256 hashes. The split lists
retain `image33.jpg`, the lexicographically first path. The raw dataset keeps
both files, so the downloaded source can still be reproduced exactly.

The seed is 42 and requested proportions are 70/15/15. Within each class, the
script sorts images by `SHA256(seed + NUL + image_sha256)`. This ranking is
independent of CSV row order and random-number-generator versions. Integer
counts use largest remainders; ties prefer test, validation, then training.

| Split | AI-generated | Real-labeled | Total |
| --- | ---: | ---: | ---: |
| Training | 376 | 305 | 681 |
| Validation | 81 | 65 | 146 |
| Test | 81 | 66 | 147 |

The five files in `data/splits` are:

- `train.csv`, `validation.csv`, `test.csv`: relative path, class, numeric label
  and original SHA-256. Resolve paths relative to `data/raw`.
- `excluded_duplicates.csv`: excluded and retained paths, hash and reason.
- `summary.json`: counts, parameters, label mapping, source fingerprint and
  output hashes.

Label 0 means real-labeled; label 1 means AI-generated. The checks require every
selected image to appear once and reject overlap by either path or SHA-256.
Conflicting labels for an identical hash stop preparation rather than assigning
a class automatically. The script also verifies source hashes against the
audit before creating any split lists.

Exact hashes do not identify visually similar images with different encodings.
Further review may reveal near-duplicates or uncertain labels. Do not use test
results to choose parameters or a decision threshold.

## Commands

Run these commands from the repository root with the environment active:

```bash
source .venv/bin/activate
python scripts/check_disk.py
dvc status
dvc repro
dvc push
```

To restore missing data or split lists from the configured remote:

```bash
dvc pull
```

To switch to a committed data version, first save unrelated work, check out the
desired Git revision, and run `dvc checkout`. If its cache objects are missing,
use `dvc pull` to obtain them from the remote.

A forced reproduction was checked with:

```bash
dvc repro --force prepare_splits
```

All five output files were byte-identical to the first run. Data and pipeline
status were clean after the push.

## Storage

The default remote is `.dvc-storage` inside this project. It provides a local
push/pull exercise and is ignored by Git. It is on the same filesystem as the
workspace and cache, so it does not protect against disk loss and is not
available to GitHub users who clone the code. Configure a DagsHub or other
shared remote before sharing a clone-and-pull workflow.

The cache uses copy mode. Workspace files do not share writable inodes with
cache objects. The cache and remote each require approximately one dataset
copy; split lists are small text files and do not duplicate images.

Keep at least 2 GiB free. Before copying data, allow for both the expected new
storage and a safety margin. Never prune unrelated Docker volumes, other
projects or the DVC remote to make room. A future shared remote can replace the
local copy after its contents have been uploaded and verified.

Local command logs and restoration evidence are kept outside Git under
`reports/versioning`. This directory is for terminal/report evidence and may
include machine-specific paths.

See the DVC documentation for [remote storage configuration](https://doc.dvc.org/command-reference/remote/add)
and [pipeline and lockfile fields](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files).
