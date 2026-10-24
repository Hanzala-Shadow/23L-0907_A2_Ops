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
dvc push -j 2
```

To restore missing data or split lists from the configured remote:

```bash
dvc pull -j 2
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

The default remote, `dagshub`, uses `s3://dvc` at
`https://dagshub.com/Hanzala-Shadow/23L-0907_A2_Ops.s3`. This is DagsHub's DVC
object store; its separate general-purpose bucket is not used for this dataset.
The repository is private and connected to the matching GitHub repository.
Push code and DVC metadata to GitHub; DagsHub syncs those Git commits. Upload
dataset contents separately with `dvc push -j 2`.

The S3 adapter is included in the locked development environment. On a new
machine, obtain access to both private repositories, clone GitHub, install the
locked requirements, and authenticate with the DagsHub client:

```bash
source .venv/bin/activate
umask 077
dagshub login
python - <<'PY'
import os
from pathlib import Path

from dagshub.auth import get_token
from dagshub.common import config
from dvc.config import Config

token = get_token(fail_if_no_token=True)
with Config(dvc_dir=".dvc").edit(level="local") as local:
    remote = local.setdefault("remote", {}).setdefault("dagshub", {})
    remote["access_key_id"] = token
    remote["secret_access_key"] = token
os.chmod(".dvc/config.local", 0o600)
cache = Path(config.cache_location)
if cache.exists():
    cache.chmod(0o600)
PY
python scripts/check_disk.py
dvc pull -j 2
dvc status
```

The authentication token is stored in `.dvc/config.local` and the client's local
credential cache. Neither belongs in Git. Reauthorize and repeat this setup when
the token expires. `.env.example` records the MLflow address and username; keep
any MLflow password or token in an ignored environment file or process environment.

The cache uses copy mode. Workspace files do not share writable inodes with
cache objects. The workspace and local cache each require approximately one
dataset copy; split lists are small text files and do not duplicate images.

Keep at least 2 GiB free. Before copying data, allow for both the expected new
storage and a safety margin. Never prune unrelated Docker volumes, other
projects or the DagsHub remote to make room. A fresh restore with an empty cache
needs about 1 GiB for the cache and workspace, in addition to the 2 GiB reserve
and a safety margin.

## Remote restore verification

On 24 October 2026, a fresh GitHub clone with an empty cache restored all 981 DVC
objects from DagsHub. SHA-256 checks matched all 975 original images and all five
split files. The restored lists covered 974 unique images with no path or hash
overlap between splits. DVC reported the cache and remote in sync.

Free space stayed above 4.54 GiB during verification. The temporary clone was
removed afterward. The initial `.dvc-storage` backup was checked against the
cache and removed only after the remote restore passed, recovering about
484 MiB of allocated space. The original images and local DVC cache remain
available for training.

Local command logs and restoration evidence are kept outside Git under
`reports/versioning`. This directory is for terminal/report evidence and may
include machine-specific paths.

See the DVC documentation for [remote storage configuration](https://doc.dvc.org/command-reference/remote/add)
and [pipeline and lockfile fields](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files),
and DagsHub's [DVC storage setup](https://dagshub.com/docs/feature_guide/dagshub_storage/).
