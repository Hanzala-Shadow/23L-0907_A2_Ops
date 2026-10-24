# Cloud model API

The cloud API runs on Vercel Hobby with Python 3.12 and CPU-only inference.
It uses a float32 LiteRT export of champion registry version 2, with the saved
threshold `0.6196606457233429`. The original Keras checkpoint, local Flask app,
PostgreSQL database, and history volume stay on this machine.

The export preserves all 146 validation decisions. The test set is not used for
conversion or deployment checks. See `reports/cloud/export.json` and
`reports/cloud/runtime.json` for probability parity and measured memory use.

## Prepare a model release

Run these commands from the project root with at least 2 GiB free plus 256 MiB
of temporary headroom:

```bash
.venv/bin/python scripts/check_disk.py
.venv/bin/python scripts/export_cloud_model.py
.venv/bin/dvc add models/cloud-api
.venv/bin/dvc push models/cloud-api.dvc
```

The exporter refuses to replace an existing export. It forces inference mode,
uses built-in LiteRT operations without quantization, and compares validation
probabilities and decisions before writing the bundle. The original champion
is never modified. Commit only the DVC pointer, manifest, and verification
reports; keep model files out of Git.

Upload `reports/training-runtime/cloud/champion-api.zip` as the private
`cloud-v1` GitHub release asset. The checked-in manifest pins its SHA-256 and
both contained files. Prepare a short-lived model-only download URL:

```bash
.venv/bin/python scripts/cloud_release_url.py
```

This writes `.env.cloud` with mode 0600. Keep this file local. The signed URL
has a short-lived embedded download token and grants download access only to
that release asset. Refresh it immediately before each build. The GitHub account token and DagsHub credentials stay on this machine.

## Vercel Hobby deployment

The existing `hanzala-shadows-projects` account uses the free Hobby plan.
The project is `23l-0907-a2-ops-api`, region Frankfurt (`fra1`). Hobby includes
2 GB RAM per function; this runtime measured about 179 MiB in local validation
checks. The Python bundle stays below Vercel's 500 MB limit. Inference loads
once per warm function instance; scale-out or a cold start creates another
independent instance. Uploaded images and history are not stored in the API.

`pyproject.toml` points Vercel at `cloud.wsgi:app` and declares the same pinned
runtime as `requirements/cloud.lock`. Vercel prioritizes this dependency manifest
over `requirements.txt`. `vercel.json` downloads and verifies the model
during the cloud build and includes it in the function bundle. The API needs
no database connection, TensorFlow installation, or training credentials.

Deployment uploads only the committed serving modules, dependency files,
manifest, downloader, and Vercel configuration through the authenticated
Vercel integration. Dataset files, local settings, training code, reports,
Git metadata, and agent files are excluded. The model is fetched from the
private release through its temporary signed URL.

Set `MODEL_BUNDLE_URL` and `MODEL_API_KEY` as encrypted Vercel production
variables. Also set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`PYTHONDONTWRITEBYTECODE=1`. Keep these secrets in ignored `.env.cloud`
locally. Each deployment must use a fresh signed download URL and a commit
whose GitHub Actions checks have passed. Automatic Git deployments are off.

Hobby is free for personal, non-commercial projects. Exceeding its usage
allowance pauses the affected feature until its quota resets. No Pro trial,
paid add-ons, or billing details are needed for this assignment. The API can
have a cold-start delay after inactivity. Request bodies are limited by Vercel
to 4.5 MB; the API uses a lower 4 MiB file limit plus multipart headroom.

The initial Render Free attempt requested card verification even with the
$0 plan selected. No card was added and no Render service was created.
`render.yaml` remains an optional configuration reference.

## HTTP contract

`GET /health` returns readiness, model version, hashes, and saved threshold.
`POST /predict` requires an `X-API-Key` header and a multipart `image` file.
It accepts still JPEG, PNG, and WebP images up to 4 MiB and 4 million pixels.
The filename extension must match the decoded image format. Images use the
same EXIF correction, white letterboxing, RGB conversion, and internal
normalization as training.

Successful responses contain `label`, `ai_probability`, `confidence`,
`threshold`, `model`, and image dimensions. Errors use JSON and return 401 for
missing/incorrect keys, 400/415 for invalid images, 413 for oversized inputs,
and 503 with `Retry-After` when the inference slot is occupied.

Use the deployed URL and key from `.env.cloud` to call the API from a server
or Postman. Keep the key out of URLs, browser JavaScript, screenshots, shell
history, and source control. The local Flask app continues using its local
champion until a separate integration step changes that behavior.

After setting the deployed origin as `MODEL_API_URL` in `.env.cloud`, run:

```bash
.venv/bin/python scripts/check_cloud_api.py
```

This checks health, a real and an AI-generated validation sample, authentication,
invalid uploads, and the pixel limit. It writes `reports/cloud/deployment.json`
without keys or signed URLs. It does not use the test set or local history.

Provider references: [Hobby plan](https://vercel.com/docs/plans/hobby),
[Python limits](https://vercel.com/docs/functions/limitations),
[Flask deployment](https://vercel.com/docs/frameworks/backend/flask), and
[LiteRT migration](https://developers.google.com/edge/litert/migration).
