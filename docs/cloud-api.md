# Cloud model API

The cloud API runs on Render's free Python web service: 0.1 CPU and 512 MB RAM.
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
expires in about one hour and grants download access only to that release
asset. The GitHub account token and DagsHub credentials stay on this machine.

## Render settings

Connect only `Hanzala-Shadow/23L-0907_A2_Ops`, branch `main`. Use `render.yaml`
as the configuration reference, or create a Blueprint from that file:

- Service: `23l-0907-a2-ops-api`, Python runtime, Frankfurt, **Free** plan.
- Python: `3.12.15`; build installs `requirements/cloud.lock`, then runs
  `python scripts/fetch_cloud_model.py`.
- Start: `gunicorn --config cloud/gunicorn.conf.py cloud.wsgi:app`.
- Health check: `/health`; one worker, two threads, one inference at a time.
- Set `MODEL_BUNDLE_URL` and `MODEL_API_KEY` from `.env.cloud` as secret
  environment variables. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  and `PYTHONDONTWRITEBYTECODE=1`.
- Disable automatic deploys. Refresh `MODEL_BUNDLE_URL` before each manual
  build, then deploy a commit whose GitHub Actions checks have passed.

Builds download and verify the model once. The built service retains the model
across restarts; uploaded images and prediction history are not stored in the
cloud API. The API needs no database connection or training credentials.

Render's free service sleeps after 15 minutes without inbound traffic. The
next request can take roughly a minute to wake it. It has 750 free instance
hours per workspace per month and limited build minutes and bandwidth. Do not
add billing details or upgrade the service for this assignment.

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

Provider references: [free service limits](https://render.com/docs/free),
[compute plans](https://render.com/docs/compute-plans),
[Blueprint configuration](https://render.com/docs/blueprint-spec), and
[LiteRT migration](https://developers.google.com/edge/litert/migration).
