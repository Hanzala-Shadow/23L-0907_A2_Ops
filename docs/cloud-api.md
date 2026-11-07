# Cloud model API

Live health check: [visionguard-image-api.vercel.app/health](https://visionguard-image-api.vercel.app/health).
Prediction endpoint: `POST https://visionguard-image-api.vercel.app/predict`.
The API key is stored only in ignored `.env.cloud` and Vercel production secrets.

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

The existing `cloud-v1` GitHub release contains the model-only bundle
`champion-api.zip`. The repository is now public, so cloud builds can use its
stable release URL without an account token or expiring download link:

```text
https://github.com/Hanzala-Shadow/23L-0907_A2_Ops/releases/download/cloud-v1/champion-api.zip
```

The checked-in manifest pins the archive SHA-256 and both contained files. The
downloader accepts only this exact public release URL or the previous signed
GitHub asset host. It verifies size, archive members and checksums before writing
the model. Never replace a release asset without updating its pinned manifest.

## Vercel Hobby deployment

The existing project uses the free Hobby plan and region Frankfurt (`fra1`).
Its public origin is `https://visionguard-image-api.vercel.app`. Hobby includes
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
public release and checked against the committed manifest.

Set `MODEL_BUNDLE_URL` and `MODEL_API_KEY` as encrypted Vercel production
variables. Also set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`PYTHONDONTWRITEBYTECODE=1`. Keep these secrets in ignored `.env.cloud`
locally. Each deployment uses the pinned release URL and a commit
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

`GET /` displays the VisionGuard upload website. Visitors choose or drop one
image, preview it locally, and submit it to `POST /classify`. The page displays
the likely class, model confidence and a caveat. No API key is sent to the
browser. Both upload routes share the same loaded interpreter, saved threshold,
image validation and single inference slot. The cloud website does not write to
the local PostgreSQL database.

The browser route uses a signed, secure, HttpOnly, SameSite=Lax session cookie
with a 30-minute lifetime and a matching form token. Its signing key is derived
on the server from the existing API secret using a separate HMAC purpose.
Cross-site form submissions are rejected. A bounded in-memory limiter allows
six checks per minute per session and twenty per minute per warm function instance,
returning 429 with Retry-After. Limits reset on cold starts and do not provide
a shared quota across instances. The existing key-protected `/predict` route
continues serving the local Flask app and Postman.

Static CSS and JavaScript are served from the same origin under a restrictive
Content Security Policy. The page uses a 55-second browser timeout and readable
validation, connectivity, busy-service and session-expiry messages.

`GET /` also includes collapsible API usage instructions.
Public pages and responses use the product name. Internal model registry names
and existing PostgreSQL rows retain their original provenance. The earlier
assignment API origin remains an alias so links in the submitted report work.


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
history, and source control. The local Flask app uses the
[server-side cloud integration](cloud-integration.md) while PostgreSQL history
stays on this machine.

After setting the deployed origin as `MODEL_API_URL` in `.env.cloud`, run:

```bash
.venv/bin/python scripts/check_cloud_api.py
```

This checks health, three generated images against the local LiteRT outputs,
authentication, invalid uploads, and the pixel limit. Dataset images stay local.
For local reference inference it uses the isolated lightweight runtime at
`reports/training-runtime/cloud/venv/bin/python`, or `CLOUD_RUNTIME_PYTHON` if set. It writes `reports/cloud/deployment.json`
without keys or signed URLs. It does not use the test set or local history.

Provider references: [Hobby plan](https://vercel.com/docs/plans/hobby),
[Python limits](https://vercel.com/docs/functions/limitations),
[Flask deployment](https://vercel.com/docs/frameworks/backend/flask), and
[LiteRT migration](https://developers.google.com/edge/litert/migration).
