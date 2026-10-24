# Local Flask app with Vercel inference

Image upload now follows this path:

1. The local Flask server checks the form token and validates the image in memory.
2. It sends the image over HTTPS to the pinned Vercel `/predict` endpoint.
3. It validates the returned champion identity, threshold, score, verdict,
   confidence, and image dimensions.
4. It commits the result to the existing local PostgreSQL table and displays
   the saved result. History pages continue reading that database.

The API key stays on the server. The outbound filename is a generic
`upload.png`, `upload.jpg`, or `upload.webp`; the original sanitized filename
stays in local history. Neither server retains uploaded image files. The
upload page explains that classification sends the image to the cloud.

## Configure and run

The ignored `.env.cloud` from deployment supplies the deployed origin and key.
The setup command copies only those two values into ignored `.env`, enables
remote inference, and preserves PostgreSQL and Flask session credentials:

```bash
.venv/bin/python scripts/configure_remote.py
sudo docker compose config --quiet
sudo docker compose up -d --no-build --no-deps --wait web
```

The setup command sets `COMPOSE_FILE=compose.yaml:compose.cloud.yaml` in `.env`.
Compose therefore keeps the same project, database container, localhost ports,
external history volume, and read-only champion mount. It changes only the web
service to `visionguard:client`. This small image uses `Dockerfile.client` and
`requirements/client.lock`; it does not install TensorFlow, DVC, or MLflow.
The original `visionguard:local` image and champion files remain available.

GitHub Actions builds and checks the client image, then publishes the private
`cloud-client-image` workflow artifact for one day. This machine downloads
and loads that image instead of rebuilding TensorFlow. Before loading it,
verify its checksum and leave enough space for the image import as well as
the 2 GiB reserve. A normal build on a machine with sufficient space is:

```bash
sudo docker compose build web
sudo docker compose up -d --no-deps --wait web
```

The native Python environment can also run `gunicorn --config gunicorn.conf.py
wsgi:app` with the same remote settings when port 5000 is free.

## Timeouts and failures

`MODEL_API_CONNECT_TIMEOUT=3` and `MODEL_API_READ_TIMEOUT=45` set separate
connection and read timeouts. Connections are reused per worker thread. POSTs
are not automatically retried, and redirects are rejected so a key is never
forwarded to another destination. TLS verification remains enabled.

The local upload limits are capped at the cloud contract: 4 MiB and 4 million
pixels. Model metadata must match the saved champion and checked-in cloud
manifest. A score must be finite and within `[0, 1]`; its verdict and confidence
must match the saved threshold. A remote mismatch never becomes a successful
history entry. New successful records store the actual LiteRT artifact hash;
older native-model records retain their original hashes.

| Failure | Local response | History status |
| --- | --- | --- |
| Connect or read timeout | 504, retry message | error, `cloud_timeout` |
| Network/TLS failure or upstream server failure | 503 | error, `cloud_unavailable` |
| Busy service or rate limit | 503 with retry guidance | error, `cloud_busy` |
| Incorrect API credentials | 503, configuration message | error, `cloud_authentication` |
| Invalid JSON, redirect, model mismatch, inconsistent score | 502 | error, `cloud_invalid_response` |
| Upstream image rejection | 400/413/415 | rejected, `cloud_rejected_upload` |
| PostgreSQL unavailable | 503; no unsaved success shown | no record until storage recovers |

Only safe messages and error codes are recorded; upstream response bodies,
keys, database credentials, and raw exception details are not exposed. There
is no automatic local-model fallback. A failed cloud request remains visible
as a failed attempt. Users can retry from the upload page.

`/health` checks the database, disk reserve, and cloud readiness. Cloud probes
use short 1-second connect and 1.5-second read timeouts and cache successful
results for 30 seconds, failures for 5 seconds. Successful predictions refresh
readiness too. After an authentication failure, correct the key and recreate
the web service. Cloud unavailability does not prevent the app from starting
or serving upload, history, and saved-result pages.

The existing disk guard still requires 2 GiB plus a 128 MiB buffer for uploads.
No database schema migration or row rewrite is required. Recreating only the
web service preserves all existing PostgreSQL rows.

## Verification

Unit checks use synthetic images and a fake HTTP transport to cover timeouts,
network/authentication failures, malformed or oversized JSON, redirects,
identity mismatches, saved results, failure records, and readable history.
The live check sends generated images through localhost to Vercel and
independently reads PostgreSQL. It does not upload dataset images or read the
held-out test set.

On this machine, `scripts/check_integration.py` uses the ignored pre-change
snapshot at `reports/training-runtime/integration/before.json` and the existing
isolated LiteRT environment to compare saved predictions with local inference.
Running it creates seven verification records, including two simulated failure
records in PostgreSQL. Run it only when those extra history entries are wanted:

```bash
.venv/bin/python scripts/check_integration.py
```

The completed checks are recorded in `reports/integration/verification.json`.
`reports/integration/final.json` also records the browser upload, preserved
database and champion, active client image, and remaining disk space. All 11
original records were preserved; verification added eight records in total.
The downloaded image archive was removed after the verified image was loaded.

To deliberately return to native inference, use the retained TensorFlow image
with only `compose.yaml` and `INFERENCE_BACKEND=local`. Preserve the external
PostgreSQL volume and the existing database/session secrets.
