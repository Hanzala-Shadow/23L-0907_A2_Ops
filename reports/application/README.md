# Application verification

The Flask app serves the DVC-tracked champion checkpoint, MLflow model version 2,
with the unchanged validation threshold of 0.6196606457233429. Startup verifies
the checkpoint hash and preprocessing policy. One Gunicorn worker loads and warms
one model; request inference is serialized with a lock.

## Checks completed

- All 77 automated tests passed, including upload validation, shared preprocessing,
  history transactions, low-space rejection, database failures and model loading.
  The model-loading test counts exactly one checkpoint load across two requests.
- A browser upload showed the preview, saved result and confidence caveat. History
  displayed all six saved attempts. The narrow layout had no page overflow.
- Two validation-image uploads matched their previously saved validation scores.
  The AI sample scored 0.9492709636688232. The real sample scored
  0.7347276210784912 and was a false positive, consistent with its existing score.
  This result was retained; neither the model nor threshold was adjusted.
- Unsupported text, corrupt PNG and oversized uploads returned 415, 400 and 413,
  respectively, with rejection records in PostgreSQL. Image bytes are not stored.
- Independent SQL queries verified all five scripted request records. Stopping
  PostgreSQL changed health from 200 to 503; restarting restored 200 and all rows.
  The model worker stayed running throughout the database outage.
- Both Gunicorn and a temporary Flask CLI server reported debug mode disabled.
  The temporary server was stopped after verification.
- Verification used validation images and synthetic files. No held-out test image
  was read or re-evaluated.

## Evidence

- `verification.json`: live HTTP checks and PostgreSQL row verification.
- `outage-check.json`: degraded health, recovery and unchanged model worker.
- `persistence.json`: existing records verified after the database restart.
- `flask-server.json`: Flask CLI entry point and shutdown check.

PostgreSQL 17.11 runs in a digest-pinned container with a persistent named volume,
localhost port binding and bounded logs. Approximately 4.66 GiB remained free
after setup, above the 2 GiB reserve. Uploads are blocked before processing or
saving when free space falls below 2 GiB plus a 128 MiB buffer.

Startup commands, routes and configuration are documented in `docs/web-app.md`.
Runtime logs and browser screenshots are local, ignored files.
