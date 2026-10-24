# Flask application and PostgreSQL history

The app can also run with PostgreSQL through the [Docker Compose stack](docker.md),
using the same champion bundle and history volume.

The app reads the local champion bundle from `models/champion/`. It does not
contact DagsHub during requests. The checkpoint SHA-256, input contract, label
mapping and saved preprocessing policy are checked at startup. The threshold
comes from `policy.json`; an environment override must match it exactly.

## Start locally

The existing pinned Python environment contains Flask, Gunicorn, SQLAlchemy,
psycopg and CPU TensorFlow. No extra Python packages are required for this stage.

```bash
source .venv/bin/activate
dvc pull models/champion.dvc
python -m scripts.configure_web
sudo docker compose --env-file .env -f compose.postgres.yaml up -d --wait
gunicorn --config gunicorn.conf.py wsgi:app
```

Open `http://127.0.0.1:5000`. This machine's Docker socket requires `sudo`; no
Docker group membership was changed. The PostgreSQL container binds to localhost
port 5433 and stores data in a named volume. The image is pinned by digest and
reports PostgreSQL 17.11. Database logs are limited to two 5 MiB files. WAL settings
and a 256 MiB container memory limit keep the local database modest in size.

`configure_web` fills missing local settings, generates secrets for placeholder
values and preserves existing credentials. `.env` stays ignored with mode 0600.
The app refuses a missing secret key, a non-PostgreSQL production database URL,
or a disk reserve below 2 GiB. The model files stay in DVC; only their pointer is
committed to Git.

For the Flask development server on a separate local port:

```bash
FLASK_DEBUG=0 flask --app wsgi:app run --no-debugger --no-reload --host 127.0.0.1 --port 5001
```

Gunicorn is the normal serving command. Its configuration uses one worker and two
threads, so a single model instance stays in memory. A lock serializes calls into
that model. Creating a model on every request would repeatedly read weights and
initialize TensorFlow, increasing latency and memory use. Both entry points keep
debug mode off; disabling the Flask reloader avoids an additional startup process.

## Routes

| Method and path | Behavior |
| --- | --- |
| `GET /` | Upload page and session-bound form token |
| `POST /predict` | Validate, classify and commit a history record; browser requests redirect to the saved result |
| `GET /result/<uuid>` | Saved successful result or rejected-attempt details |
| `GET /history?page=1` | Newest-first, paginated history, 20 rows per page |
| `GET /health` | Database connectivity, model version, saved threshold, free disk space and debug state |

`Accept: application/json` returns JSON from prediction, result and history routes
for local integration checks. Browser requests receive HTML and friendly error
pages. Successful JSON predictions return 201; browser predictions use a 303
redirect to prevent form resubmission on refresh.

## Uploads and inference

JPEG, PNG and still WebP files are accepted, up to 8 MiB and 16 megapixels by
default. The filename is sanitized and the decoded format must match its suffix.
Empty files, corrupt images, oversized bodies, pixel limits and animations are
rejected before inference. File type alone is not trusted.

The Flask request class uses an in-memory stream factory, including for files
above Werkzeug's normal temporary-file threshold. Request and multipart limits
bound memory use. Uploaded bytes are never saved to a server directory or database.
The model receives the same EXIF orientation correction, white transparency
compositing, RGB conversion and aspect-preserving white letterbox as training.
Normalization remains inside the checkpoint. Changing preprocessing would change
the model's inputs and undermine the validation-derived decision policy.

The saved threshold is 0.6196606457233429. Scores at or above it receive an
AI-generated verdict. Confidence is the selected class's uncalibrated score,
which is explained on the result page. A real verdict can have a score below 50%
because the cost-based policy deliberately treats false AI accusations as more
costly. The interface states that a verdict does not establish image provenance.

## PostgreSQL records

Each accepted classification and rejected upload gets a UUID, UTC timestamp,
sanitized filename, status, processing time, model name/version/run/hash,
preprocessing policy and threshold. Successful checks also record image size,
dimensions, content hash, label, AI probability and confidence. Rejections store a
short error code and message. Image pixels, form tokens and credentials are not
stored. History is shared by this local demo.

SQLAlchemy creates the initial table and indexes if absent, leaving existing rows
intact. Every operation uses a short session and transaction; sessions are not
shared between requests. The connection pool checks stale connections. An
unavailable database produces 503 rather than presenting an unsaved successful
result. Database exceptions are logged by class without exposing connection
details in responses. A named PostgreSQL volume preserves history across container
restarts. Stop the database with `docker compose -f compose.postgres.yaml stop`;
starting it again reuses that volume.

The health route returns 503 if PostgreSQL is unavailable or disk space enters
the reserve buffer. Uploads are blocked with 507 before processing or saving
when less than 2 GiB plus a 128 MiB buffer is available. Unit tests simulate this
condition without filling the disk.

## Verification

```bash
python -m pytest -q
python -m scripts.check_web
python -m scripts.check_web --verify-only
```

Unit tests use synthetic images and isolated in-memory SQLite for transactions.
The production configuration requires PostgreSQL, and the live check independently
queries PostgreSQL rows. The live check uses two validation images and compares
their scores with saved validation predictions. It also checks unsupported,
corrupt and oversized uploads. It never reads the held-out test images.

Verification records are in `reports/application/`. The PostgreSQL restart check
confirmed a degraded health response, recovery, persistent rows and the same
Gunicorn model worker throughout the outage. Browser checks cover the upload
preview, result, caveat, history and narrow viewport layout. Runtime logs and
screenshots remain under ignored `reports/training-runtime/web/`.

References: [Flask uploads](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/),
[SQLAlchemy sessions](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
and [official PostgreSQL image](https://hub.docker.com/_/postgres).
