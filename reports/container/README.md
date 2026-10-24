# Container deployment verification

The Flask app now runs through Gunicorn in Docker Compose at
`http://127.0.0.1:5000`, connected to PostgreSQL at `db:5432`. Both services are
healthy. The previous host Gunicorn process was stopped during the port handoff.

## Persistence and behavior

- The six original history records were verified unchanged before the handoff.
  Five container upload checks added two predictions and three rejections.
  All 11 complete rows were verified unchanged after recreating both containers.
- PostgreSQL uses the existing `23l-0907_a2_ops_postgres_data` external volume.
  The checkpoint and saved policy remain in the host's DVC-managed champion
  directory, mounted read-only in the app container.
- The checkpoint and policy hashes match their pre-migration values. Registry
  version 2 and threshold 0.6196606457233429 remain unchanged.
- Two validation images produced exactly the saved validation scores, including
  the real image's existing false positive. Unsupported text, corrupt PNG and
  oversized uploads returned 415, 400 and 413 and were recorded in PostgreSQL.
- A database outage produced HTTP 503, followed by 200 after recovery. The same
  Gunicorn master and model worker remained alive during database recreation.
- The upload page, history, health, saved result, stylesheet and JavaScript all
  returned 200 on the final container URL. Debug mode is disabled.
- The 22 serving tests and 14 repository-policy tests passed. DVC reports the
  data and pipelines up to date. No held-out test images were read or evaluated.

## Image and storage

The image uses Python 3.12.15 on the digest-pinned official slim Bookworm base,
with a pinned serving dependency closure and CPU TensorFlow 2.19.1. An isolated
image audit confirmed that credentials, agent directories, Git metadata, training
tools, raw data, model binaries, tests and runtime reports are absent. Only the
serving modules and assets are copied into `/app`.

The app runs as UID/GID 10001 with a read-only root filesystem and model mount,
dropped capabilities, bounded temporary memory, resource limits and bounded logs.
Its health probe uses the HTTP route and does not import the model.

The first build was stopped by the disk guard at 2.4935 GiB free during Docker's
temporary unpack duplication. The 2 GiB reserve was preserved. A deferred export
then completed using the cached dependencies. Removing this task's disposable
BuildKit cache before first startup avoided the duplication; monitored startup
stayed above 3.058 GiB free. About 3.05 GiB remained after all live checks.

The image ID is recorded in `final.json`. Docker's reported image size is
362,707,277 bytes; containerd also retains unpacked layers on the host, so this
number alone does not describe physical disk usage. Build and runtime receipts
record actual available host space.

## Evidence

- `build-initial-attempt.json`: guard-triggered cancellation and minimum space.
- `build.json`: successful deferred export and disk measurements.
- `startup.json`: monitored initial unpack and startup on temporary port 5002.
- `initial.json`: image audit, read-only mounts and six original records.
- `application/verification.json`: validation uploads and rejection checks.
- `recreation.json`: database replacement, degraded health and recovery.
- `application/persistence.json`: five new records verified on final port 5000.
- `final.json`: final deployment, unchanged hashes and all 11 preserved records.
- `routes.json`: final HTTP route and asset checks.

The database backup, row snapshots and runtime logs stay ignored under
`reports/training-runtime/docker/`. See `docs/docker.md` for startup, cleanup and
rebuild instructions that retain the model and history.
