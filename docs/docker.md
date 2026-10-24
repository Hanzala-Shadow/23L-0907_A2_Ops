# Docker Compose deployment

The stack runs the Flask application through Gunicorn and connects it to
PostgreSQL at `db:5432` on the Compose network. The browser-facing port binds to
`127.0.0.1:5000`. PostgreSQL retains its existing localhost port 5433 for local
administration and verification.

## Start

```bash
source .venv/bin/activate
python -m scripts.configure_web
dvc pull models/champion.dvc
sudo docker volume create 23l-0907_a2_ops_postgres_data
python -m scripts.build_container --sudo
sudo docker builder prune --all --force
sudo docker compose up -d --no-build --wait --wait-timeout 120
```

The volume-create command is idempotent: it reuses an existing named volume.
`POSTGRES_VOLUME_NAME` can select another existing volume. Compose declares this
volume external, so stack removal does not own its lifecycle. On this machine
the volume is the same one created during the Flask stage; the six original
history records were backed up before the container migration.

Stop a host Gunicorn process using port 5000 before starting the container on
that port. An alternate local port can be set through `WEB_PORT`. The app waits
for PostgreSQL's health check before starting. Both services use restart policies;
the web health probe checks the HTTP readiness route without importing TensorFlow
or loading a second model.

## Model and history persistence

`./models/champion` is mounted read-only at `/app/models/champion`, including the
checkpoint and its saved policy. It must exist before startup. The model is not
copied into the image or downloaded on requests. Each new worker verifies its
SHA-256, loads it once, warms up on synthetic pixels and retains the saved
threshold. One worker and two threads match the CPU serving configuration.

The PostgreSQL named volume stores history independently of the app image and
container. Recreating or rebuilding the app does not reset the database. A
database restart also retains rows. Keep the volume and the DVC-managed champion
directory when moving or recreating the stack.

```bash
sudo docker compose ps
curl --fail http://127.0.0.1:5000/health
sudo docker compose restart web
sudo docker compose stop
sudo docker compose up -d --no-build --wait
```

`docker compose down` removes the containers and network while preserving the
external history volume and host model files. Do not delete the history volume
when cleaning build artifacts. A local pre-migration `pg_dump` is stored under
ignored `reports/training-runtime/docker/` with mode 0600.

## Build size and free space

`requirements/serving.lock` pins the runtime dependency closure to the versions
used in the host training environment. The Python slim and PostgreSQL images are
pinned by digest. Project training code, DVC, MLflow, dataset files, model binaries,
Git metadata, local credentials and development tooling are excluded from the
image. `.dockerignore` starts by excluding everything and admits only serving
source, templates, static assets and the runtime lock.

The dependency-install step downloads wheels into a temporary memory mount and
disables pip caching and bytecode generation. It removes TensorFlow's C++
development headers before committing the layer; inference libraries and package
licenses remain installed. No compiler toolchain is installed.

Use the guarded build command on this disk. It requires 4 GiB free to start,
samples available space every 0.1 seconds and cancels the build below 2.5 GiB,
leaving a buffer above the 2 GiB reserve. A failed or cancelled build needs
inspection before retrying. Its progress log stays ignored; a small disk-use
receipt is saved in `reports/container/build.json`.

The guarded build exports with `unpack=false`. Remove unused build cache before
the first start, as shown above, so Docker can unpack the image without retaining
a second uncompressed dependency tree. This clears disposable build artifacts;
it does not remove the app image, model directory or history volume. A normal
Compose build hit the guard during that temporary duplication on this machine.
The deferred export and separate unpack fit within the reserve.

For a later rebuild when less than 4 GiB is free, first remove only the replaceable
web container and its old image, then repeat the guarded build and startup steps:

```bash
sudo docker compose rm --stop --force web
sudo docker image rm visionguard:local
```

Keep PostgreSQL running during this operation. Its external volume and the
host-mounted champion files remain in place while the web image is rebuilt.

The app checks available space before processing or saving an upload and blocks
writes below 2 GiB plus 128 MiB. The container's overlay filesystem shares the
host disk, so its health route observes the same constrained storage. Uploads
stay in memory. Both containers limit their logs to two 5 MiB files. The web
container has a read-only root filesystem, a 128 MiB temporary memory mount,
a 2 GiB memory limit and a two-CPU limit. The database keeps its earlier bounds.

## Configuration and verification

`configure_web` preserves current local credentials and creates an encoded
`DATABASE_URL_CONTAINER` using the existing PostgreSQL credentials with host `db`.
The host's `DATABASE_URL` remains valid for localhost checks. Compose passes only
the application's required settings; MLflow and DVC credentials are not passed
to the web container. Secrets are not build arguments or image layers.

The app runs as UID/GID 10001, drops Linux capabilities and disables privilege
escalation. The existing upload validation, session-bound form token, debug-off
configuration, preprocessing and threshold checks remain in place.

```bash
python -m scripts.check_web --output-dir reports/container/application
python -m scripts.check_web --output-dir reports/container/application --verify-only
python -m scripts.check_container --sudo
```

The live application check uses two validation images and three rejected-upload
cases. It checks actual PostgreSQL rows and compares scores with the saved
validation predictions. It does not read held-out test images. The second command
verifies existing rows without submitting another upload, useful after restarts.
Container and persistence evidence is recorded under `reports/container/`.

References: [Compose volumes](https://docs.docker.com/reference/compose-file/volumes/),
[service mounts](https://docs.docker.com/reference/compose-file/services/#volumes)
[containerd disk use](https://docs.docker.com/engine/storage/containerd/)
and [deferred image unpacking](https://docs.docker.com/build/exporters/image-registry/).
