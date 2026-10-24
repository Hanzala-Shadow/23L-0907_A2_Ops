FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-create-home --uid 10001 --user-group --shell /usr/sbin/nologin app \
    && mkdir -p models/champion

COPY requirements/serving.lock requirements/serving.lock
# Downloads stay in RAM. C++ development headers are not needed for inference.
RUN --mount=type=tmpfs,target=/tmp \
    python -m pip install --no-cache-dir --no-compile --only-binary=:all: -r requirements/serving.lock \
    && rm -rf /usr/local/lib/python3.12/site-packages/tensorflow/include \
    && python -m pip check

COPY --chown=10001:10001 visionguard/ visionguard/
COPY --chown=10001:10001 wsgi.py gunicorn.conf.py ./
COPY --chown=10001:10001 docker/healthcheck.py ./healthcheck.py

ENV HOME=/tmp \
    KERAS_HOME=/tmp/keras \
    APP_BIND=0.0.0.0:8000
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=45s --retries=6 \
    CMD ["python", "healthcheck.py"]
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]
