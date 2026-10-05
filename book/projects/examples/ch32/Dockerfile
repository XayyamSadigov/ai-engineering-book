# path: book/projects/examples/ch32/Dockerfile
# Build context: book/projects (so aie_core is available). In CI: docker build -f examples/ch32/Dockerfile ..
FROM python:3.12-slim
ARG TRIAGE_GIT_SHA=unknown
ARG TRIAGE_APP_VERSION=0.0.0-dev
# Code identity is baked into the image; artifact versions come from config and flags at runtime.
ENV TRIAGE_GIT_SHA=${TRIAGE_GIT_SHA} TRIAGE_APP_VERSION=${TRIAGE_APP_VERSION} PYTHONUNBUFFERED=1
WORKDIR /app
COPY aie_core /aie_core
COPY examples/ch32 /app
RUN pip install --no-cache-dir /aie_core . uvicorn
USER 10001
CMD ["uvicorn", "--factory", "northwind_triage.composition:create_app_from_env", "--host", "0.0.0.0", "--port", "8080"]
