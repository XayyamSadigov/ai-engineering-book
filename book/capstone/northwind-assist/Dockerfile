# path: book/capstone/northwind-assist/Dockerfile
# Build from the book root so every path dependency and the shared data are in the context:
#   docker build -f capstone/northwind-assist/Dockerfile -t northwind/assist:1.0.0 book
# (`--build-arg BASE_IMAGE=<mirror>/python:3.12` when Docker Hub is not reachable.)
# One image, three roles: API (default CMD), ingestion worker, nightly eval (compose overrides CMD).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS base
ARG NA_GIT_SHA=local
ENV NA_GIT_SHA=${NA_GIT_SHA}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NA_BOOK_ROOT=/book
WORKDIR /book
# Libraries first (they change less often than the capstone), so layers cache well.
COPY projects/aie_core projects/aie_core
COPY projects/evalkit projects/evalkit
COPY projects/p2-semantic-search projects/p2-semantic-search
COPY projects/ragkit projects/ragkit
COPY projects/toolkit projects/toolkit
COPY projects/agentkit projects/agentkit
COPY projects/memorykit projects/memorykit
COPY projects/guardrails projects/guardrails
COPY projects/reliability projects/reliability
COPY projects/p1-extraction-api projects/p1-extraction-api
COPY projects/p3-rag-assistant projects/p3-rag-assistant
COPY projects/p4-support-assistant projects/p4-support-assistant
COPY projects/shared-data projects/shared-data
COPY projects/examples/ch04 projects/examples/ch04
COPY projects/examples/ch05 projects/examples/ch05
COPY projects/examples/ch07 projects/examples/ch07
COPY projects/examples/ch25 projects/examples/ch25
COPY projects/examples/ch26 projects/examples/ch26
COPY projects/examples/ch30 projects/examples/ch30
COPY projects/examples/ch31 projects/examples/ch31
COPY projects/examples/ch32 projects/examples/ch32
# The book's packages and their declared dependencies resolve together in one install (path
# dependencies such as aie-core are satisfied by the editable copies in the same command).
RUN pip install -e projects/aie_core -e projects/evalkit -e "projects/p2-semantic-search[pg]" -e projects/ragkit \
      -e projects/toolkit -e projects/agentkit -e projects/memorykit -e projects/guardrails \
      -e "projects/reliability[redis]" -e projects/p1-extraction-api -e "projects/p3-rag-assistant[pg,redis]" \
      -e projects/p4-support-assistant \
      "pyjwt[crypto]>=2.8" "jinja2>=3.1" "opentelemetry-sdk>=1.24" "opentelemetry-exporter-otlp-proto-http>=1.24"
COPY capstone/northwind-assist capstone/northwind-assist
RUN pip install -e capstone/northwind-assist
WORKDIR /book/capstone/northwind-assist
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /data
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "--factory", "northwind_assist.api.app:build_app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "75"]
