# path: book/projects/p1-extraction-api/Dockerfile
# Build from book/projects so the image can include aie_core and the sample data:
#   docker build -f p1-extraction-api/Dockerfile -t northwind/extraction-api .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# dependencies first, so source edits do not invalidate the layer
COPY aie_core /deps/aie_core
RUN pip install /deps/aie_core "fastapi>=0.110" "uvicorn>=0.27" "pydantic-settings>=2.0"

COPY p1-extraction-api/extraction_api /app/extraction_api
COPY shared-data /app/shared-data

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app /app/data
USER app

ENV LLM_PROVIDER=fake \
    EXTRACT_SHARED_DATA_DIR=/app/shared-data \
    EXTRACT_REVIEW_BACKEND=sqlite \
    EXTRACT_REVIEW_DB_PATH=/app/data/review.db \
    TRACE_SINK=jsonl \
    TRACE_PATH=/app/data/traces.jsonl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "extraction_api.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
