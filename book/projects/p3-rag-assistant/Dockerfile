# path: book/projects/p3-rag-assistant/Dockerfile
# Build from book/projects so the path dependencies and shared data are in the context:
#   docker build -f p3-rag-assistant/Dockerfile -t northwind/rag-assistant .
# One image, two roles: the API (default CMD) and the ingestion worker (compose overrides CMD).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY aie_core /app/aie_core
COPY evalkit /app/evalkit
COPY p2-semantic-search /app/p2-semantic-search
COPY ragkit /app/ragkit
COPY guardrails /app/guardrails
COPY reliability /app/reliability
COPY shared-data/docs /app/shared-data/docs
COPY shared-data/eval /app/shared-data/eval
COPY p3-rag-assistant /app/p3-rag-assistant
RUN pip install --no-cache-dir -e /app/aie_core -e /app/evalkit -e "/app/p2-semantic-search[pg]" \
      -e /app/ragkit -e /app/guardrails -e /app/reliability \
 && pip install --no-cache-dir -e "/app/p3-rag-assistant[pg,redis]"
WORKDIR /app/p3-rag-assistant
RUN useradd --create-home app && mkdir -p /data && chown -R app /data
USER app
ENV RAG_DOCS_DIR=/app/shared-data/docs \
    RAG_BLOB_DIR=/data/blobs \
    RAG_SNAPSHOT_DIR=/data/snapshots
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "rag_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
