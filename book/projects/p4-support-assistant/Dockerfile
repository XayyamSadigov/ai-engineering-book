# path: book/projects/p4-support-assistant/Dockerfile
# Build from book/projects so the path dependencies and shared data are in the context:
#   docker build -f p4-support-assistant/Dockerfile -t northwind/support-assistant .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY aie_core /app/aie_core
COPY toolkit /app/toolkit
COPY shared-data/shared_data.py shared-data/tickets.jsonl /app/shared-data/
COPY p4-support-assistant /app/p4-support-assistant
RUN pip install --no-cache-dir -e /app/aie_core -e /app/toolkit -e /app/p4-support-assistant
WORKDIR /app/p4-support-assistant
RUN useradd --create-home app && mkdir -p /app/state && chown -R app /app/state
USER app
ENV P4_SHARED_DATA_DIR=/app/shared-data \
    P4_IDEMPOTENCY_DB=/app/state/idempotency.db \
    P4_AUDIT_LOG_PATH=/app/state/audit.jsonl
EXPOSE 8000
CMD ["uvicorn", "support_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
