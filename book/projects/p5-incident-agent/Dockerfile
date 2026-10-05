# path: book/projects/p5-incident-agent/Dockerfile
# Build from book/projects so the path dependencies and shared data are in the context:
#   docker build -f p5-incident-agent/Dockerfile -t northwind/incident-agent .
#   docker run -p 8000:8000 -v p5-state:/app/state northwind/incident-agent
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY aie_core /app/aie_core
COPY agentkit /app/agentkit
COPY ragkit /app/ragkit
COPY evalkit /app/evalkit
COPY shared-data/docs /app/shared-data/docs
COPY p5-incident-agent /app/p5-incident-agent
RUN pip install --no-cache-dir -e /app/aie_core -e /app/agentkit -e /app/ragkit -e /app/evalkit \
    -e /app/p5-incident-agent
WORKDIR /app/p5-incident-agent
RUN useradd --create-home app && mkdir -p /app/state && chown -R app /app/state
USER app
ENV P5_SHARED_DATA_DIR=/app/shared-data \
    P5_STATE_DIR=/app/state \
    LLM_PROVIDER=fake
EXPOSE 8000
# One worker: approvals resume runs from files under /app/state; scale out with a shared store (Chapter 38).
CMD ["uvicorn", "incident_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
