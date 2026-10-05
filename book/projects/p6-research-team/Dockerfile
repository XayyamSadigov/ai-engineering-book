# path: book/projects/p6-research-team/Dockerfile
# Build from book/projects so the path dependencies and shared data are in the context:
#   docker build -f p6-research-team/Dockerfile -t northwind/research-team .
#   docker run --rm northwind/research-team ask "What do I need to do before travelling abroad with a company laptop?"
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY aie_core /app/aie_core
COPY agentkit /app/agentkit
COPY shared-data/shared_data.py /app/shared-data/shared_data.py
COPY shared-data/docs /app/shared-data/docs
COPY p6-research-team /app/p6-research-team
RUN pip install --no-cache-dir -e /app/aie_core -e /app/agentkit -e /app/p6-research-team
RUN useradd --create-home app && mkdir -p /app/runs && chown -R app /app/runs
USER app
WORKDIR /app/p6-research-team
ENV P6_SHARED_DATA_DIR=/app/shared-data P6_LOG_DIR=/app/runs P6_OFFLINE=true
ENTRYPOINT ["python", "-m", "research_team"]
CMD ["--help"]
