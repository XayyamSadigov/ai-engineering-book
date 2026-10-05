# path: book/projects/p2-semantic-search/Dockerfile
# Build from book/projects so the aie_core path dependency is in the context:
#   docker build -f p2-semantic-search/Dockerfile -t northwind/semsearch .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY aie_core /app/aie_core
COPY p2-semantic-search /app/p2-semantic-search
COPY shared-data/docs /app/shared-data/docs
COPY shared-data/eval /app/shared-data/eval
RUN pip install --no-cache-dir -e /app/aie_core \
 && pip install --no-cache-dir -e "/app/p2-semantic-search[pg]"
WORKDIR /app/p2-semantic-search
RUN useradd --create-home app && chown -R app /app/p2-semantic-search
USER app
EXPOSE 8000
CMD ["uvicorn", "semsearch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
