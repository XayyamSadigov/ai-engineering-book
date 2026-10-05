# path: book/projects/examples/ch28/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api_skeleton.py .
EXPOSE 8000
CMD ["python", "api_skeleton.py"]
