FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
CMD ["python", "scripts/smoke_test.py"]
