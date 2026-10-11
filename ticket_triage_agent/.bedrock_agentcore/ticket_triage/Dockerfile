FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data

RUN pip install --no-cache-dir . "bedrock-agentcore>=1.0.0"

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "scripts/agentcore_app.py"]
