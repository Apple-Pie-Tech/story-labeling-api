FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir uv \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY app ./app

RUN uv sync --no-dev \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import json, sys, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3); payload = json.load(response); sys.exit(0 if payload.get('status') == 'ok' else 1)"]

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
