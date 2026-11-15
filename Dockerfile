# syntax=docker/dockerfile:1
FROM python:3.12-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# libgomp1 — OpenMP-рантайм, без него LightGBM не импортируется в slim-образе
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
COPY artifacts/ artifacts/
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# Один воркер на контейнер: метрики Prometheus и окно дрейфа живут в памяти процесса,
# масштабируемся репликами (k8s/hpa.yaml), а не воркерами
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM api AS bot
USER root
COPY requirements-bot.txt .
RUN pip install -r requirements-bot.txt
COPY bot/ bot/
USER appuser
HEALTHCHECK NONE
CMD ["python", "-m", "bot.bot"]
