FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qwen-tts
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

EXPOSE 8091
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
