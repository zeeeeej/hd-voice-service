FROM python:3.11-slim-bookworm

ARG PIP_INDEX_URL=

WORKDIR /opt/gateway

COPY requirements.txt ./
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
