FROM python:3.12-slim

ARG MODEL_NAME=jinaai/jina-reranker-v2-base-multilingual
ARG CACHE_DIR=/app/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    MODEL_NAME=${MODEL_NAME} \
    CACHE_DIR=${CACHE_DIR}

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . /app/
RUN mkdir -p "${CACHE_DIR}" \
    && chown -R app:app /app "${CACHE_DIR}"

USER app

# Pre-download the model into CACHE_DIR during build
#RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; import os; TextCrossEncoder(model_name=os.getenv('MODEL_NAME'), cache_dir=os.getenv('CACHE_DIR'))"

EXPOSE 8000

CMD ["python", "main.py"]
