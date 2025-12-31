FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV POETRY_VERSION=2.0.1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PIP_NO_CACHE_DIR=1

ENV NO_PROXY=127.0.0.1,localhost
ENV no_proxy=127.0.0.1,localhost
ENV MLFLOW_ENABLE_ASYNC_LOGGING=false
ENV MLFLOW_HTTP_REQUEST_TIMEOUT=5

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY src ./src
COPY train.py .
COPY configs ./configs

CMD ["python", "train.py"]
