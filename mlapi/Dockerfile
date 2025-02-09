#Copy Slimmed Version from lab 1 and add slim

FROM python:3.11-slim AS build
ARG APP_DIR=/app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3 
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"


WORKDIR ${APP_DIR}
COPY pyproject.toml poetry.lock ./

RUN python -m venv --copies ${APP_DIR}/venv
RUN . ${APP_DIR}/venv/bin/activate && poetry install --only main

#Deployment Run
FROM python:3.11-slim AS run
ARG APP_DIR=/app

ENV PATH ${APP_DIR}/venv/bin:$PATH

COPY --from=build ${APP_DIR}/venv ${APP_DIR}/venv/
COPY distilbert-base-uncased-finetuned-sst2/ ./distilbert-base-uncased-finetuned-sst2/
COPY src/ ./src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/project/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
