FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt requirements-embeddings.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Real semantic embeddings are optional (large download). Build with:
#   docker compose build --build-arg WITH_EMBEDDINGS=true
ARG WITH_EMBEDDINGS=false
RUN if [ "$WITH_EMBEDDINGS" = "true" ]; then pip install --no-cache-dir -r requirements-embeddings.txt; fi

COPY . .
EXPOSE 8001 8002
