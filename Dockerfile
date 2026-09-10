FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

EXPOSE 8000

ENV HTTP_TIMEOUT_SECONDS=15 \
    SUPERHERO_TIMEOUT_SECONDS=30 \
    CEREBRAS_MODEL=gpt-oss-120b \
    LITELLM_MODEL=cerebras/gpt-oss-120b

ENTRYPOINT ["sh", "-c", "\
    if [ ! -f data/imdb.db ]; then \
        echo 'Building IMDb SQLite index (first run, ~5-10 min)...'; \
        python -m scripts.sync_imdb --max-movies 250000; \
    fi && \
    exec uvicorn main:app --host 0.0.0.0 --port 8000 \
"]
