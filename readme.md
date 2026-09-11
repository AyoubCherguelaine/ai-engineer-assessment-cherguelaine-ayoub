# AI Engineer cinema Chatbot

A small FastAPI chatbot with one endpoint, `POST /ask`. It answers film questions from IMDb's official non-commercial datasets, superhero questions via [Superhero API](https://superheroapi.com/), or combines both. Google ADK runs the tool-using agent and its LiteLLM adapter calls Cerebras `gpt-oss-120b` by default. Every response includes its sources.

The API stays in `main.py`; the Google ADK coordinator exposes exactly two function tools: `search_imdb(query)` and `search_superhero(name)`. The small `agents/` folder contains those tools and shared contracts. Set `CEREBRAS_API_KEY`; the default model is `cerebras/gpt-oss-120b`. Optionally change `LITELLM_MODEL` to any LiteLLM-supported model identifier.

## Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add the Cerebras and Superhero API keys to .env
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API docs.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Who directed Inception?"}'
```

## Docker

```bash
cp .env.docker.example .env
# Add your API keys to .env

docker build -t cinema .
docker run -p 8000:8000 --env-file .env cinema
```

The first run builds the IMDb SQLite index inside the container (`data/imdb.db`). To keep the index between runs, mount a volume:

```bash
docker run -p 8000:8000 --env-file .env -v cinema-data:/app/data cinema
```

Open `http://127.0.0.1:8000/docs` for interactive API docs.

## Tests

```bash
pytest -q
```

## IMDb dataset setup

IMDb is the film dataset. Its public TSV archives are indexed locally because they are large; they are never downloaded during an API request.

```bash
python -m scripts.sync_imdb --max-movies 250000
```

Use `--max-movies 0` for the full movie index. Add `--all-datasets` to cache `principals` too; `names` is always used to resolve IMDb director IDs into readable names. The API returns a clear setup error for film questions until `data/imdb.db` exists.

## Design note

IMDb is the only film dataset, and Superhero API is the only other data source. The tradeoff is keyword search rather than semantic search; the ADK agent receives only matching rows and API data and is instructed to reject unsupported claims.
