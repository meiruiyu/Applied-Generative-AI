# Assignment 1: FastAPI and Word Embeddings

This project extends the Module 3 bigram text generation API with a spaCy word embedding endpoint. It uses the `en_core_web_lg` model.

## Run with Docker

From the project root, with Docker Desktop running:

```bash
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```

The first command builds the image and installs the Python dependencies and spaCy model. The second command starts the API. The first build may take several minutes.

Open http://127.0.0.1:8000/docs to test the endpoints.

## API endpoints

- `GET /embedding?word=apple`: returns the queried word, its 300-dimensional embedding vector, and the vector dimensions.
- `POST /generate`: generates text from the bigram model. Example request body:

```json
{"start_word": "the", "length": 10}
```

## Run locally without Docker

With Python and uv installed, run these commands from the project root:

```bash
uv sync
uv run fastapi dev app/main.py
```

Then open http://127.0.0.1:8000/docs.