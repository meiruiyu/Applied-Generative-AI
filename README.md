# Assignment 1: FastAPI and Word Embeddings

This project provides a bigram text generator and a spaCy word embedding API.

## Run with Docker

docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai

Open http://127.0.0.1:8000/docs to query the API.

- GET /embedding?word=apple returns a 300-dimensional word vector.
- POST /generate accepts {"start_word": "the", "length": 10}.