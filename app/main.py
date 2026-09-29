from fastapi import FastAPI, HTTPException
import spacy
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI()
nlp = spacy.load("en_core_web_lg")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word, request.length
    )
    return {"generated_text": generated_text}

@app.get("/embedding")
def get_embedding(word: str):
    word = word.strip()

    if not word or len(word.split()) != 1:
        raise HTTPException(status_code=400, detail="Please provide one word.")

    doc = nlp(word)
    if len(doc) != 1 or not doc[0].has_vector:
        raise HTTPException(status_code=404, detail="Word vector not found.")

    vector = doc[0].vector.tolist()
    return {"word": word, "embedding": vector, "dimensions": len(vector)}