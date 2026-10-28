from fastapi import FastAPI, HTTPException
import spacy
from pydantic import BaseModel
from app.bigram_model import BigramModel

from cnn_api import router as cnn_router

app = FastAPI()
app.include_router(cnn_router)

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


# Assignment 3: MNIST GAN image generation
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from fastapi.responses import Response
from pydantic import Field
from torchvision.utils import save_image

from helper_lib.model import Generator


GAN_WEIGHTS = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gan_generator.pth"
)


class GANGenerationRequest(BaseModel):
    num_samples: int = Field(default=10, ge=1, le=100)
    seed: int | None = Field(
        default=None,
        ge=0,
        le=4294967295,
    )


@lru_cache(maxsize=1)
def load_gan_generator():
    if not GAN_WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail="GAN weights missing: models/gan_generator.pth",
        )

    generator = Generator(latent_dim=100)
    generator.load_state_dict(
        torch.load(
            GAN_WEIGHTS,
            map_location="cpu",
            weights_only=True,
        )
    )
    generator.eval()
    return generator


@app.post(
    "/generate-digits",
    response_class=Response,
    responses={
        200: {
            "description": "A PNG grid of generated handwritten digits.",
            "content": {"image/png": {}},
        }
    },
)
def generate_digits(request: GANGenerationRequest):
    generator = load_gan_generator()

    # Use a separate random generator for each request.
    rng = torch.Generator(device="cpu")
    if request.seed is None:
        rng.seed()
    else:
        rng.manual_seed(request.seed)

    noise = torch.randn(
        request.num_samples,
        generator.latent_dim,
        generator=rng,
    )

    with torch.inference_mode():
        images = generator(noise)
        images = ((images + 1) / 2).clamp(0, 1)

    buffer = BytesIO()
    save_image(
        images,
        buffer,
        format="PNG",
        nrow=min(10, request.num_samples),
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition":
                'inline; filename="generated_digits.png"'
        },
    )