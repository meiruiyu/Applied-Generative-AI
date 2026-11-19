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




# Assignment 4: Diffusion and Energy-Based image generation

from helper_lib.model import get_model
from helper_lib.generator import (
    generate_diffusion_images,
    generate_ebm_images,
)


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

DIFFUSION_WEIGHTS = MODEL_DIR / "diffusion_cifar10.pth"
EBM_WEIGHTS = MODEL_DIR / "ebm_cifar10.pth"


class DiffusionGenerationRequest(BaseModel):
    num_samples: int = Field(default=2, ge=1, le=4)


class EBMGenerationRequest(BaseModel):
    num_samples: int = Field(default=2, ge=1, le=4)
    steps: int = Field(default=10, ge=1, le=100)
    step_size: float = Field(default=0.1, gt=0, le=1.0)
    noise_scale: float = Field(default=0.01, ge=0, le=0.1)


@lru_cache(maxsize=1)
def load_diffusion_model():
    if not DIFFUSION_WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail="Diffusion model weights are missing.",
        )

    checkpoint = torch.load(
        DIFFUSION_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )

    model = get_model("DIFFUSION")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["timesteps"]


@lru_cache(maxsize=1)
def load_ebm_model():
    if not EBM_WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail="EBM model weights are missing.",
        )

    checkpoint = torch.load(
        EBM_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )

    model = get_model("EBM")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def images_to_png_response(images, filename):
    """Convert generated image tensors to a PNG grid."""
    images = ((images.detach().cpu() + 1) / 2).clamp(0, 1)

    buffer = BytesIO()

    save_image(
        images,
        buffer,
        format="PNG",
        nrow=min(4, images.size(0)),
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition":
                f'inline; filename="{filename}"'
        },
    )


@app.post(
    "/generate-diffusion",
    response_class=Response,
    responses={
        200: {
            "description": "Generated CIFAR-10 diffusion images.",
            "content": {"image/png": {}},
        }
    },
)
def generate_diffusion_endpoint(
    request: DiffusionGenerationRequest,
):
    model, timesteps = load_diffusion_model()

    images = generate_diffusion_images(
        model=model,
        num_samples=request.num_samples,
        timesteps=timesteps,
        device="cpu",
    )

    return images_to_png_response(
        images,
        "generated_diffusion.png",
    )


@app.post(
    "/generate-ebm",
    response_class=Response,
    responses={
        200: {
            "description": "Generated CIFAR-10 energy-based images.",
            "content": {"image/png": {}},
        }
    },
)
def generate_ebm_endpoint(
    request: EBMGenerationRequest,
):
    model = load_ebm_model()

    # Langevin sampling needs gradients for the input images.
    images = generate_ebm_images(
        model=model,
        num_samples=request.num_samples,
        steps=request.steps,
        step_size=request.step_size,
        noise_scale=request.noise_scale,
        device="cpu",
    )

    return images_to_png_response(
        images,
        "generated_ebm.png",
    )
