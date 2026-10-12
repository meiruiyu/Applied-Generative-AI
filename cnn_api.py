from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from helper_lib.model import AssignmentCNN
from helper_lib.data_loader import get_transform


router = APIRouter()

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

WEIGHTS = (
    Path(__file__).resolve().parent
    / "models"
    / "cnn_cifar10.pth"
)

transform = get_transform()


@lru_cache(maxsize=1)
def get_model():
    if not WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model weights missing.",
        )

    model = AssignmentCNN()
    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model


@router.post("/classify-image")
def classify_image(file: UploadFile = File(...)):
    try:
        with Image.open(file.file) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image.",
        )

    model = get_model()

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    class_id = int(probabilities.argmax().item())

    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "confidence": float(probabilities[class_id].item()),
    }