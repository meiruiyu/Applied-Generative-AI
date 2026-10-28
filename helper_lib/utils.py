from pathlib import Path
import torch


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")