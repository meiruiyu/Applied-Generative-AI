
"""Train a DDPM diffusion model on CIFAR-10."""

import argparse
from pathlib import Path

import torch

from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.data_loader import get_cifar10_generation_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="models/diffusion_cifar10.pth",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Training device:", device, flush=True)

    loader = get_cifar10_generation_loader(
        batch_size=args.batch_size,
        train=True,
        download=False,
    )

    print("Dataset size:", len(loader.dataset), flush=True)

    model = get_model("DIFFUSION").to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    model = train_diffusion(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        timesteps=args.timesteps,
        max_batches=args.max_batches,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "timesteps": args.timesteps,
            "epochs": args.epochs,
            "max_batches": args.max_batches,
            "batch_size": args.batch_size,
        },
        output_path,
    )

    print(f"Model saved to: {output_path}", flush=True)
    print("Diffusion training completed!", flush=True)


if __name__ == "__main__":
    main()
