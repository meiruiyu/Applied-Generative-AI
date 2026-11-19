
"""Train an Energy-Based Model on CIFAR-10."""

import argparse
from pathlib import Path

import torch

from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.data_loader import get_cifar10_generation_loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--langevin-steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--energy-reg", type=float, default=0.001)
    parser.add_argument("--max-batches", type=int, default=None)

    parser.add_argument(
        "--output",
        type=str,
        default="models/ebm_cifar10.pth",
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

    model = get_model("EBM").to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    model = train_ebm(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        langevin_steps=args.langevin_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale,
        energy_reg=args.energy_reg,
        max_batches=args.max_batches,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "langevin_steps": args.langevin_steps,
            "step_size": args.step_size,
            "noise_scale": args.noise_scale,
            "energy_reg": args.energy_reg,
            "max_batches": args.max_batches,
        },
        output_path,
    )

    print(f"Model saved to: {output_path}", flush=True)
    print("EBM training completed!", flush=True)


if __name__ == "__main__":
    main()
