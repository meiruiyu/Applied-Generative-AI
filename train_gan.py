import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from helper_lib.data_loader import get_mnist_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from helper_lib.utils import save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    torch.manual_seed(42)

    # Use CUDA if available; otherwise use CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    project_dir = Path(__file__).resolve().parent
    run_name = "gan_smoke" if args.smoke_test else "gan"
    model_dir = project_dir / "models"
    sample_dir = project_dir / "sample_images" / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Explicit dataset source: torchvision.datasets.MNIST.
    loader = get_mnist_data_loader(
        data_dir=str(project_dir / "data"),
        batch_size=args.batch_size,
        train=True,
        download=True,
    )

    if args.smoke_test:
        subset = Subset(loader.dataset, range(1024))
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

    print(f"Device: {device}", flush=True)
    print("Dataset source: torchvision.datasets.MNIST", flush=True)
    print(f"Training samples: {len(loader.dataset)}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)

    model = get_model("GAN").to(device)
    criterion = nn.BCELoss()

    optimizers = {
        "generator": torch.optim.Adam(
            model.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
        ),
        "discriminator": torch.optim.Adam(
            model.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
        ),
    }

    # Save untrained samples for comparison.
    generate_samples(
        model,
        device=device,
        num_samples=40,
        save_path=sample_dir / "epoch_000.png",
        show=False,
    )

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nTraining epoch {epoch}/{args.epochs}",
            flush=True,
        )

        model = train_gan(
            model=model,
            data_loader=loader,
            criterion=criterion,
            optimizer=optimizers,
            device=device,
            epochs=1,
        )

        # Save generator weights after every completed epoch.
        save_model(
            model.generator,
            model_dir / f"{run_name}_generator.pth",
        )

        # Save a checkpoint, including optimizer states.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "generator_optimizer_state_dict":
                    optimizers["generator"].state_dict(),
                "discriminator_optimizer_state_dict":
                    optimizers["discriminator"].state_dict(),
                "latent_dim": model.generator.latent_dim,
                "dataset": "MNIST",
                "batch_size": args.batch_size,
            },
            model_dir / f"{run_name}_checkpoint.pth",
        )

        generate_samples(
            model,
            device=device,
            num_samples=40,
            save_path=sample_dir / f"epoch_{epoch:03d}.png",
            show=False,
        )

    print("\nTraining completed.", flush=True)
    print(
        f"Generator weights: "
        f"{model_dir / f'{run_name}_generator.pth'}",
        flush=True,
    )
    print(f"Generated samples: {sample_dir}", flush=True)


if __name__ == "__main__":
    main()