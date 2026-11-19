from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def generate_samples(
    model,
    device="cpu",
    num_samples=10,
    save_path=None,
    show=True,
):
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    generator = model.generator.to(device)
    was_training = generator.training
    generator.eval()

    with torch.no_grad():
        z = torch.randn(
            num_samples,
            generator.latent_dim,
            device=device,
        )
        images = generator(z).cpu()

    generator.train(was_training)

    # Convert Tanh output from [-1, 1] to [0, 1].
    images = ((images + 1) / 2).clamp(0, 1)
    nrow = min(10, num_samples)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(images, path, nrow=nrow)

    if show:
        import matplotlib.pyplot as plt

        grid = make_grid(images, nrow=nrow)
        plt.figure(figsize=(10, 3))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return images



# Assignment 4: Diffusion and Energy-Based image generation

import torch

from helper_lib.trainer import (
    DiffusionSchedule,
    langevin_sampling,
)


@torch.no_grad()
def generate_diffusion_images(
    model,
    num_samples=4,
    timesteps=200,
    device="cpu",
):
    """Generate CIFAR-10 images using DDPM reverse diffusion."""
    model = model.to(device)
    model.eval()

    schedule = DiffusionSchedule(
        timesteps=timesteps, device=device
    )

    x = torch.randn(num_samples, 3, 32, 32, device=device)

    for step in reversed(range(timesteps)):
        t = torch.full(
            (num_samples,), step,
            dtype=torch.long, device=device
        )

        predicted_noise = model(x, t)

        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        beta = schedule.betas[step]

        mean = (
            x - beta / torch.sqrt(1 - alpha_bar)
            * predicted_noise
        ) / torch.sqrt(alpha)

        if step > 0:
            previous_alpha_bar = schedule.alpha_bars[step - 1]
            posterior_variance = (
                beta * (1 - previous_alpha_bar)
                / (1 - alpha_bar)
            )
            x = mean + torch.sqrt(
                posterior_variance
            ) * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1, 1).detach()


def generate_ebm_images(
    model,
    num_samples=4,
    steps=100,
    step_size=0.1,
    noise_scale=0.01,
    device="cpu",
):
    """Generate CIFAR-10 images using Langevin sampling."""
    model = model.to(device)
    model.eval()

    initial_images = (
        torch.rand(
            num_samples, 3, 32, 32, device=device
        ) * 2 - 1
    )

    # EBM sampling requires gradients with respect to images.
    with torch.enable_grad():
        samples = langevin_sampling(
            model,
            initial_images,
            steps=steps,
            step_size=step_size,
            noise_scale=noise_scale,
        )

    return samples.clamp(-1, 1).detach()
