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