import torch


def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    """Train a classification model."""
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / total_samples:.4f}"
        )

    return model


def train_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    """Train a GAN using separate generator/discriminator optimizers."""
    model = model.to(device)
    generator = model.generator
    discriminator = model.discriminator

    optimizer_g = optimizer["generator"]
    optimizer_d = optimizer["discriminator"]

    for epoch in range(epochs):
        model.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_samples = 0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 1. Update the discriminator.
            optimizer_d.zero_grad(set_to_none=True)

            z = torch.randn(
                batch_size, generator.latent_dim, device=device
            )
            fake_images = generator(z)

            real_loss = criterion(
                discriminator(real_images), real_labels
            )
            fake_loss = criterion(
                discriminator(fake_images.detach()), fake_labels
            )
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            # 2. Update the generator.
            # Keep gradient flow through D to G, but do not update D.
            discriminator.requires_grad_(False)
            discriminator.train()
            optimizer_g.zero_grad(set_to_none=True)

            z = torch.randn(
                batch_size, generator.latent_dim, device=device
            )
            generated_images = generator(z)

            g_loss = criterion(
                discriminator(generated_images), real_labels
            )
            g_loss.backward()
            optimizer_g.step()

            discriminator.requires_grad_(True)
            discriminator.train()

            total_g_loss += g_loss.item() * batch_size
            total_d_loss += d_loss.item() * batch_size
            total_samples += batch_size

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"D loss: {total_d_loss / total_samples:.4f} | "
            f"G loss: {total_g_loss / total_samples:.4f}"
        )

    return model



class DiffusionSchedule:
    """Linear beta schedule for a DDPM diffusion model."""

    def __init__(
        self,
        timesteps=200,
        beta_start=0.0001,
        beta_end=0.02,
        device="cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps,
            device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_images, noise, t):
        """Sample x_t from q(x_t | x_0)."""
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        return (
            torch.sqrt(alpha_bar) * clean_images
            + torch.sqrt(1.0 - alpha_bar) * noise
        )


def train_diffusion(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs=10,
    timesteps=200,
    max_batches=None,
):
    """Train a DDPM noise-prediction model on CIFAR-10."""
    import torch.nn.functional as F

    model = model.to(device)
    schedule = DiffusionSchedule(
        timesteps=timesteps, device=device
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (images, _) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            batch_size = images.size(0)

            t = torch.randint(
                0, timesteps, (batch_size,), device=device
            )
            noise = torch.randn_like(images)
            noisy_images = schedule.add_noise(
                images, noise, t
            )

            predicted_noise = model(noisy_images, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("No training batches were processed.")

        print(
            f"Diffusion Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / total_samples:.4f}",
            flush=True,
        )

    return model



def langevin_sampling(
    model,
    initial_images,
    steps=20,
    step_size=0.1,
    noise_scale=0.01,
):
    """Sample lower-energy images using Langevin dynamics."""
    if steps < 1:
        raise ValueError("steps must be at least 1")

    # Only the input images need gradients during sampling.
    images = initial_images.detach().clone()

    for _ in range(steps):
        images = images.detach().requires_grad_(True)

        energy = model(images).sum()

        # Compute gradients with respect to images, not weights.
        image_grad = torch.autograd.grad(
            outputs=energy,
            inputs=images,
        )[0]

        with torch.no_grad():
            images = (
                images
                - step_size * image_grad
                + noise_scale * torch.randn_like(images)
            )
            images.clamp_(-1.0, 1.0)

    return images.detach()




def train_ebm(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs=10,
    langevin_steps=20,
    step_size=0.1,
    noise_scale=0.01,
    energy_reg=0.001,
    max_batches=None,
):
    """Train an energy-based model using contrastive learning."""
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_real_energy = 0.0
        total_fake_energy = 0.0
        total_samples = 0

        for batch_idx, (real_images, _) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Generate negative samples using Langevin dynamics.
            initial_images = (
                torch.rand_like(real_images) * 2.0 - 1.0
            )

            fake_images = langevin_sampling(
                model,
                initial_images,
                steps=langevin_steps,
                step_size=step_size,
                noise_scale=noise_scale,
            )

            # Stop gradients through the negative sampling process.
            fake_images = fake_images.detach()

            # Energy of real and generated images.
            real_energy = model(real_images)
            fake_energy = model(fake_images)

            # Encourage low energy for real images
            # and higher energy for negative samples.
            contrastive_loss = (
                real_energy.mean() - fake_energy.mean()
            )

            # Regularization prevents energy values from
            # growing without bound.
            regularization = energy_reg * (
                real_energy.square().mean()
                + fake_energy.square().mean()
            )

            loss = contrastive_loss + regularization

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_real_energy += real_energy.mean().item() * batch_size
            total_fake_energy += fake_energy.mean().item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("No training batches were processed.")

        print(
            f"EBM Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / total_samples:.4f} | "
            f"Real Energy: {total_real_energy / total_samples:.4f} | "
            f"Fake Energy: {total_fake_energy / total_samples:.4f}",
            flush=True,
        )

    return model
