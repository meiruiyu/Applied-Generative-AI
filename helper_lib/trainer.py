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