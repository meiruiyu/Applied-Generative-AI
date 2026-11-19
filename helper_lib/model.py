import torch
import torch.nn as nn


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: [batch, 3, 64, 64]
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Shape: [batch, 16, 32, 32]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Shape: [batch, 32, 16, 16]
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(z.size(0), 128, 7, 7)
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.layers(images)


class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)


import math


class SinusoidalTimeEmbedding(nn.Module):
    """Encode diffusion timesteps using sinusoidal embeddings."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim, device=t.device, dtype=torch.float32
            )
            / half_dim
        )

        angles = t.float()[:, None] * frequencies[None, :]

        return torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )


class TimeConditionedBlock(nn.Module):
    """A convolutional block conditioned on diffusion timestep."""

    def __init__(self, in_channels, out_channels, time_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, time_emb):
        h = self.activation(self.norm1(self.conv1(x)))

        time_features = self.time_proj(time_emb)
        h = h + time_features[:, :, None, None]

        h = self.activation(self.norm2(self.conv2(h)))
        return h


class DiffusionUNet(nn.Module):
    """A compact time-conditioned U-Net for CIFAR-10 images."""

    def __init__(self, base_channels=32, time_dim=64):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.encoder1 = TimeConditionedBlock(
            3, base_channels, time_dim
        )
        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2,
            kernel_size=4, stride=2, padding=1
        )

        self.encoder2 = TimeConditionedBlock(
            base_channels * 2, base_channels * 2, time_dim
        )
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 4,
            kernel_size=4, stride=2, padding=1
        )

        self.bottleneck = TimeConditionedBlock(
            base_channels * 4, base_channels * 4, time_dim
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            kernel_size=4, stride=2, padding=1
        )
        self.decoder1 = TimeConditionedBlock(
            base_channels * 4, base_channels * 2, time_dim
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.decoder2 = TimeConditionedBlock(
            base_channels * 2, base_channels, time_dim
        )

        self.output = nn.Conv2d(
            base_channels, 3, kernel_size=1
        )

    def forward(self, x, t):
        time_emb = self.time_embedding(t)

        # Encoder: 32x32 -> 16x16 -> 8x8
        e1 = self.encoder1(x, time_emb)
        e2 = self.encoder2(self.down1(e1), time_emb)

        # Bottleneck: 8x8
        b = self.bottleneck(self.down2(e2), time_emb)

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = self.decoder1(torch.cat([d1, e2], dim=1), time_emb)

        d2 = self.up2(d1)
        d2 = self.decoder2(torch.cat([d2, e1], dim=1), time_emb)

        return self.output(d2)


class EnergyModel(nn.Module):
    """Energy-based model for 32x32 CIFAR-10 RGB images."""

    def __init__(self, base_channels=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.Conv2d(
                base_channels, base_channels * 2,
                kernel_size=4, stride=2, padding=1
            ),
            nn.SiLU(),

            nn.Conv2d(
                base_channels * 2, base_channels * 4,
                kernel_size=4, stride=2, padding=1
            ),
            nn.SiLU(),

            nn.Conv2d(
                base_channels * 4, base_channels * 4,
                kernel_size=3, padding=1
            ),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, x):
        energy = self.network(x)
        return energy.squeeze(-1)



def get_model(model_name):
    name = model_name.upper()

    if name == "CNN":
        return AssignmentCNN()

    if name == "GAN":
        return GAN()

    if name == "DIFFUSION":
        return DiffusionUNet()

    if name == "EBM":
        return EnergyModel()

    raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    model = AssignmentCNN()
    output = model(torch.randn(2, 3, 64, 64))

    print(model)
    print("Output shape:", output.shape)
    assert output.shape == (2, 10)