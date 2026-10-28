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


def get_model(model_name):
    name = model_name.upper()

    if name == "CNN":
        return AssignmentCNN()
    if name == "GAN":
        return GAN()

    raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    model = AssignmentCNN()
    output = model(torch.randn(2, 3, 64, 64))

    print(model)
    print("Output shape:", output.shape)
    assert output.shape == (2, 10)