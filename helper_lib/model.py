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


if __name__ == "__main__":
    model = AssignmentCNN()
    output = model(torch.randn(2, 3, 64, 64))

    print(model)
    print("Output shape:", output.shape)
    assert output.shape == (2, 10)