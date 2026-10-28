from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


def get_data_loader(train=True):
    dataset = datasets.CIFAR10(
        root="data",
        train=train,
        download=False,
        transform=get_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=train,
        num_workers=0,
    )


def get_mnist_data_loader(
    data_dir="data",
    batch_size=64,
    train=True,
    download=True,
):
    """Load MNIST images normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
    )