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