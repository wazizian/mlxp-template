import pathlib

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tasks.utils import to_dataloaders


@to_dataloaders
def build_cifar10(
    data_storage: str,
    name: str,
    download: bool,
    n_train: int = 50000,
    n_val: int = 10000,
    batch_size: int = 256,
):
    pth = pathlib.Path.cwd() / data_storage / name
    pth.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        pth.as_posix(), train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(pth.as_posix(), train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:n_train]
    val_indices = torch.randperm(len(val_dataset))[:n_val]

    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)
    return train_dataset, val_dataset


@to_dataloaders
def build_dummy(num_classes: int = 10, batch_size: int = 256, **kwargs):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img_size = (3, 32, 32)
    train_dataset = datasets.FakeData(5 * batch_size, img_size, num_classes, transform=transform)
    val_dataset = datasets.FakeData(2 * batch_size, img_size, num_classes, transform=transform)
    return train_dataset, val_dataset
