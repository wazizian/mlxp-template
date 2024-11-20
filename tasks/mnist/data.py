import pathlib

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tasks.utils import to_dataloaders


@to_dataloaders
def build_mnist(
    data_storage: str,
    name: str,
    download: bool,
    n_train: int = 60000,
    n_val: int = 10000,
    batch_size: int = 256,
):
    pth = pathlib.Path.cwd() / data_storage / name
    pth.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(pth.as_posix(), train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(pth.as_posix(), train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:n_train]
    val_indices = torch.randperm(len(val_dataset))[:n_val]

    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)

    return train_dataset, val_dataset
