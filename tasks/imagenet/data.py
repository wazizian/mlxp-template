import pathlib

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tasks.utils import to_dataloaders


@to_dataloaders
def build_imagenette(
    data_storage: str,
    name: str,
    download: bool,
    batch_size: int = 256,
):
    pth = pathlib.Path.cwd() / data_storage / name
    pth.mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.Imagenette(
        pth.as_posix(), split="train", download=False, transform=train_transform
    )
    val_dataset = datasets.Imagenette(
        pth.as_posix(), split="val", download=False, transform=val_transform
    )
    return train_dataset, val_dataset


@to_dataloaders
def build_dummy(num_classes: int = 10, batch_size: int = 256, **kwargs):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.FakeData(
        5 * batch_size, (3, 224, 224), num_classes, transform=transform
    )
    val_dataset = datasets.FakeData(
        2 * batch_size, (3, 224, 224), num_classes, transform=transform
    )
    return train_dataset, val_dataset
