import torch
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from torch.utils import data

from tasks.utils import to_dataloaders


@to_dataloaders
def build_moons(
    n_train: int = 1000,
    n_val: int = 1000,
    train_only: bool = False,
    noise: float = 0.1,
    seed: int = 0,
    **kwargs
):
    n = n_train if train_only else n_train + n_val
    X, y = make_moons(n, noise=noise, random_state=seed)

    if train_only:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        train_dataset = data.TensorDataset(X, y)
        return train_dataset, train_dataset
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_val, random_state=seed)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.int64)
        train_dataset = data.TensorDataset(X_train, y_train)
        val_dataset = data.TensorDataset(X_val, y_val)
        return train_dataset, val_dataset


@to_dataloaders
def build_circle(
    n_train: int = 1000,
    n_val: int = 1000,
    train_only: bool = False,
    noise: float = 0.1,
    factor: float = 0.8,
    seed: int = 0,
    **kwargs
):
    n = n_train if train_only else n_train + n_val
    X, y = make_circles(n, noise=noise, random_state=seed, factor=factor)

    if train_only:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        train_dataset = data.TensorDataset(X, y)
        return train_dataset, train_dataset
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_val, random_state=seed)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        train_dataset = data.TensorDataset(X_train, y_train)
        val_dataset = data.TensorDataset(X_val, y_val)
        return train_dataset, val_dataset
