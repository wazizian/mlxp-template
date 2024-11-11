import torch
import torch.utils.data as data

from tasks.utils import to_dataloaders

@to_dataloaders
def build_linear(
        d = 1000,
        n_train = 900,
        n_val = 250,
        sigma_norm = 1.5
        ):
    opt = torch.ones(d)
    x_train = torch.randn(n_train, d)
    sigma = sigma_norm / torch.sqrt(d)
    y_train = torch.matmul(x_train, opt) + sigma * torch.randn(n_train)
    x_val = torch.randn(n_val, d)
    y_val = torch.matmul(x_val, opt) + sigma * torch.randn(n_val)
    train_dataset = data.TensorDataset(x_train, y_train)
    val_dataset = data.TensorDataset(x_val, y_val)
    return train_dataset, val_dataset
