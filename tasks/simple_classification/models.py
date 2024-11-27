import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def build_MLP(
    input_size: int = 2,
    output_size: int = 2,
    hidden_sizes: list[int] = [32],
    activation: str = "gelu",
):
    return MLP(input_size, output_size, hidden_sizes, activation)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation):
        super().__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self.activation = getattr(F, activation)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
