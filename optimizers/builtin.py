from functools import partial

import torch
import torch.optim as optim


def build_sgd(learning_rate: float):
    return partial(optim.SGD, lr=learning_rate)


def build_adam(learning_rate: float):
    return partial(optim.Adam, lr=learning_rate)
