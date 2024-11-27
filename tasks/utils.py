import torch
import torch.utils.data as data


def to_dataloaders(func):
    def new_func(
        *args,
        batch_size: int = 256,
        val_batch_size: int = None,
        pin_memory: bool = False,
        num_workers: int = -1,
        **kwargs
    ):
        if val_batch_size is None:
            val_batch_size = batch_size
        new_kwargs = {"batch_size": batch_size, **kwargs}
        train_dataset, val_dataset = func(*args, **new_kwargs)
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_dataloader, val_dataloader

    return new_func
