import os

import hydra
import mlxp
import rootutils
import torch

import trainer
import optimizers.utils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def set_seeds(seed):
    import torch

    torch.manual_seed(seed)


@mlxp.launch(config_path="./configs", seeding_function=set_seeds)
def main(ctx):
    cfg = ctx.config

    # Device
    print(f"Preparing device (force GPU: {cfg.train.force_gpu})...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    if cfg.train.force_gpu and device != "cuda":
        raise ValueError("CUDA is not available")

    # Dataset
    print(f"Preparing dataset ({cfg.dataset._target_})...")
    dataloaders = hydra.utils.call(cfg.dataset)

    # Model
    print(f"Preparing model ({cfg.model._target_})...")
    model = hydra.utils.call(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    # Optimizer
    print(f"Preparing optimizer ({cfg.optimizer._target_})...")
    optimizer_builder = hydra.utils.call(cfg.optimizer)
    optimizer = optimizer_builder(model.parameters())
    optimizer = optimizers.utils.with_extra_args_support(optimizer)

    # Averager
    print(f"Preparing averager ({cfg.averager._target_})...")
    averager_builder = hydra.utils.call(cfg.averager)
    avg_model = averager_builder(model)

    # Loss
    print(f"Preparing loss ({cfg.loss._target_})...")
    criterion = hydra.utils.call(cfg.loss)

    # Train
    trainer.main(ctx, device, dataloaders, model, avg_model, optimizer, criterion)


if __name__ == "__main__":
    main()
