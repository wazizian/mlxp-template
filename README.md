# Deep Learning Template with PyTorch and MLXP

A flexible deep learning project template using PyTorch, Hydra, and MLXP for experiment tracking.

## Project Structure

```
.
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Base config
│   ├── cifar10.yaml           # CIFAR-10 specific config
│   ├── mnist.yaml             # MNIST specific config
│   ├── imagenette.yaml        # Imagenette specific config
│   └── mlxp.yaml              # MLXP logger config
├── tasks/                     # Task-specific implementations
│   ├── cifar10/              # CIFAR-10 dataset and models
│   ├── mnist/                # MNIST dataset and models
│   └── imagenet/             # ImageNet/Imagenette dataset and models
├── optimizers/               # Optimizer implementations
├── launcher.py              # Main training script
├── trainer.py               # Training loop implementation
└── requirements.txt         # Project dependencies
```

## Key Features

- **Hydra Configuration**: Uses Hydra for managing configurations and command-line overrides
- **MLXP Integration**: Experiment tracking and logging with MLXP
- **Modular Design**: Separate tasks, models, and training logic
- **Multiple Datasets**: Support for CIFAR-10, MNIST, and Imagenette
- **Flexible Optimization**: Customizable optimizers and averaging schemes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

To train a model using a specific configuration:

```bash
python launcher.py --config-name=mnist
```

### Configuration Overrides

Hydra allows overriding any configuration parameter from the command line:

```bash
# Change batch size and learning rate
python launcher.py --config-name=mnist dataset.batch_size=32 optimizer.learning_rate=0.01

# Use a different model architecture
python launcher.py --config-name=cifar10 model.depth=34

# Modify training parameters
python launcher.py --config-name=imagenette train.epochs=20 train.force_gpu=true
```

### Understanding Hydra's `utils.call`

The template uses `hydra.utils.call` to instantiate objects from configurations. For example:

```yaml
# In configs/mnist.yaml
model:
  _target_: tasks.mnist.models.build_net  # Function to call
  # Additional parameters would be passed as kwargs
```

This configuration is instantiated in `launcher.py` using:
```python
model = hydra.utils.call(cfg.model)
```

This pattern allows for:
- Dynamic object instantiation from config files
- Easy parameter modification via command line
- Clean separation of configuration and code

### Logging with MLXP

Training metrics are automatically logged using MLXP. Logs are stored in the `logs/` directory by default. The logging configuration can be modified in `configs/mlxp.yaml`.

## Adding New Tasks

1. Create a new directory under `tasks/`
2. Implement dataset loading in `data.py`
3. Implement model architecture in `models.py`
4. Create a corresponding config file in `configs/`

## Development

This project uses pre-commit hooks to maintain code quality. After installing the requirements, set up pre-commit:

```bash
pre-commit run -a
```

## Acknowledgments

- [MLXP](https://github.com/inria-thoth/mlxp) - Experiment tracking framework
- [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template) - Project structure inspiration
