import torchvision
import torchvision.models as models


def build_resnet(classes: int, depth: int = 18):
    if depth == 18:
        return models.resnet18(num_classes=classes)
    elif depth == 34:
        return models.resnet34(num_classes=classes)
    elif depth == 50:
        return models.resnet50(num_classes=classes)
    elif depth == 101:
        return models.resnet101(num_classes=classes)
    elif depth == 152:
        return models.resnet152(num_classes=classes)
    else:
        raise ValueError(f"Unsupported depth {depth}")
