import torch
from torch import nn
from torchvision import models

from .data import CLASSES


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes=len(CLASSES), pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(path, model, epoch, metrics, settings):
    torch.save({"model": model.state_dict(), "classes": list(CLASSES), "epoch": epoch, "metrics": metrics,
                "settings": settings}, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
