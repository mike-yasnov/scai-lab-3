import numpy as np
import torch
from PIL import Image

from classifier.gradcam import GradCAM, overlay
from classifier.model import build_model


def test_gradcam_map_matches_layer4_grid():
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    cam = GradCAM(model, model.layer4)
    heat, index, probabilities = cam(torch.randn(3, 224, 224))
    assert heat.shape == (7, 7) and 0 <= heat.min() and heat.max() <= 1
    assert index == int(probabilities.argmax()) and np.isclose(probabilities.sum(), 1, atol=1e-5)
    other = (index + 1) % 6
    assert cam(torch.randn(3, 224, 224), other)[1] == other
    cam.close()
    assert not model.layer4._forward_hooks and not model.layer4._backward_hooks


def test_overlay_keeps_image_size():
    image = Image.new("RGB", (224, 224), "white")
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[3, 3] = 1
    result = overlay(image, cam)
    assert result.shape == (224, 224, 3) and 0 <= result.min() and result.max() <= 1
    assert result[112, 112, 0] != result[0, 0, 0]
