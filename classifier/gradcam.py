import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            layer.register_forward_hook(self._save_activations),
            layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image, class_index=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if class_index is None:
            class_index = int(logits.argmax(dim=1))
        logits[0, class_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), class_index, probabilities

    def close(self):
        for handle in self.handles:
            handle.remove()


def overlay(image, cam, alpha=0.45):
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    resized = F.interpolate(torch.from_numpy(cam)[None, None], size=rgb.shape[:2], mode="bilinear",
                            align_corners=False)[0, 0].numpy()
    heat = colormaps["jet"](np.clip(resized, 0, 1))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)
