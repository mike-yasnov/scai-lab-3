import open_clip
import torch

from .data import CLASSES

CLASS_PHRASES = {
    "buildings": "buildings",
    "forest": "a forest",
    "glacier": "a glacier",
    "mountain": "mountains",
    "sea": "the sea",
    "street": "a city street",
}
SINGLE_TEMPLATE = ("a photo of {}.",)
ENSEMBLE_TEMPLATES = (
    "a photo of {}.",
    "a landscape photo of {}.",
    "a blurry photo of {}.",
    "a bright photo of {}.",
    "a low resolution photo of {}.",
    "a photo taken on a phone of {}.",
    "a picture of {}.",
)


class ClipZeroShot:
    def __init__(self, model_name, pretrained, device, templates=ENSEMBLE_TEMPLATES):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.set_templates(templates)

    @torch.no_grad()
    def set_templates(self, templates):
        features = []
        for name in CLASSES:
            tokens = self.tokenizer([t.format(CLASS_PHRASES[name]) for t in templates]).to(self.device)
            embedding = self.model.encode_text(tokens).float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            mean = embedding.mean(dim=0)
            features.append(mean / mean.norm())
        self.text_features = torch.stack(features)

    @torch.no_grad()
    def image_features(self, images):
        features = self.model.encode_image(images.to(self.device)).float()
        return features / features.norm(dim=-1, keepdim=True)

    def classify(self, features):
        return 100.0 * features.to(self.device) @ self.text_features.T

    def logits(self, images):
        return self.classify(self.image_features(images))
