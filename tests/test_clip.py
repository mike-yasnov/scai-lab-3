import torch

from classifier.clip_zeroshot import ENSEMBLE_TEMPLATES, SINGLE_TEMPLATE, ClipZeroShot
from classifier.data import CLASSES


def test_clip_text_features_and_logits():
    clip = ClipZeroShot("ViT-B-32", None, torch.device("cpu"), SINGLE_TEMPLATE)
    assert clip.text_features.shape == (len(CLASSES), 512)
    assert torch.allclose(clip.text_features.norm(dim=1), torch.ones(len(CLASSES)), atol=1e-5)
    single = clip.text_features.clone()
    clip.set_templates(ENSEMBLE_TEMPLATES)
    assert not torch.allclose(single, clip.text_features)
    images = torch.randn(2, 3, 224, 224)
    logits = clip.logits(images)
    assert logits.shape == (2, len(CLASSES))
    assert torch.allclose(logits, clip.classify(clip.image_features(images)))
