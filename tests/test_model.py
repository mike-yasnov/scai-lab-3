import torch

from classifier.data import CLASSES
from classifier.model import build_model, load_checkpoint, pick_device, save_checkpoint


def test_model_outputs_one_logit_per_class():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, len(CLASSES))


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    save_checkpoint(tmp_path / "m.pt", model, 3, {"val_f1": 0.5}, {"lr": 1e-3})
    loaded, checkpoint = load_checkpoint(tmp_path / "m.pt", torch.device("cpu"))
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    assert checkpoint["epoch"] == 3 and checkpoint["classes"] == list(CLASSES)
    assert not loaded.training


def test_pick_device_respects_explicit_name():
    assert pick_device("cpu").type == "cpu"
