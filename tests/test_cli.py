import pytest
import torch

import prepare_data
from classifier.model import build_model, save_checkpoint
from conftest import make_image, make_tree
from predict import check_args, main, parse_args


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    torch.manual_seed(0)
    save_checkpoint(path, build_model(pretrained=False), 1, {}, {})
    return str(path)


@pytest.fixture
def image(tmp_path):
    return str(make_image(tmp_path / "scene.jpg"))


@pytest.mark.parametrize("extra, fragment", [
    (["--top", "0"], "--top"),
    (["--top", "7"], "--top"),
    (["--gradcam", "out.jpg"], ".png"),
])
def test_rejects_bad_arguments(image, checkpoint, extra, fragment):
    error = check_args(parse_args([image, "--checkpoint", checkpoint, *extra]))
    assert error is not None and fragment in error


def test_rejects_missing_files(image, checkpoint, tmp_path):
    assert "не найден" in check_args(parse_args(["nope.jpg", "--checkpoint", checkpoint]))
    assert "чекпойнт" in check_args(parse_args([image, "--checkpoint", str(tmp_path / "none.pt")]))
    heic = tmp_path / "photo.heic"
    heic.write_bytes(b"x")
    assert "не поддерживается" in check_args(parse_args([str(heic), "--checkpoint", checkpoint]))


def test_prints_top_classes_and_saves_gradcam(image, checkpoint, tmp_path, capsys):
    out = tmp_path / "cam" / "out.png"
    main([image, image, "--checkpoint", checkpoint, "--top", "2", "--gradcam", str(out), "--device", "cpu"])
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.strip().startswith(("1.", "2.")) for line in lines) == 4
    assert out.is_file() and out.stat().st_size > 0


def test_prepare_data_writes_splits(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    make_tree(data / "raw" / "seg_train" / "seg_train", per_class=10)
    make_tree(data / "raw" / "seg_test" / "seg_test", per_class=2)
    make_tree(data / "own_test", per_class=1)
    monkeypatch.setattr(prepare_data, "DATA", data)
    monkeypatch.setattr(prepare_data, "RAW", data / "raw")
    prepare_data.main(["--val-fraction", "0.2"])
    out = capsys.readouterr().out
    assert "не меньше 50" in out
    lines = {name: (data / "splits" / f"{name}.csv").read_text().splitlines()
             for name in ("train", "val", "test_kaggle", "own")}
    assert [len(v) - 1 for v in lines.values()] == [48, 12, 12, 6]
    assert lines["own"][1].startswith("own_test/")
