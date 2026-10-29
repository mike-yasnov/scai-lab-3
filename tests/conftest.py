import numpy as np
import pytest
from PIL import Image

from classifier.data import CLASSES


def make_image(path, size=(64, 48), seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def make_tree(root, per_class=4, size=(64, 48)):
    for label, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            make_image(folder / f"{i}.jpg", size, seed=label * 100 + i)
    return root


@pytest.fixture
def tree(tmp_path):
    return make_tree(tmp_path / "images")
