import torch
from PIL import Image

from classifier.data import (CLASSES, ImageListDataset, eval_transform, list_images, load_image, read_split,
                             stratified_split, train_transform, write_split)


def test_list_images_labels_follow_class_order(tree):
    (tree / "forest" / "notes.txt").write_text("x")
    items = list_images(tree)
    assert len(items) == 4 * len(CLASSES)
    assert all(CLASSES[label] in path for path, label in items)


def test_stratified_split_is_disjoint_balanced_and_repeatable(tree):
    items = list_images(tree)
    train, val = stratified_split(items, 0.25, seed=3)
    assert len(val) == len(CLASSES) and len(train) == 3 * len(CLASSES)
    assert not set(train) & set(val)
    assert sorted(label for _, label in val) == list(range(len(CLASSES)))
    assert stratified_split(items, 0.25, seed=3) == (train, val)
    assert stratified_split(items, 0.25, seed=4) != (train, val)


def test_split_roundtrip(tree, tmp_path):
    items = list_images(tree)
    write_split(tmp_path / "splits" / "all.csv", items, tree.parent)
    assert read_split(tmp_path / "splits" / "all.csv", tree.parent) == items


def test_transforms_give_square_input(tree):
    image = load_image(list_images(tree)[0][0])
    assert eval_transform()(image).shape == (3, 224, 224)
    assert train_transform()(image).shape == (3, 224, 224)
    assert torch.equal(train_transform(augment=False)(image), eval_transform()(image))


def test_load_image_applies_exif_rotation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), "red").save(path, exif=exif)
    assert load_image(path).size == (20, 40)


def test_load_image_converts_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (30, 30), 128).save(path)
    assert load_image(path).mode == "RGB"


def test_dataset_returns_tensor_and_label(tree):
    dataset = ImageListDataset(list_images(tree), eval_transform())
    image, label = dataset[5]
    assert image.shape == (3, 224, 224) and label == 1
