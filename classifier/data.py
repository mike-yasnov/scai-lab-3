import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
CLASS_NAMES_RU = {"buildings": "здания", "forest": "лес", "glacier": "ледник", "mountain": "горы",
                  "sea": "море", "street": "улица"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def list_images(root):
    items = []
    for label, name in enumerate(CLASSES):
        folder = Path(root) / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((str(path), label))
    return items


def stratified_split(items, val_fraction, seed=0):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(len(CLASSES)):
        group = [item for item in items if item[1] == label]
        order = rng.permutation(len(group))
        n_val = round(len(group) * val_fraction)
        val += [group[i] for i in order[:n_val]]
        train += [group[i] for i in order[n_val:]]
    return train, val


def write_split(path, items, root):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for item_path, label in items:
            writer.writerow([str(Path(item_path).relative_to(root)), CLASSES[label]])


def read_split(path, root):
    with open(path, newline="", encoding="utf-8") as f:
        return [(str(Path(root) / row["path"]), CLASSES.index(row["label"])) for row in csv.DictReader(f)]


def load_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def train_transform(augment=True):
    if not augment:
        return eval_transform()
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def display_transform():
    return transforms.Compose([transforms.Resize(INPUT_SIZE), transforms.CenterCrop(INPUT_SIZE)])


class ImageListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        return self.transform(load_image(path)), label


def make_loader(items, transform, batch_size=64, shuffle=False, workers=4):
    return DataLoader(ImageListDataset(items, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
