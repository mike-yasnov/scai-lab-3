import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from classifier.data import CLASSES, list_images, load_image, stratified_split, write_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DATASET = "puneet6060/intel-image-classification"
MIN_OWN = 50


def download():
    kaggle = shutil.which("kaggle") or str(Path.home() / ".local" / "bin" / "kaggle")
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run([kaggle, "datasets", "download", "-d", DATASET, "-p", str(RAW), "--unzip"], check=True)


def counts(items):
    return {name: sum(1 for _, label in items if label == i) for i, name in enumerate(CLASSES)}


def check_readable(items):
    bad = []
    for path, _ in items:
        try:
            load_image(path)
        except OSError:
            bad.append(path)
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(description="Подготовка Intel Image Classification и своего теста")
    parser.add_argument("--download", action="store_true", help="скачать датасет через Kaggle API")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="доля seg_train под валидацию")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.download:
        download()
    train_root = RAW / "seg_train" / "seg_train"
    test_root = RAW / "seg_test" / "seg_test"
    if not train_root.is_dir() or not test_root.is_dir():
        sys.exit("Ошибка: нет data/raw/seg_train и seg_test, запустите с --download")

    train, val = stratified_split(list_images(train_root), args.val_fraction, args.seed)
    test = list_images(test_root)
    splits = {"train": train, "val": val, "test_kaggle": test}
    own_root = DATA / "own_test"
    if own_root.is_dir():
        own = list_images(own_root)
        bad = check_readable(own)
        if bad:
            sys.exit("Ошибка: не читаются файлы своего теста: " + ", ".join(bad))
        if len(own) < MIN_OWN:
            print(f"Внимание: в своём тесте {len(own)} изображений, по заданию нужно не меньше {MIN_OWN}")
        splits["own"] = own

    for name, items in splits.items():
        write_split(DATA / "splits" / f"{name}.csv", items, DATA)
        per_class = ", ".join(f"{k} {v}" for k, v in counts(items).items())
        print(f"{name:12} {len(items):6} | {per_class}")


if __name__ == "__main__":
    main()
