import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from classifier.data import CLASS_NAMES_RU, CLASSES, display_transform, eval_transform, load_image, read_split
from classifier.gradcam import GradCAM, overlay
from classifier.model import load_checkpoint, pick_device
from plot_style import CORRECT, WRONG, apply_style

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Тепловые карты Grad-CAM для своих фото и ошибок на Kaggle test")
    parser.add_argument("--run", default="resnet50")
    parser.add_argument("--own", nargs="*", default=None, help="имена файлов из data/own_test, по умолчанию по одному на класс")
    parser.add_argument("--errors", type=int, default=6, help="сколько уверенных ошибок показать")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def pick_own(items, names):
    if names:
        chosen = [item for item in items if Path(item[0]).name in set(names)]
        return sorted(chosen, key=lambda item: names.index(Path(item[0]).name))
    first = {}
    for item in items:
        first.setdefault(item[1], item)
    return [first[label] for label in sorted(first)]


def confident_errors(run, count):
    path = RESULTS / f"predictions_{run}_test_kaggle.csv"
    with open(path, newline="", encoding="utf-8") as f:
        wrong = [row for row in csv.DictReader(f) if row["label"] != row["prediction"]]
    wrong.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return [(str(DATA / row["path"]), CLASSES.index(row["label"])) for row in wrong[:count]]


def heatmap(cam, image, device, class_index=None):
    result, index, probabilities = cam(eval_transform()(image).to(device), class_index)
    return overlay(display_transform()(image), result), index, probabilities


def plot_own(cam, items, device, path):
    fig, axes = plt.subplots(2, len(items), figsize=(2.6 * len(items), 5.6), squeeze=False)
    for column, (item_path, label) in enumerate(items):
        image = load_image(item_path)
        heat, predicted, probabilities = heatmap(cam, image, device)
        axes[0, column].imshow(display_transform()(image))
        axes[0, column].set_title(f"истина: {CLASS_NAMES_RU[CLASSES[label]]}", fontsize=10)
        axes[1, column].imshow(heat)
        axes[1, column].set_title(f"{CLASS_NAMES_RU[CLASSES[predicted]]} {probabilities[predicted]:.0%}", fontsize=10,
                                  color=CORRECT if predicted == label else WRONG)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def plot_errors(cam, items, device, path):
    fig, axes = plt.subplots(len(items), 3, figsize=(7.8, 2.7 * len(items)), squeeze=False)
    for row, (item_path, label) in enumerate(items):
        image = load_image(item_path)
        heat_pred, predicted, probabilities = heatmap(cam, image, device)
        heat_true, _, _ = heatmap(cam, image, device, label)
        axes[row, 0].imshow(display_transform()(image))
        axes[row, 0].set_title(Path(item_path).name, fontsize=9)
        axes[row, 1].imshow(heat_pred)
        axes[row, 1].set_title(f"прогноз: {CLASS_NAMES_RU[CLASSES[predicted]]} {probabilities[predicted]:.0%}",
                               fontsize=9, color=WRONG)
        axes[row, 2].imshow(heat_true)
        axes[row, 2].set_title(f"истина: {CLASS_NAMES_RU[CLASSES[label]]} {probabilities[label]:.0%}", fontsize=9,
                               color=CORRECT)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    apply_style()
    device = pick_device(args.device)
    model, _ = load_checkpoint(CHECKPOINTS / f"{args.run}.pt", device)
    cam = GradCAM(model, model.layer4)
    own_split = DATA / "splits" / "own.csv"
    if own_split.is_file():
        items = pick_own(read_split(own_split, DATA), args.own)
        plot_own(cam, items, device, RESULTS / "gradcam.png")
        print(f"gradcam.png: {len(items)} своих фото")
    if args.errors and (RESULTS / f"predictions_{args.run}_test_kaggle.csv").is_file():
        items = confident_errors(args.run, args.errors)
        plot_errors(cam, items, device, RESULTS / "gradcam_errors.png")
        print(f"gradcam_errors.png: {len(items)} ошибок")
    cam.close()


if __name__ == "__main__":
    main()
