import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from classifier.data import CLASS_NAMES_RU, CLASSES, display_transform, eval_transform, load_image, make_loader, read_split
from classifier.engine import cross_entropy, predict, softmax
from classifier.metrics import confusion, per_class, summary
from classifier.model import load_checkpoint, pick_device
from plot_style import CORRECT, INK_SECONDARY, RUN_STYLE, SERIES, SPLIT_STYLE, SURFACE, WRONG, apply_style

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
RUNS = ("resnet50", "resnet50_noaug")
SPLITS = ("train", "val", "test_kaggle", "own")
SPLIT_NAMES = {"train": "train (режим eval)", "val": "val", "test_kaggle": "test Kaggle", "own": "свой тест"}
GRID_LIST = DATA / "own_grid.txt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Оценка дообученных моделей на всех выборках")
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def ru(index):
    return CLASS_NAMES_RU[CLASSES[index]]


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_predictions(path, items, labels, probabilities):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "prediction", "confidence", *[f"p_{name}" for name in CLASSES]])
        for (item_path, _), label, probs in zip(items, labels, probabilities):
            writer.writerow([str(Path(item_path).relative_to(DATA)), CLASSES[label], CLASSES[int(probs.argmax())],
                             f"{probs.max():.4f}", *[f"{p:.4f}" for p in probs]])


def read_history(name):
    path = RESULTS / f"history_{name}.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def plot_curves(runs, path):
    histories = {name: read_history(name) for name in runs}
    histories = {name: h for name, h in histories.items() if h is not None}
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, h in histories.items():
        color = RUN_STYLE.get(name, {"color": SERIES[4]})["color"]
        for split, key_loss, key_acc in (("train", "train_ce", "train_acc"), ("val", "val_ce", "val_acc")):
            style = SPLIT_STYLE[split]
            label = f"{RUN_STYLE.get(name, {'label': name})['label']}, {style['label']}"
            axes[0].plot(h["epoch"], h[key_loss], color=color, linestyle=style["linestyle"], marker=style["marker"],
                         label=label)
            axes[1].plot(h["epoch"], h[key_acc], color=color, linestyle=style["linestyle"], marker=style["marker"],
                         label=label)
    axes[0].set_title("Кросс-энтропия по эпохам")
    axes[0].set_ylabel("CE")
    axes[1].set_title("Accuracy по эпохам")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.set_xlabel("эпоха")
    axes[1].legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_confusions(matrices, path):
    cmap = LinearSegmentedColormap.from_list("ink", [SURFACE, SERIES[0]])
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.8), squeeze=False)
    names = [ru(i) for i in range(len(CLASSES))]
    for ax, (split, matrix) in zip(axes[0], matrices.items()):
        rows = matrix.sum(axis=1, keepdims=True)
        share = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if matrix[i, j]:
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if share[i, j] > 0.55 else INK_SECONDARY)
        ax.set_xticks(range(len(CLASSES)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), names)
        ax.set_xlabel("прогноз")
        ax.set_ylabel("истинный класс")
        ax.set_title(SPLIT_NAMES[split])
    fig.savefig(path)
    plt.close(fig)


def grid_subset(items, labels, probabilities):
    if not GRID_LIST.is_file():
        return items, labels, probabilities
    allowed = set(GRID_LIST.read_text(encoding="utf-8").split())
    keep = [i for i, (path, _) in enumerate(items) if "/".join(Path(path).parts[-2:]) in allowed]
    return [items[i] for i in keep], labels[keep], probabilities[keep]


def plot_grid(items, labels, probabilities, path, columns=8):
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.5 * rows), squeeze=False)
    show = display_transform()
    for ax in axes.flat:
        ax.axis("off")
    for ax, (item_path, _), label, probs in zip(axes.flat, items, labels, probabilities):
        predicted = int(probs.argmax())
        ax.imshow(show(load_image(item_path)))
        ax.set_title(f"{ru(predicted)} {probs.max():.0%}\nистина: {ru(label)}", fontsize=9,
                     color=CORRECT if predicted == label else WRONG)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    apply_style()
    device = pick_device(args.device)
    RESULTS.mkdir(exist_ok=True)
    splits = {name: read_split(DATA / "splits" / f"{name}.csv", DATA) for name in args.splits
              if (DATA / "splits" / f"{name}.csv").is_file()}
    metrics, per_class_md, matrices, own = [], [], {}, None
    for run in args.runs:
        model, checkpoint = load_checkpoint(CHECKPOINTS / f"{run}.pt", device)
        print(f"{run}: эпоха {checkpoint['epoch']}")
        for split, items in splits.items():
            loader = make_loader(items, eval_transform(), args.batch_size, False, args.workers)
            logits, labels = predict(model, loader, device)
            probabilities = softmax(logits)
            result = summary(labels, logits)
            result.update(run=run, split=split, ce=cross_entropy(logits, labels))
            metrics.append(result)
            write_predictions(RESULTS / f"predictions_{run}_{split}.csv", items, labels, probabilities)
            print(f"  {split:12} n={result['n']:6} acc {result['accuracy']:.4f} F1 {result['macro_f1']:.4f} "
                  f"CE {result['ce']:.3f}")
            if run != args.runs[0] or split == "train":
                continue
            predictions = probabilities.argmax(axis=1)
            matrices[split] = confusion(labels, predictions)
            rows = [(ru(i), f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", r["support"])
                    for i, r in enumerate(per_class(labels, predictions))]
            per_class_md.append(f"### {SPLIT_NAMES[split]}\n\n"
                                + markdown_table(["класс", "precision", "recall", "F1", "N"], rows))
            if split == "own":
                own = (items, labels, probabilities)

    with open(RESULTS / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "split", "n", "accuracy", "macro_f1", "ce"])
        writer.writeheader()
        writer.writerows(metrics)
    header = ["выборка", "N"] + [f"{run}: {m}" for run in args.runs for m in ("acc", "F1", "CE")]
    rows = []
    for split in splits:
        cells = [SPLIT_NAMES[split], next(m["n"] for m in metrics if m["split"] == split)]
        for run in args.runs:
            m = next(m for m in metrics if m["run"] == run and m["split"] == split)
            cells += [f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['ce']:.3f}"]
        rows.append(cells)
    (RESULTS / "metrics.md").write_text(markdown_table(header, rows), encoding="utf-8")
    (RESULTS / "per_class.md").write_text("\n".join(per_class_md), encoding="utf-8")
    plot_confusions(matrices, RESULTS / "confusion.png")
    plot_curves(args.runs, RESULTS / "curves.png")
    if own is not None:
        plot_grid(*grid_subset(*own), RESULTS / "own_test_grid.png")
    print(f"Готово: {RESULTS}")


if __name__ == "__main__":
    main()
