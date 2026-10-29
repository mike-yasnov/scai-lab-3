import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from classifier.clip_zeroshot import ENSEMBLE_TEMPLATES, SINGLE_TEMPLATE, ClipZeroShot
from classifier.data import make_loader, read_split
from classifier.engine import softmax
from classifier.metrics import summary
from classifier.model import pick_device
from evaluate import SPLIT_NAMES, markdown_table, write_predictions

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = {"ViT-B-32": "laion2b_s34b_b79k", "ViT-L-14": "laion2b_s32b_b82k"}
PROMPTS = {"один шаблон": SINGLE_TEMPLATE, "ансамбль из 7 шаблонов": ENSEMBLE_TEMPLATES}
SPLITS = ("val", "test_kaggle", "own")
PROBE_C = (0.1, 1.0, 10.0, 100.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CLIP без обучения и linear probe на тех же выборках")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--no-probe", action="store_true", help="не обучать логистическую регрессию на признаках")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def encode(clip, items, batch_size, workers):
    loader = make_loader(items, clip.preprocess, batch_size, False, workers)
    features, labels = [], []
    for images, batch_labels in loader:
        features.append(clip.image_features(images).cpu())
        labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels).numpy()


def fit_probe(train, val):
    best = None
    for c in PROBE_C:
        probe = LogisticRegression(C=c, max_iter=3000).fit(*train)
        score = summary(val[1], probe.decision_function(val[0]))["macro_f1"]
        print(f"    C={c:g}: val F1 {score:.4f}")
        if best is None or score > best[0]:
            best = (score, c, probe)
    return best[1], best[2]


def mcnemar(first, second, labels):
    only_first = int(np.sum((first == labels) & (second != labels)))
    only_second = int(np.sum((first != labels) & (second == labels)))
    n = only_first + only_second
    if n == 0:
        return only_first, only_second, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_first, only_second) + 1)) / 2 ** n
    return only_first, only_second, min(1.0, 2 * tail)


def read_predictions(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: row["path"])
    return np.array([row["label"] for row in rows]), np.array([row["prediction"] for row in rows])


def mcnemar_table(models):
    rows = []
    for split in ("test_kaggle", "own"):
        resnet = RESULTS / f"predictions_resnet50_{split}.csv"
        if not resnet.is_file():
            continue
        labels, first = read_predictions(resnet)
        for model_name in models:
            path = RESULTS / f"predictions_clip_{model_name}_{split}.csv"
            if not path.is_file():
                continue
            _, second = read_predictions(path)
            only_resnet, only_clip, p = mcnemar(first, second, labels)
            rows.append([SPLIT_NAMES[split], f"CLIP {model_name}, ансамбль", len(labels), only_resnet, only_clip,
                         f"{p:.2g}"])
    header = ["выборка", "с чем сравниваем ResNet-50", "N", "верна только ResNet", "верен только CLIP", "p (Макнемар)"]
    return markdown_table(header, rows) if rows else ""


def resnet_rows():
    path = RESULTS / "metrics.csv"
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{"model": "ResNet-50, дообучение", "method": "вся сеть, 12 эпох", "split": row["split"],
                 "n": int(row["n"]), "accuracy": float(row["accuracy"]), "macro_f1": float(row["macro_f1"])}
                for row in csv.DictReader(f) if row["run"] == "resnet50" and row["split"] in SPLITS]


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    RESULTS.mkdir(exist_ok=True)
    splits = {name: read_split(DATA / "splits" / f"{name}.csv", DATA) for name in args.splits
              if (DATA / "splits" / f"{name}.csv").is_file()}
    rows = []
    for model_name in args.models:
        clip = ClipZeroShot(model_name, MODELS[model_name], device)
        started = time.perf_counter()
        features = {split: encode(clip, items, args.batch_size, args.workers) for split, items in splits.items()}
        print(f"{model_name}: признаки за {time.perf_counter() - started:.0f} с")
        for prompt, templates in PROMPTS.items():
            clip.set_templates(templates)
            for split, (feats, labels) in features.items():
                logits = clip.classify(feats).cpu().numpy()
                result = summary(labels, logits)
                rows.append({"model": f"CLIP {model_name}", "method": f"zero-shot, {prompt}", "split": split, **result})
                print(f"  {prompt:24} {split:12} acc {result['accuracy']:.4f} F1 {result['macro_f1']:.4f}")
                if templates is ENSEMBLE_TEMPLATES:
                    write_predictions(RESULTS / f"predictions_clip_{model_name}_{split}.csv", splits[split], labels,
                                      softmax(logits))
        if args.no_probe:
            continue
        if "val" not in features:
            print("  linear probe пропущен: C подбирается по val")
            continue
        train = encode(clip, read_split(DATA / "splits" / "train.csv", DATA), args.batch_size, args.workers)
        c, probe = fit_probe(train, features["val"])
        for split, (feats, labels) in features.items():
            result = summary(labels, probe.decision_function(feats.numpy()))
            rows.append({"model": f"CLIP {model_name}", "method": f"linear probe, C={c:g}", "split": split, **result})
            print(f"  linear probe C={c:g}      {split:12} acc {result['accuracy']:.4f} F1 {result['macro_f1']:.4f}")

    with open(RESULTS / "clip.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", "split", "n", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    table = resnet_rows() + rows
    methods = list(dict.fromkeys((r["model"], r["method"]) for r in table))
    present = [s for s in SPLITS if any(r["split"] == s for r in table)]
    header = ["модель", "способ"] + [f"{SPLIT_NAMES[s]}: {m}" for s in present for m in ("acc", "F1")]
    lines = []
    for model, method in methods:
        cells = [model, method]
        for split in present:
            r = next((r for r in table if (r["model"], r["method"], r["split"]) == (model, method, split)), None)
            cells += [f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}"] if r else ["", ""]
        lines.append(cells)
    (RESULTS / "clip.md").write_text(markdown_table(header, lines), encoding="utf-8")
    (RESULTS / "mcnemar.md").write_text(mcnemar_table(args.models), encoding="utf-8")
    print(f"Готово: {RESULTS / 'clip.md'}")


if __name__ == "__main__":
    main()
