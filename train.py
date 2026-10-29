import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from classifier.data import eval_transform, make_loader, read_split, train_transform
from classifier.engine import cross_entropy, predict, train_one_epoch
from classifier.metrics import summary
from classifier.model import build_model, pick_device, save_checkpoint

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Дообучение ResNet-50 на Intel Image Classification")
    parser.add_argument("--name", default="resnet50", help="имя прогона: чекпойнт и история")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true", help="без аугментаций (абляция)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="взять только столько train-изображений (проверка)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def make_scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    train_items = read_split(DATA / "splits" / "train.csv", DATA)
    val_items = read_split(DATA / "splits" / "val.csv", DATA)
    if args.limit:
        order = np.random.default_rng(args.seed).permutation(len(train_items))[:args.limit]
        train_items = [train_items[i] for i in order]
        val_items = val_items[:max(64, args.limit // 4)]
    train_loader = make_loader(train_items, train_transform(not args.no_augment), args.batch_size, True, args.workers)
    val_loader = make_loader(val_items, eval_transform(), args.batch_size * 2, False, args.workers)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, max(1, int(args.warmup_epochs * len(train_loader))))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    CHECKPOINTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    history, best, started = [], None, time.perf_counter()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"{args.name}: {len(train_items)} train, {len(val_items)} val, {gpu}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_ce, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                                                          device, scaler)
        logits, labels = predict(model, val_loader, device)
        val = summary(labels, logits)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss, "train_ce": train_ce,
               "train_acc": train_acc, "val_ce": cross_entropy(logits, labels), "val_acc": val["accuracy"],
               "val_f1": val["macro_f1"], "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if best is None or row["val_f1"] > best["val_f1"]:
            best = row
            save_checkpoint(CHECKPOINTS / f"{args.name}.pt", model, epoch, row, vars(args))
        with open(RESULTS / f"history_{args.name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"  эпоха {epoch:2}: train CE {train_ce:.3f} acc {train_acc:.4f} | val CE {row['val_ce']:.3f} "
              f"acc {row['val_acc']:.4f} F1 {row['val_f1']:.4f} | {row['seconds']:.0f} с", flush=True)
    total = time.perf_counter() - started
    info = {"name": args.name, "settings": vars(args), "device": gpu, "train_images": len(train_items),
            "val_images": len(val_items), "best_epoch": best["epoch"], "best": best, "total_seconds": total,
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None}
    (RESULTS / f"train_{args.name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Лучшая эпоха {best['epoch']}: val acc {best['val_acc']:.4f}, F1 {best['val_f1']:.4f}; всего {total / 60:.1f} мин")


if __name__ == "__main__":
    main()
