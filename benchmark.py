import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import open_clip
import torch
import torchvision

from classifier.model import build_model, pick_device

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIELDS = ["device", "model", "params_m", "batch", "precision", "ms_per_batch", "images_per_s", "peak_memory_gb"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Скорость инференса ResNet-50 и визуальных энкодеров CLIP")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return f"{chip} ({device.type.upper()})"
    return f"{platform.processor() or platform.machine()} ({device.type.upper()})"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def models():
    yield "ResNet-50", build_model(pretrained=False)
    for name in ("ViT-B-32", "ViT-L-14"):
        yield f"CLIP {name} (изображения)", open_clip.create_model(name, pretrained=None).visual


@torch.no_grad()
def measure(model, device, batch, iters, warmup, amp):
    images = torch.randn(batch, 3, 224, 224, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            model(images)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(iters):
            model(images)
        synchronize(device)
    seconds = (time.perf_counter() - started) / iters
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    return seconds * 1000, batch / seconds, peak


def environment(device):
    info = {"device": device_name(device), "python": platform.python_version(), "os": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "open_clip": open_clip.__version__,
            "cpu_count": os.cpu_count(),
            "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu_memory_gb=round(props.total_memory / 2**30, 1))
    return info


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    name = device_name(device)
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for model_name, model in models():
        model = model.to(device).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        for precision in ["fp32", "fp16"] if device.type == "cuda" else ["fp32"]:
            for batch in args.batches:
                ms, speed, peak = measure(model, device, batch, args.iters, args.warmup, precision == "fp16")
                rows.append({"device": name, "model": model_name, "params_m": f"{params:.1f}", "batch": batch,
                             "precision": precision, "ms_per_batch": f"{ms:.1f}", "images_per_s": f"{speed:.0f}",
                             "peak_memory_gb": f"{peak:.2f}" if peak is not None else ""})
                print(f"{model_name:28} {precision} batch {batch:3}: {ms:7.1f} мс, {speed:6.0f} изобр./с")
        del model
    path = RESULTS / "speed.csv"
    old = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            old = [row for row in csv.DictReader(f) if row["device"] != name]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(old + rows)
    env_path = RESULTS / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    env[name] = environment(device)
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {path}")


if __name__ == "__main__":
    main()
