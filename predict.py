import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from classifier.data import CLASS_NAMES_RU, CLASSES, IMAGE_EXTENSIONS, display_transform, eval_transform, load_image
from classifier.gradcam import GradCAM, overlay
from classifier.model import load_checkpoint, pick_device
from plot_style import apply_style

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Классификация сцены: здания, лес, ледник, горы, море, улица")
    parser.add_argument("images", nargs="+", help="пути к изображениям (jpg, png, bmp, webp)")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "resnet50.pt"))
    parser.add_argument("--top", type=int, default=3, help="сколько классов вывести")
    parser.add_argument("--gradcam", default=None, help="сохранить тепловые карты в этот .png")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def check_args(args):
    for image in args.images:
        path = Path(image)
        if not path.is_file():
            return f"файл {image} не найден"
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return f"формат {path.suffix} не поддерживается, нужен один из {', '.join(sorted(IMAGE_EXTENSIONS))}"
    if not Path(args.checkpoint).is_file():
        return f"чекпойнт {args.checkpoint} не найден, сначала запустите train.py"
    if not 1 <= args.top <= len(CLASSES):
        return f"--top должен быть от 1 до {len(CLASSES)}"
    if args.gradcam is not None and Path(args.gradcam).suffix.lower() != ".png":
        return "--gradcam ждёт путь к файлу .png"
    return None


def run(args):
    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    classes = checkpoint["classes"]
    cam = GradCAM(model, model.layer4) if args.gradcam else None
    panels = []
    for image_path in args.images:
        image = load_image(image_path)
        tensor = eval_transform()(image).to(device)
        if cam is None:
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0].cpu().numpy()
        else:
            heat, predicted, probabilities = cam(tensor)
            shown = display_transform()(image)
            panels.append((image_path, shown, overlay(shown, heat), CLASS_NAMES_RU[classes[predicted]],
                           float(probabilities[predicted])))
        print(image_path)
        for rank, index in enumerate(probabilities.argsort()[::-1][:args.top], start=1):
            name = classes[index]
            print(f"  {rank}. {CLASS_NAMES_RU[name]} ({name}): {probabilities[index]:.1%}")
    if cam is not None:
        cam.close()
        save_gradcam(panels, args.gradcam)
        print(f"Grad-CAM: {args.gradcam}")


def save_gradcam(panels, path):
    apply_style()
    fig, axes = plt.subplots(len(panels), 2, figsize=(6, 3.1 * len(panels)), squeeze=False)
    for row, (image_path, image, heat, name, probability) in enumerate(panels):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(Path(image_path).name, fontsize=9)
        axes[row, 1].imshow(heat)
        axes[row, 1].set_title(f"{name} {probability:.0%}", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    error = check_args(args)
    if error:
        sys.exit(f"Ошибка: {error}")
    try:
        run(args)
    except (OSError, RuntimeError) as exc:
        sys.exit(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()
