import numpy as np
import torch
import torch.nn.functional as F


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler=None):
    model.train()
    total_loss = total_ce = correct = count = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        n = len(labels)
        total_loss += loss.item() * n
        total_ce += F.cross_entropy(logits.float(), labels).item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n
    return total_loss / count, total_ce / count, correct / count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device, non_blocking=True)).float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def cross_entropy(logits, labels):
    return float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels).long()))


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
