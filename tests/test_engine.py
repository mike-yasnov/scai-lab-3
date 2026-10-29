import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classifier.engine import cross_entropy, predict, softmax, train_one_epoch


def blobs(n=96, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 3
    images = torch.randn(n, 3, 8, 8, generator=generator) * 0.3
    images[:, 0] += labels.view(-1, 1, 1) - 1.0
    return TensorDataset(images, labels)


def tiny_model():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(8, 3))


def test_training_reduces_loss():
    torch.manual_seed(0)
    model, device = tiny_model(), torch.device("cpu")
    loader = DataLoader(blobs(), batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    first = train_one_epoch(model, loader, optimizer, scheduler, criterion, device)
    for _ in range(15):
        last = train_one_epoch(model, loader, optimizer, scheduler, criterion, device)
    assert last[1] < first[1] and last[2] > 0.9
    logits, labels = predict(model, DataLoader(blobs(seed=1), batch_size=32), device)
    assert logits.shape == (96, 3) and labels.shape == (96,)
    assert (logits.argmax(1) == labels).mean() > 0.9


def test_softmax_and_cross_entropy_match_torch():
    logits = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32) * 10
    labels = np.array([0, 1, 2, 3, 5])
    assert np.allclose(softmax(logits).sum(axis=1), 1.0)
    assert np.allclose(softmax(logits), torch.softmax(torch.from_numpy(logits), 1).numpy(), atol=1e-6)
    expected = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    assert cross_entropy(logits, labels) == pytest.approx(expected, rel=1e-5)
