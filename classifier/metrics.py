import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .data import CLASSES


def accuracy(labels, predictions):
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def macro_f1(labels, predictions):
    return float(f1_score(labels, predictions, labels=list(range(len(CLASSES))), average="macro", zero_division=0))


def confusion(labels, predictions):
    return confusion_matrix(labels, predictions, labels=list(range(len(CLASSES))))


def per_class(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(CLASSES))), zero_division=0)
    return [{"class": name, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(CLASSES, precision, recall, f1, support)]


def summary(labels, logits):
    predictions = np.asarray(logits).argmax(axis=1)
    return {"n": int(len(labels)), "accuracy": accuracy(labels, predictions), "macro_f1": macro_f1(labels, predictions)}
