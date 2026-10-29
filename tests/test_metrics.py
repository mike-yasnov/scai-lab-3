import numpy as np
import pytest

from classifier.metrics import accuracy, confusion, macro_f1, per_class, summary


def test_accuracy_and_confusion():
    labels = [0, 0, 1, 2, 3, 4, 5, 5]
    predictions = [0, 1, 1, 2, 3, 4, 5, 0]
    assert accuracy(labels, predictions) == pytest.approx(0.75)
    matrix = confusion(labels, predictions)
    assert matrix.shape == (6, 6) and matrix.sum() == 8
    assert matrix[0, 1] == 1 and matrix[5, 0] == 1 and np.trace(matrix) == 6


def test_macro_f1_counts_missing_classes_as_zero():
    assert macro_f1([0, 0], [0, 0]) == pytest.approx(1 / 6)


def test_per_class_and_summary():
    labels = np.array([0, 0, 1, 1])
    logits = np.eye(6)[[0, 1, 1, 1]]
    rows = per_class(labels, logits.argmax(1))
    assert rows[0]["precision"] == 1.0 and rows[0]["recall"] == 0.5 and rows[0]["support"] == 2
    assert rows[1]["recall"] == 1.0 and rows[1]["precision"] == pytest.approx(2 / 3)
    result = summary(labels, logits)
    assert result["n"] == 4 and result["accuracy"] == pytest.approx(0.75)


def test_mcnemar_exact_test():
    from clip_eval import mcnemar
    labels = np.zeros(14, dtype=int)
    first = np.array([0] * 2 + [1] * 12)
    second = np.array([1] * 1 + [0] * 13)
    only_first, only_second, p = mcnemar(first, second, labels)
    assert (only_first, only_second) == (1, 12)
    assert p == pytest.approx(2 * 14 / 2 ** 13)
    assert mcnemar(labels, labels, labels) == (0, 0, 1.0)
