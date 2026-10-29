import numpy as np

import evaluate


def test_grid_subset_keeps_only_listed_photos(tmp_path, monkeypatch):
    items = [(str(tmp_path / "own_test" / "sea" / "a.jpg"), 4), (str(tmp_path / "own_test" / "street" / "b.jpg"), 5)]
    labels, probabilities = np.array([4, 5]), np.eye(6)[[4, 5]]
    monkeypatch.setattr(evaluate, "GRID_LIST", tmp_path / "none.txt")
    assert evaluate.grid_subset(items, labels, probabilities)[0] == items
    listed = tmp_path / "grid.txt"
    listed.write_text("street/b.jpg\n")
    monkeypatch.setattr(evaluate, "GRID_LIST", listed)
    kept, kept_labels, kept_probabilities = evaluate.grid_subset(items, labels, probabilities)
    assert kept == items[1:] and kept_labels.tolist() == [5] and kept_probabilities.shape == (1, 6)


def test_markdown_table():
    assert evaluate.markdown_table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |\n"
