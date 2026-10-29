import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

RUN_STYLE = {
    "resnet50": {"color": SERIES[0], "label": "с аугментациями и label smoothing"},
    "resnet50_noaug": {"color": SERIES[1], "label": "без аугментаций (абляция)"},
}

SPLIT_STYLE = {
    "train": {"linestyle": "--", "marker": None, "label": "train"},
    "val": {"linestyle": "-", "marker": "o", "label": "val"},
}

CORRECT = SERIES[2]
WRONG = SERIES[7]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "font.size": 11,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
