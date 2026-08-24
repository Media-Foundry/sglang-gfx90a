#!/usr/bin/env python3
"""Generate publication figures from the validated experiment ledger."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

BLUE = "#2F5D8C"
TEAL = "#2A8C82"
ORANGE = "#D17A3A"
RED = "#B44A4A"
INK = "#25313C"
MUTED = "#66727D"


def preferred_font() -> str:
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in names else "Liberation Sans"


FONT = preferred_font()
sns.set_theme(
    context="paper",
    style="white",
    font=FONT,
    rc={
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "font.family": FONT,
        "font.size": 9.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    },
)
mpl.rcParams["savefig.transparent"] = False


def finish(ax: plt.Axes, *, ygrid: bool = True) -> None:
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if ygrid:
        ax.grid(axis="y", color="#DDE3E8", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def decode_progress() -> None:
    data = pd.DataFrame(
        {
            "Stage": [
                "EP4\nCKTile",
                "EP1\nCKTile",
                "INT8-dot\nMoE",
                "Peer-read\nall-reduce",
                "Router +\ngeometry",
                "Final\ncheckpoint",
            ],
            "Native AR (tok/s)": [14.6, 20.14, 25.54, 60.20, 66.10, 74.50],
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    colors = ["#A8B8C8", "#8AA7C1", "#6391B5", BLUE, TEAL, "#176D63"]
    sns.barplot(
        data=data,
        x="Stage",
        y="Native AR (tok/s)",
        hue="Stage",
        palette=colors,
        legend=False,
        ax=ax,
    )
    ax.set_title("Correctness-valid single-request decode progression", loc="left", pad=10)
    ax.set_xlabel("")
    ax.set_ylim(0, 82)
    for patch, value in zip(ax.patches, data["Native AR (tok/s)"], strict=True):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 1.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.5,
        )
    finish(ax)
    save(fig, "decode_progress")


def prefill_progress() -> None:
    data = pd.DataFrame(
        {
            "Stage": [
                "Initial\nMFMA",
                "Down\ngeometry",
                "4-wave\nattention",
                "MHC\nselector",
                "Chunk\n2048",
                "1-wave\nattention",
                "Grid\nretune",
                "64-row\nsorter",
            ],
            "TTFT (s)": [3.445, 2.924, 2.608, 2.554, 2.421, 2.280, 2.184, 2.061],
        }
    )
    data["Input throughput (tok/s)"] = 4604 / data["TTFT (s)"]
    fig, ax = plt.subplots(figsize=(7.15, 3.45))
    sns.lineplot(
        data=data,
        x="Stage",
        y="Input throughput (tok/s)",
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        color=TEAL,
        ax=ax,
    )
    ax.fill_between(
        range(len(data)),
        data["Input throughput (tok/s)"],
        data["Input throughput (tok/s)"].min() - 80,
        color=TEAL,
        alpha=0.09,
    )
    ax.set_title("4604-token prefill optimization sequence", loc="left", pad=10)
    ax.set_xlabel("")
    ax.set_ylim(1200, 2350)
    for i, row in data.iterrows():
        ax.annotate(
            f"{row['Input throughput (tok/s)']:.0f}",
            (i, row["Input throughput (tok/s)"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=8,
        )
    finish(ax)
    save(fig, "prefill_progress")


def prefill_abba() -> None:
    data = pd.DataFrame(
        {
            "Run": ["A1\n32-row", "B1\n64-row", "B2\n64-row", "A2\n32-row"],
            "TTFT (s)": [2.184, 2.061, 2.062, 2.185],
            "Variant": ["32-row", "64-row", "64-row", "32-row"],
        }
    )
    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    sns.barplot(
        data=data,
        x="Run",
        y="TTFT (s)",
        hue="Variant",
        palette={"32-row": "#9AA9B6", "64-row": TEAL},
        dodge=False,
        ax=ax,
    )
    ax.set_title("Strict ABBA: M2048 expert sorter", loc="left", pad=10)
    ax.set_xlabel("")
    ax.set_ylim(1.95, 2.23)
    ax.legend(title="", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    bars = [patch for patch in ax.patches if patch.get_height() > 0]
    for patch in bars:
        value = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )
    finish(ax)
    save(fig, "prefill_abba")


def moe_microbenchmark() -> None:
    data = pd.DataFrame(
        {
            "Kernel": ["Gate/up", "Gate/up", "Down", "Down"],
            "Sorter block": ["32-row", "64-row", "32-row", "64-row"],
            "Latency (ms)": [7.26, 5.55, 6.01, 5.26],
        }
    )
    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    sns.barplot(
        data=data,
        x="Kernel",
        y="Latency (ms)",
        hue="Sorter block",
        palette={"32-row": "#9AA9B6", "64-row": BLUE},
        ax=ax,
    )
    ax.set_title("M2048 routed-MoE kernel latency", loc="left", pad=10)
    ax.set_xlabel("")
    ax.set_ylim(0, 8.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8.5, color=INK)
    ax.legend(title="", loc="upper right")
    finish(ax)
    save(fig, "moe_microbenchmark")


def main() -> None:
    decode_progress()
    prefill_progress()
    prefill_abba()
    moe_microbenchmark()
    print(f"Generated figures in {FIGURES} using font: {FONT}")


if __name__ == "__main__":
    main()
