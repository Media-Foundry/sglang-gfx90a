#!/usr/bin/env python3
"""Generate publication figures from the validated experiment ledger."""

from __future__ import annotations

from pathlib import Path
import json
import statistics

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
DATA = json.loads((ROOT / "data/results.json").read_text())

BLUE = "#2F5D8C"
TEAL = "#2A8C82"
ORANGE = "#D17A3A"
RED = "#B44A4A"
INK = "#25313C"
MUTED = "#66727D"


def preferred_font() -> str:
    # Matplotlib's persistent font cache may predate a newly installed
    # msttcorefonts package. Register the real Arial files explicitly so a
    # report rebuild does not silently keep using the fontconfig fallback.
    arial_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for font_path in arial_dir.glob("[Aa]rial*.ttf"):
        font_manager.fontManager.addfont(font_path)
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
                "Aug. 24\ncheckpoint",
            ],
            "Native AR / HTTP (tok/s)": [14.6, 20.14, 25.54, 60.20, 66.10, 74.50],
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    colors = ["#A8B8C8", "#8AA7C1", "#6391B5", BLUE, TEAL, "#176D63"]
    sns.barplot(
        data=data,
        x="Stage",
        y="Native AR / HTTP (tok/s)",
        hue="Stage",
        palette=colors,
        legend=False,
        ax=ax,
    )
    ax.set_title("Historical TP4 single-request milestones (different checkpoints)", loc="left", pad=10)
    ax.set_xlabel("")
    ax.set_ylim(0, 82)
    for patch, value in zip(ax.patches, data["Native AR / HTTP (tok/s)"], strict=True):
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
    ax.set_title("Historical TP4 prefill: 4604-token single request", loc="left", pad=10)
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
    fig, ax = plt.subplots(figsize=(3.45, 2.75), layout='constrained')
    for i, row in data.iterrows():
        color = "#71879A" if row['Variant'] == '32-row' else TEAL
        ax.scatter(i, row['TTFT (s)'], color=color, s=45, zorder=3)
        ax.annotate(f"{row['TTFT (s)']:.3f}", (i, row['TTFT (s)']),
                    xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8.5)
    ax.set_xticks(range(len(data)), data['Run'])
    ax.set_ylabel('TTFT (s), expanded axis')
    ax.set_title("M2048 sorter: historical ABBA", loc="left", pad=10, fontsize=9)
    ax.set_xlabel("")
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(1.95, 2.23)
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
    fig, ax = plt.subplots(figsize=(3.45, 2.75), layout='constrained')
    sns.barplot(
        data=data,
        x="Kernel",
        y="Latency (ms)",
        hue="Sorter block",
        palette={"32-row": "#9AA9B6", "64-row": BLUE},
        ax=ax,
    )
    ax.set_title("M2048 gate/up and down", loc="left", pad=10, fontsize=9)
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
    tp8_pd()
    native_abba()
    consumer_abba()
    strict_dspark_history()
    print(f"Generated figures in {FIGURES} using font: {FONT}")


def tp8_pd() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.5), layout="constrained")
    records = []
    for cell in DATA['matrix']:
        c = cell['concurrency']
        for phase in ('prefill', 'decode'):
            for value in cell[phase]['rates']:
                records.append(dict(C=c, Phase=phase, Rate=value))
    frame = pd.DataFrame(records)
    for ax, phase, title, ylabel, color, ymax in zip(
        axes, ('prefill', 'decode'),
        ('(a) Prefill: 8K input, 1 output', '(b) Native decode: 512-token input'),
        ('Input tokens / s', 'Output tokens / s'), (TEAL, BLUE), (5900, 1500), strict=True
    ):
        sub = frame[frame.Phase == phase]
        sns.lineplot(data=sub, x='C', y='Rate', estimator='median', errorbar=None,
                     marker='o', linewidth=1.9, color=color, ax=ax,
                     label='Resident median' if phase == 'decode' else 'Wave median')
        for c, group in sub.groupby('C'):
            ax.vlines(c, group.Rate.min(), group.Rate.max(), color=color, lw=1)
            ax.scatter([c]*len(group), group.Rate, s=10, facecolors='white', edgecolors=color, zorder=4)
        if phase == 'decode':
            ax.plot([r['concurrency'] for r in DATA['matrix']],
                    [r['decode']['http_aggregate_tok_s'] for r in DATA['matrix']],
                    color=MUTED, ls='--', marker='s', markersize=3.5, label='Whole HTTP')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ['1', '2', '4', '8', '16', '32', '64'])
        ax.set(xlabel='Client concurrency C', ylabel=ylabel, ylim=(0, ymax))
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.legend(loc='upper left', fontsize=8)
        finish(ax)
    save(fig, 'tp8_pd_matrix')


def arm_points(ax, groups, ylabel, title, ylim):
    for i, (label, samples) in enumerate(groups):
        color = TEAL if label.startswith('B') else BLUE
        ax.scatter([i + (j-(len(samples)-1)/2)*.10 for j in range(len(samples))],
                   samples, s=27, color=color, alpha=.9, zorder=3)
        med = statistics.median(samples)
        ax.plot([i-.22,i+.22], [med,med], color=color, lw=2)
        ax.annotate(f'{med:.1f}', (i, max(samples)), xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set(xlim=(-.55,len(groups)-.45), ylim=ylim, ylabel=ylabel, xlabel='Service leg (A: control; B: candidate)')
    ax.set_title(title, loc='left', fontsize=9.5)
    finish(ax)


def native_abba() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), layout='constrained')
    arm_points(axes[0], [(r['label'],r['rates']) for r in DATA['long8k_abba']],
               'Resident output tokens / s', '(a) Empty C4 tiles: C32, 8K input', (0,710))
    arm_points(axes[1], [(r['label'].removeprefix('C1'),r['rates']) for r in DATA['c1_gemv_abba']['arms']],
               'Resident output tokens / s', '(b) Restored C1 wo_a GEMV', (74,93))
    axes[0].text(.5,.70,'3.60x resident speedup',transform=axes[0].transAxes,ha='center',fontsize=9,color=TEAL)
    axes[1].text(.5,.91,'+10.32%; expanded y-axis',transform=axes[1].transAxes,ha='center',fontsize=8.5,color=TEAL)
    save(fig, 'native_abba')


def consumer_abba() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), layout='constrained')
    for ax, key, title, limits in zip(axes, ('down_consumer','c1_isolation'),
                                    ('(a) C32: scoped down consumer', '(b) C1: unchanged kernel / isolation'),
                                    ((1032,1076),(89.2,90.55)), strict=True):
        result=DATA[key]
        arm_points(ax, [(label,r['waves_tok_s']) for label,r in result['arms'].items()],
                   'Resident output tokens / s',title,limits)
        note='+1.54% in this small screen' if key=='down_consumer' else 'No attributed C1 speedup'
        ax.text(.5,.91,note,transform=ax.transAxes,ha='center',fontsize=8.5,color=TEAL if key=='down_consumer' else MUTED)
    fig.supxlabel('Expanded y-axes; two natural-EOS code waves per leg', fontsize=8, color=MUTED)
    save(fig, 'down_consumer_abba')


def strict_dspark_history() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), layout='constrained')
    colors=[BLUE,TEAL,ORANGE]
    for i,(row,color) in enumerate(zip(DATA['historical_dspark'],colors,strict=True)):
        ax.hlines(i,row['low'],row['high'],color=color,lw=2)
        ax.scatter(row['resident'],i,color=color,s=48,zorder=3)
        ax.annotate(f"{row['resident']:.2f}   |   accept {row['accept']:.3f}",
                    (row['resident'],i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8.5)
    ax.set_yticks([0,1,2],[r['family'] for r in DATA['historical_dspark']])
    ax.set(xlim=(1040,1180),ylim=(-.45,2.55),xlabel='Historical resident output tokens / s (not native AR)')
    ax.set_title('Strict TP8 DSpark, C32: September 11 evidence',loc='left',pad=12)
    ax.invert_yaxis()
    finish(ax,ygrid=False)
    save(fig,'strict_dspark_history')


if __name__ == "__main__":
    main()
