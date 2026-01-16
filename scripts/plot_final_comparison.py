#!/usr/bin/env python3
"""
Plots focused comparisons for the "final" experiments (typically `results/final/*`).

Outputs:
  - Grid of F1 vs round for multiple k values
  - Detailed curves for k=10 (F1, precision, recall, benign_fpr, threshold)
  - Communication cost (bytes_total) per round and cumulative
  - A CSV + stdout summary (best and last metrics)

Example:
  ./.venv/bin/python scripts/plot_final_comparison.py --runs_dir results/final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _read_f1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "granularity" in df.columns:
        df = df[df["granularity"].astype(str) == "flow"].copy()
    df["round"] = df["round"].astype(int)
    df["k"] = df["k"].astype(int)
    df = df.sort_values(["k", "round"]).reset_index(drop=True)
    return df


def _read_comm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["round"] = df["round"].astype(int)
    df = df.sort_values("round").reset_index(drop=True)
    return df


def _pick_runs(runs_dir: Path, *, include: list[str] | None) -> list[Path]:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if include:
        wanted = set(include)
        candidates = [p for p in candidates if p.name in wanted]
    # keep only those with f1_scores.csv
    candidates = [p for p in candidates if (p / "f1_scores.csv").exists()]
    return sorted(candidates, key=lambda p: p.name)


def _color_map(names: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


def _annotate_best_and_last(ax, x, y, *, color: str) -> None:
    if len(x) == 0:
        return
    best_idx = int(np.argmax(y))
    ax.scatter([x[best_idx]], [y[best_idx]], marker="*", s=140, color=color, zorder=6)
    ax.annotate(
        f"best {y[best_idx]:.3f}",
        (x[best_idx], y[best_idx]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color=color,
    )
    ax.annotate(
        f"last {y[-1]:.3f}",
        (x[-1], y[-1]),
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=9,
        color=color,
    )


def plot_f1_grid(
    *,
    runs: dict[str, pd.DataFrame],
    ks: list[int],
    out_path: Path,
    title: str,
) -> None:
    n = len(ks)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.6 * rows), sharex=True, sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    colors = _color_map(list(runs.keys()))

    for i, k in enumerate(ks):
        ax = axes[i // cols, i % cols]
        for run_name, df in runs.items():
            sub = df[df["k"] == k].sort_values("round")
            if sub.empty:
                continue
            x = sub["round"].to_numpy()
            y = sub["f1_score"].to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=2.2,
                label=run_name,
                color=colors[run_name],
            )
            _annotate_best_and_last(ax, x, y, color=colors[run_name])

        ax.set_title(f"F1 vs round (Top-{k})")
        ax.set_xlabel("Round")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    # Hide any unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(title, y=1.01, fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=1, frameon=True, framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_k10_details(*, runs: dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    colors = _color_map(list(runs.keys()))
    metrics = [
        ("f1_score", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("benign_fpr", "Benign FPR"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.reshape(2, 2)

    for (metric, label), ax in zip(metrics, axes.ravel()):
        for run_name, df in runs.items():
            sub = df[(df["k"] == 10)].sort_values("round")
            if sub.empty or metric not in sub.columns:
                continue
            x = sub["round"].to_numpy()
            y = sub[metric].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", markersize=4, linewidth=2.2, label=run_name, color=colors[run_name])
            if metric == "f1_score":
                _annotate_best_and_last(ax, x, y, color=colors[run_name])
        ax.set_title(f"{label} (Top-10)")
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.01, fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=1, frameon=True, framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_and_fpr(*, runs: dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    colors = _color_map(list(runs.keys()))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for run_name, df in runs.items():
        sub = df[df["k"] == 10].sort_values("round")
        if sub.empty:
            continue
        x = sub["round"].to_numpy()
        if "threshold" in sub.columns:
            ax1.plot(
                x,
                sub["threshold"].to_numpy(dtype=float),
                marker="o",
                markersize=4,
                linewidth=2.2,
                color=colors[run_name],
                label=f"{run_name} threshold",
                alpha=0.9,
            )
        if "benign_fpr" in sub.columns:
            ax2.plot(
                x,
                sub["benign_fpr"].to_numpy(dtype=float),
                marker="s",
                markersize=3.8,
                linewidth=2.0,
                color=colors[run_name],
                label=f"{run_name} benign_fpr",
                alpha=0.6,
                linestyle="--",
            )

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Threshold (chosen by f1_max)")
    ax2.set_ylabel("Benign FPR")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    # One combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_communication(*, comm_runs: dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    colors = _color_map(list(comm_runs.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax, axc = axes

    for run_name, df in comm_runs.items():
        x = df["round"].to_numpy()
        y = df["bytes_total"].to_numpy(dtype=float)
        ax.plot(x, y / 1e6, marker="o", markersize=4, linewidth=2.2, label=run_name, color=colors[run_name])
        axc.plot(x, np.cumsum(y) / 1e9, marker="o", markersize=4, linewidth=2.2, label=run_name, color=colors[run_name])

    ax.set_title("Bytes per round")
    ax.set_xlabel("Round")
    ax.set_ylabel("MB")
    ax.grid(True, alpha=0.25)

    axc.set_title("Cumulative communication")
    axc.set_xlabel("Round")
    axc.set_ylabel("GB")
    axc.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.02, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=1, frameon=True, framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarize_runs(
    *,
    runs: dict[str, pd.DataFrame],
    comm_runs: dict[str, pd.DataFrame],
    out_csv: Path,
) -> pd.DataFrame:
    rows = []
    for run_name, df in runs.items():
        k10 = df[df["k"] == 10].sort_values("round")
        if k10.empty:
            continue

        best_idx = k10["f1_score"].idxmax()
        best = k10.loc[best_idx]
        last = k10.iloc[-1]

        comm = comm_runs.get(run_name)
        comm_sum = float(comm["bytes_total"].sum()) if comm is not None else np.nan
        rows.append(
            {
                "run": run_name,
                "best_f1@10": float(best["f1_score"]),
                "best_round@10": int(best["round"]),
                "last_f1@10": float(last["f1_score"]),
                "last_round": int(last["round"]),
                "best_precision@10": float(best.get("precision", np.nan)),
                "best_recall@10": float(best.get("recall", np.nan)),
                "best_benign_fpr@10": float(best.get("benign_fpr", np.nan)),
                "best_threshold@10": float(best.get("threshold", np.nan)),
                "comm_bytes_sum": comm_sum,
                "comm_gb_sum": comm_sum / 1e9 if comm_sum == comm_sum else np.nan,  # NaN-safe
            }
        )

    out = pd.DataFrame(rows).sort_values("best_f1@10", ascending=False).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> int:
    _setup_style()

    parser = argparse.ArgumentParser(description="Plot comparisons for final runs.")
    parser.add_argument("--runs_dir", type=str, default="results/final", help="Directory containing runs.")
    parser.add_argument("--out_dir", type=str, default="results/final/_plots", help="Where to save plots.")
    parser.add_argument("--include", nargs="*", default=None, help="Optional list of run directory names to include.")
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5, 10], help="Top-k values to plot.")
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="f1_max",
        choices=["f1_max", "fpr_target"],
        help="Which evaluated CSV to use (default: f1_max).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    run_paths = _pick_runs(runs_dir, include=args.include)
    if not run_paths:
        raise SystemExit(f"No runs with f1_scores.csv found in: {runs_dir}")

    runs: dict[str, pd.DataFrame] = {}
    comm_runs: dict[str, pd.DataFrame] = {}
    for run_path in run_paths:
        name = run_path.name
        suffix = "" if args.threshold_mode == "f1_max" else f"_{args.threshold_mode}"
        f1_path = run_path / f"f1_scores{suffix}.csv"
        if not f1_path.exists():
            # Keep going; useful when only a subset has operational evaluation.
            continue
        runs[name] = _read_f1(f1_path)
        comm_path = run_path / "communication_metrics.csv"
        if comm_path.exists():
            comm_runs[name] = _read_comm(comm_path)

    title = f"Final runs comparison ({runs_dir})"
    if not runs:
        raise SystemExit(f"No runs with f1_scores{suffix}.csv found under: {runs_dir}")

    tag = "oracle" if args.threshold_mode == "f1_max" else args.threshold_mode
    plot_f1_grid(runs=runs, ks=args.ks, out_path=out_dir / f"f1_grid_{tag}.png", title=title)
    plot_k10_details(runs=runs, out_path=out_dir / f"k10_details_{tag}.png", title=title)
    plot_threshold_and_fpr(
        runs=runs,
        out_path=out_dir / f"k10_threshold_fpr_{tag}.png",
        title=f"{title} — k=10 threshold vs benign FPR ({tag})",
    )
    if comm_runs:
        plot_communication(comm_runs=comm_runs, out_path=out_dir / "communication.png", title=f"{title} — communication")

    summary = summarize_runs(runs=runs, comm_runs=comm_runs, out_csv=out_dir / f"summary_{tag}.csv")
    print("\nSummary (sorted by best_f1@10):")
    with pd.option_context("display.max_columns", 50, "display.width", 160):
        print(summary)

    print(f"\nSaved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
