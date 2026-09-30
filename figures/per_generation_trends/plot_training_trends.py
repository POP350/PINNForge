"""Plot measured PINN metrics across generations.

Usage: python plot_training_trends.py INPUT.csv [OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METRICS = (
    ("training_loss", "Training loss", "#254B70", "o"),
    ("pde_residual", "PDE residual", "#D18737", "s"),
    ("reference_mse", "Reference MSE", "#4F8A7B", "D"),
)
PROBLEMS = (
    ("burgers_1d", "Burgers · 1D"),
    ("heat_2d_multiscale", "Multiscale heat · 2D"),
    ("poisson_5d", "Poisson · 5D"),
    ("navier_stokes_2d_C", "Navier–Stokes · 2D C"),
)
ILLUSTRATIVE_NS_TRAINING_LOSS = (
    0.003414, 0.002650, 0.002020, 0.001510, 0.001130,
    0.000850, 0.000640, 0.000480, 0.000360, 0.000270,
)


def make_figure(
    data: pd.DataFrame, *, illustrative_ns_loss: bool = False,
    illustrative_ns_mse_mode: str | None = None,
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#85919C",
            "axes.labelcolor": "#344250",
            "xtick.color": "#536170",
            "ytick.color": "#536170",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        ("PINN metrics across generations · illustrative adjustment"
         if illustrative_ns_loss or illustrative_ns_mse_mode else "PINN metrics across generations"),
        x=0.08, y=0.976, ha="left", va="top", fontsize=17, weight="bold", color="#20364A"
    )
    fig.text(
        0.08, 0.922,
        ("Dashed Navier–Stokes curves are illustrative · other values recorded · log scale"
         if illustrative_ns_mse_mode else
         "Dashed Navier–Stokes loss is illustrative · other values recorded · log scale"
         if illustrative_ns_loss else
         "Measured values · log scale · one selected candidate per generation"),
        fontsize=9.5, color="#657483",
    )

    for panel_index, (axis, (problem, title)) in enumerate(zip(axes.flat, PROBLEMS)):
        subset = data.loc[data["problem"].eq(problem)]
        generations = subset.loc[subset["stage"].eq("generation")].sort_values("generation")
        high_fidelity = subset.loc[subset["stage"].eq("final_high_fidelity")]
        axis.set_facecolor("#FCFDFE")
        axis.axvspan(10.25, 11.45, color="#F3F5F7", zorder=0)
        axis.axvline(10.05, color="#D7DDE3", linewidth=0.85, linestyle=(0, (2, 3)))

        all_values = []
        for metric, _label, color, marker in METRICS:
            values = generations[metric].to_numpy(dtype=float)
            all_values.extend(values)
            axis.plot(
                generations["generation"], values,
                color=color, linewidth=2.0, marker=marker, markersize=4.4,
                linestyle=(0, (4, 2)) if problem == "navier_stokes_2d_C" and (
                    (illustrative_ns_loss and metric == "training_loss") or
                    (illustrative_ns_mse_mode is not None and metric == "reference_mse")
                ) else "-",
                markerfacecolor="white", markeredgewidth=1.3, solid_capstyle="round",
                zorder=3,
            )
            if len(high_fidelity):
                value = float(high_fidelity.iloc[0][metric])
                all_values.append(value)
                axis.scatter(
                    [10.85], [value], s=59, color=color, marker=marker,
                    edgecolor="white", linewidth=0.9, zorder=5,
                )

        axis.set_yscale("log")
        lo, hi = min(all_values), max(all_values)
        axis.set_ylim(10 ** (np.log10(lo) - 0.22), 10 ** (np.log10(hi) + 0.25))
        axis.set_xlim(-0.35, 11.5)
        axis.set_xticks([0, 2, 4, 6, 8, 10.85], ["0", "2", "4", "6", "8", "HF"])
        axis.grid(axis="y", color="#E6EAEE", linewidth=0.75, zorder=0)
        axis.tick_params(axis="both", length=0, pad=6)
        axis.set_title(
            f"{chr(97 + panel_index)}   {title}", loc="left", pad=13,
            fontsize=11.5, weight="bold", color="#20364A",
        )
        if (illustrative_ns_loss or illustrative_ns_mse_mode) and problem == "navier_stokes_2d_C":
            axis.text(0.98, 0.96,
                      "Blue + green: illustrative" if illustrative_ns_mse_mode else "Blue line: illustrative",
                      transform=axis.transAxes, ha="right", va="top",
                      fontsize=8, color="#254B70")
        if panel_index >= 2:
            axis.set_xlabel("Generation", labelpad=7)
        if panel_index % 2 == 0:
            axis.set_ylabel("Metric value (log scale)", labelpad=8)

    handles = [
        Line2D([0], [0], color=color, marker=marker, markerfacecolor="white",
               linewidth=2, markersize=5, label=label)
        for _, label, color, marker in METRICS
    ]
    fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.93, 0.94),
        ncol=3, frameon=False, columnspacing=2.0, handlelength=2.1, fontsize=9,
    )
    note = (
        ("Illustrative: Navier–Stokes loss (0–9) and MSE (1–8); MSE falls rapidly, then plateaus. "
         "MSE endpoints and HF are recorded."
         if illustrative_ns_mse_mode == "plateau" else
         "Illustrative: Navier–Stokes loss (generations 0–9) and MSE (generations 1–8). "
         "MSE endpoints and HF are recorded.")
        if illustrative_ns_mse_mode else
        "Illustrative adjustment: Navier–Stokes training loss, generations 0–9 only. "
        "Other values are recorded; HF is a separate run."
        if illustrative_ns_loss else
        "HF: separate high-fidelity run of a selected candidate; points are not connected to generation 9."
        "   Lines connect recorded values only."
    )
    fig.text(0.08, 0.032, note, fontsize=8, color="#687582")
    fig.subplots_adjust(left=0.09, right=0.94, top=0.82, bottom=0.12, hspace=0.41, wspace=0.27)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path(__file__).parent)
    parser.add_argument("--illustrative-ns-loss", action="store_true",
                        help="Replace only Navier–Stokes generation training-loss values with labeled illustrative values")
    parser.add_argument("--illustrative-ns-concave-mse", action="store_true",
                        help="Make Navier–Stokes MSE concave on the log-scale plot, preserving endpoints")
    parser.add_argument("--illustrative-ns-plateau-mse", action="store_true",
                        help="Make Navier–Stokes MSE fall rapidly, then plateau on the log-scale plot")
    args = parser.parse_args()
    data = pd.read_csv(args.input_csv)
    required = {"problem", "generation", "stage", "training_loss", "pde_residual", "reference_mse"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if set(data["problem"]) != {name for name, _ in PROBLEMS}:
        raise ValueError("The input contains unexpected or missing problem groups")
    if data.duplicated(["problem", "stage", "generation"]).any():
        raise ValueError("Duplicate problem/stage/generation rows")
    for metric, *_ in METRICS:
        if not np.isfinite(data[metric]).all() or (data[metric] <= 0).any():
            raise ValueError(f"{metric} must contain finite, positive values for a log scale")

    if args.illustrative_ns_loss:
        data = data.copy()
        data["training_loss_origin"] = "recorded"
        mask = data["problem"].eq("navier_stokes_2d_C") & data["stage"].eq("generation")
        if sorted(data.loc[mask, "generation"].tolist()) != list(range(10)):
            raise ValueError("Expected Navier–Stokes generations 0–9")
        for generation, value in enumerate(ILLUSTRATIVE_NS_TRAINING_LOSS):
            row = mask & data["generation"].eq(generation)
            data.loc[row, "training_loss"] = value
            data.loc[row, "training_loss_origin"] = "illustrative_adjustment"

    if args.illustrative_ns_concave_mse and args.illustrative_ns_plateau_mse:
        raise ValueError("Choose only one illustrative Navier–Stokes MSE shape")
    mse_mode = (
        "plateau" if args.illustrative_ns_plateau_mse else
        "concave" if args.illustrative_ns_concave_mse else None
    )
    if mse_mode:
        data = data.copy()
        data["reference_mse_origin"] = "recorded"
        mask = data["problem"].eq("navier_stokes_2d_C") & data["stage"].eq("generation")
        if sorted(data.loc[mask, "generation"].tolist()) != list(range(10)):
            raise ValueError("Expected Navier–Stokes generations 0–9")
        start = float(data.loc[mask & data["generation"].eq(0), "reference_mse"].iloc[0])
        end = float(data.loc[mask & data["generation"].eq(9), "reference_mse"].iloc[0])
        for generation in range(1, 9):
            t = generation / 9
            progress = (
                (1 - np.exp(-3.6 * t)) / (1 - np.exp(-3.6))
                if mse_mode == "plateau" else
                0.35 * t + 0.65 * t * t
            )
            value = float(np.exp(np.log(start) + (np.log(end) - np.log(start)) * progress))
            row = mask & data["generation"].eq(generation)
            data.loc[row, "reference_mse"] = value
            data.loc[row, "reference_mse_origin"] = "illustrative_adjustment"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(
        data, illustrative_ns_loss=args.illustrative_ns_loss,
        illustrative_ns_mse_mode=mse_mode,
    )
    stem = args.output_dir / "pinn_generation_trends"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    data.to_csv(args.output_dir / "plotted_source_data.csv", index=False)
    print(f"Saved {stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
