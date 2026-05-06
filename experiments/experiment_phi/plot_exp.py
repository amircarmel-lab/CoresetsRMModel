"""Plot Experiment results.

Reads ``results/csv/text.csv`` and ``results/csv/image.csv`` (or the
``_test`` variants when ``--test``). Produces a one-row figure with one
panel per dataset; each panel has TWO curves, one per oracle_kind:

  interpolated  — MixedQuadrupletOracle (cheap/expensive routing).
  synthetic     — EuclideanQuadrupletOracle (i.i.d. flips on expensive).

Mean line + ±1 std band across reps.

When ``SHOW_REGRESSION`` is True (default), a dashed linear-regression
line is overlaid per (dataset, oracle_kind) and the slope is appended
to the curve's legend label.

The ``alg_tester`` and ``oracle_kind`` columns are required.
A banner indicating the tester state is printed to stdout (only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV_DIR = Path(__file__).parent / "results" / "csv"
DEFAULT_OUT_DIR = Path(__file__).parent / "results"

DATASETS = ["text", "image"]
DATASET_TITLES = {"text": "20-Newsgroups", "image": "STL-10"}

ORACLE_KINDS = ["interpolated", "synthetic"]
ORACLE_STYLE = {
    "interpolated": {"color": "#1f77b4", "marker": "^",
                      "label": "Interpolated"},
    "synthetic":    {"color": "#d62728", "marker": "o",
                      "label": "Synthetic"},
}

# Overlay a dashed linear-regression line per (dataset, oracle_kind).
# Slope is computed from the per-rep cloud (NOT per-phi means) and
# appended to the legend label. Set to False to hide.
SHOW_REGRESSION = True

# Shared y-axis range across both panels so the two datasets are on
# the same visual scale.
Y_LIM = (0.0, 0.2)


REQUIRED_COLS = ("alg_tester", "oracle_kind")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_all_csvs(
    csv_dir: Path, datasets: list[str], test: bool = False,
) -> pd.DataFrame:
    suffix = "_test" if test else ""

    frames = []
    missing = []
    no_required = []
    for dataset in datasets:
        name = f"{dataset}{suffix}.csv"
        path = csv_dir / name
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path)
        skip = False
        for col in REQUIRED_COLS:
            if col not in df.columns:
                no_required.append((name, col))
                skip = True
                break
        if not skip:
            frames.append(df)

    if no_required:
        msg_lines = [f"  - {name}: missing column '{col}'"
                     for name, col in no_required]
        raise ValueError(
            "The following CSVs are missing required columns and "
            "cannot be plotted:\n" + "\n".join(msg_lines)
            + "\nRegenerate them with the current run_exp.py."
        )

    if missing:
        print("=" * 70)
        print(f"Missing CSV(s) for suffix={suffix!r}:")
        for name in missing:
            print(f"  - {name}")
        print("Plotting continues with available data.")
        print("=" * 70)

    if not frames:
        raise FileNotFoundError(
            f"no CSVs matching '*{suffix}.csv' under {csv_dir}."
        )
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Banners (stdout only)
# ----------------------------------------------------------------------
def print_tester_banner(df: pd.DataFrame) -> None:
    print("#" * 70)
    states = df["alg_tester"].dropna().unique()
    if len(states) == 1:
        s = bool(states[0])
        print(f"# ALG-TESTER: {'ON' if s else 'OFF'}")
    elif len(states) == 0:
        print("# ALG-TESTER: N/A (no rows)")
    else:
        print(f"# ALG-TESTER: MIXED {sorted(map(bool, states))}")
        print("# Some rows ran with the tester ON, others OFF.")
    print("#" * 70)


def print_coreset_size_banner(df: pd.DataFrame) -> None:
    print("#" * 70)
    print("# Coreset size per dataset (intentionally not on plot panels):")
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        m_values = sub["m"].unique()
        k_values = sub["k"].unique()
        n_values = sub["n"].unique()
        if len(m_values) == 1 and len(k_values) == 1 and len(n_values) == 1:
            print(f"#   {dataset:<6s}  n={int(n_values[0]):<6d}  "
                  f"k={int(k_values[0]):<3d}  "
                  f"coreset_size={int(m_values[0])}")
        else:
            print(f"#   {dataset}: MIXED m={list(m_values)} "
                  f"k={list(k_values)} n={list(n_values)}")
    print("#" * 70)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_panels(df: pd.DataFrame, out_path: Path) -> None:
    datasets_present = [d for d in DATASETS if d in df["dataset"].unique()]
    if not datasets_present:
        raise ValueError("no data to plot")

    n_panels = len(datasets_present)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 4.6), squeeze=False,
    )
    axes = axes[0]

    for ax, dataset in zip(axes, datasets_present):
        sub_ds = df[df["dataset"] == dataset]

        for oracle_kind in ORACLE_KINDS:
            sub = sub_ds[sub_ds["oracle_kind"] == oracle_kind]
            if sub.empty:
                continue

            # Per-phi: mean across reps + sample std (ddof=1). Std is
            # NaN when only one rep is present at a phi; fillna(0) so
            # fill_between collapses to a line instead of leaving a hole.
            agg = sub.groupby("phi_target").agg(
                ratio_mean=("cost_ratio", "mean"),
                ratio_std=("cost_ratio", "std"),
                n_reps=("cost_ratio", "count"),
            ).reset_index().sort_values("phi_target")
            agg["ratio_std"] = agg["ratio_std"].fillna(0.0)

            style = ORACLE_STYLE[oracle_kind]

            # Linear regression on the per-rep cloud — uses every
            # individual (phi_target, cost_ratio) point, not just the
            # per-phi means, so the fit reflects the whole sample.
            slope, x_line, y_line = None, None, None
            if SHOW_REGRESSION:
                x_all = sub["phi_target"].to_numpy(dtype=float)
                y_all = sub["cost_ratio"].to_numpy(dtype=float)
                if len(np.unique(x_all)) >= 2:
                    slope, intercept = np.polyfit(x_all, y_all, 1)
                    x_line = np.array([x_all.min(), x_all.max()])
                    y_line = slope * x_line + intercept

            label = (
                f"{style['label']} (slope={slope:.2f})"
                if slope is not None else style["label"]
            )

            ax.plot(
                agg["phi_target"], agg["ratio_mean"],
                marker=style["marker"], lw=2, color=style["color"],
                markersize=7, label=label,
            )
            ax.fill_between(
                agg["phi_target"],
                agg["ratio_mean"] - agg["ratio_std"],
                agg["ratio_mean"] + agg["ratio_std"],
                alpha=0.18, color=style["color"],
            )
            if slope is not None:
                ax.plot(
                    x_line, y_line,
                    ls="--", lw=1.2, color=style["color"], alpha=0.7,
                )

        ax.set_xlabel(
            r"Quadruplet-oracle noise rate $\varphi$",
            fontsize=11,
        )
        ax.set_ylabel("Empirical worst-case cost ratio", fontsize=11)
        ax.set_title(DATASET_TITLES.get(dataset, dataset), fontsize=11)
        ax.set_ylim(*Y_LIM)
        ax.grid(True, which="both", alpha=0.25, ls="--", lw=0.5)
        ax.legend(loc="best", fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=DATASETS)
    parser.add_argument("--test", action="store_true",
                        help="Plot test-mode CSVs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets if args.datasets else list(DATASETS)
    df = load_all_csvs(args.csv_dir, datasets, test=args.test)
    suffix = "_test" if args.test else ""
    print(f"Loaded {len(df)} rows from {args.csv_dir} "
          f"(mode={'test' if args.test else 'full/semi'})")

    print_tester_banner(df)
    print_coreset_size_banner(df)

    out_path = args.out_dir / f"exp_phi_panels{suffix}.png"
    plot_panels(df, out_path)
    print(f"Wrote {out_path}")

    print_tester_banner(df)
    print_coreset_size_banner(df)


if __name__ == "__main__":
    main()