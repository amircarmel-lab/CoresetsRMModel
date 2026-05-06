"""Plot Experiment results.

Reads every ``<method>_<dataset>.csv`` under ``results/csv/`` that
exists, produces:

  - ``results/exp_panels.png``       — 1 row × 3 datasets, one curve
                                         per method.
  - ``results/companion_table.csv``   — at requested_m=1000, mean
                                         across reps: construction time
                                         and queries per (dataset,
                                         method).

The plot reads the ``alg_tester`` column and prints a banner (stdout
only) indicating whether ALG-TESTER was ON or OFF when the data was
produced. Mixed states across CSVs trigger a warning.

"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV_DIR = Path(__file__).parent / "results" / "csv"
DEFAULT_OUT_DIR = Path(__file__).parent / "results"

METHODS = [
    "lightweight", "sensitivity", "bico",
    "ours_phi015",
]
DATASETS = ["census", "text", "image"]

STYLE = {
    "lightweight": {"color": "#9467bd", "label": "Lightweight", "marker": "v"},
    "sensitivity": {"color": "#2ca02c", "label": "Sensitivity", "marker": "D"},
    "bico":        {"color": "#d62728", "label": "BICO",        "marker": "P"},
    # Ours: base label is just "Ours". The plotter appends the
    # per-panel effective phi from the CSV, e.g. "Ours (φ=0.15)" on
    # census, "Ours (φ=0.42)" on image. See plot_panels().
    "ours_phi015": {"color": "#17becf", "label": "Ours",        "marker": "s"},
}

DATASET_TITLES = {
    "text":   "20-Newsgroups",
    "image":  "STL-10",
    "census": "Census 1990",
}
COMPANION_REFERENCE_M = 1000

MIN_CORESET_SIZE = 0


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _expected_row_count(test: bool) -> int:
    if test:
        n_reps, n_m = 2, 3
    else:
        n_reps, n_m = 10, 9     # 10 reps, 9 m values (200..1000)
    return n_reps * n_m


def load_all_csvs(csv_dir: Path, test: bool = False) -> pd.DataFrame:
    suffix = "_test" if test else ""
    expected = _expected_row_count(test)

    frames = []
    missing: list[str] = []
    partial: list[tuple[str, int]] = []
    no_tester_col: list[str] = []

    for method in METHODS:
        for dataset in DATASETS:
            name = f"{method}_{dataset}{suffix}.csv"
            path = csv_dir / name
            if not path.exists():
                missing.append(name)
                continue
            df = pd.read_csv(path)
            if "alg_tester" not in df.columns:
                no_tester_col.append(name)
                continue
            if len(df) < expected:
                partial.append((name, len(df)))
            frames.append(df)

    if no_tester_col:
        raise ValueError(
            "The following CSVs are missing the 'alg_tester' column "
            "and cannot be plotted:\n  - "
            + "\n  - ".join(no_tester_col)
            + "\nRegenerate them with the current run_exp.py."
        )

    if missing or partial:
        print("=" * 70)
        print(f"Completeness check (expected {expected} rows per CSV in "
              f"{'test' if test else 'full'} mode):")
        if missing:
            print(f"  missing {len(missing)} CSV(s):")
            for name in missing:
                print(f"    - {name}")
        if partial:
            print(f"  incomplete {len(partial)} CSV(s):")
            for name, n in partial:
                print(f"    - {name}  ({n}/{expected} rows)")
        print("Plotting continues with available data.")
        print("=" * 70)

    if not frames:
        raise FileNotFoundError(
            f"no CSVs matching '*{suffix}.csv' under {csv_dir}. "
            f"Run `python -m experiments.experiment_strong.run_exp"
            f"{' --test' if test else ''}` first."
        )
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Tester-state banner — stdout only.
# ----------------------------------------------------------------------
def print_tester_banner(df: pd.DataFrame) -> None:
    print("#" * 70)
    ours_rows = df[df["method"].astype(str).str.startswith("ours_")]
    states = ours_rows["alg_tester"].dropna().unique() if not ours_rows.empty else []

    if len(states) == 0:
        print("# ALG-TESTER: N/A (no ours_phi* rows in this dataset)")
    elif len(states) == 1:
        s = bool(states[0])
        print(f"# ALG-TESTER: {'ON' if s else 'OFF'}")
    else:
        print(f"# ALG-TESTER: MIXED {sorted(map(bool, states))}")
        print("# Some rows ran with the tester ON, others with it OFF.")
    print("#" * 70)


# ----------------------------------------------------------------------
# Panel plot: 1 × len(datasets)
# ----------------------------------------------------------------------
def plot_panels(df: pd.DataFrame, out_path: Path) -> None:
    if MIN_CORESET_SIZE > 0:
        # Clip on requested_m, not coreset_size: BICO returns varying
        df = df[df["requested_m"] >= MIN_CORESET_SIZE]
        if df.empty:
            raise ValueError(
                f"all rows have requested_m < MIN_CORESET_SIZE="
                f"{MIN_CORESET_SIZE}; nothing to plot."
            )

    datasets_present = [d for d in DATASETS if d in df["dataset"].unique()]
    n_cols = len(datasets_present)
    if n_cols == 0:
        raise ValueError("no data to plot after filtering.")

    fig, axes = plt.subplots(
        1, n_cols, figsize=(5.2 * n_cols, 4.6), squeeze=False,
    )
    axes = axes[0]

    for j, dataset in enumerate(datasets_present):
        ax = axes[j]
        sub = df[df["dataset"] == dataset]
        # Single-k plot — verify only one k value is present.
        ks_in_panel = sorted(int(k) for k in sub["k"].unique())
        k_label = ks_in_panel[0] if len(ks_in_panel) == 1 else f"mixed {ks_in_panel}"

        for method in METHODS:
            s = STYLE[method]
            m_sub = sub[sub["method"] == method]
            if m_sub.empty:
                continue
            # Group by REQUESTED m, not actual coreset_size — BICO's
            agg = m_sub.groupby("requested_m").agg(
                coreset_size_mean=("coreset_size", "mean"),
                ratio_mean=("cost_ratio", "mean"),
                ratio_std=("cost_ratio", "std"),
            ).reset_index().sort_values("requested_m")
            agg["ratio_std"] = agg["ratio_std"].fillna(0.0)

            # Build the legend label. For ours, append the per-panel
            # effective phi (read from the `phi` column — different
            # per dataset because text/image use cheap-only routing
            # at phi=phi_real, while census uses synthetic phi=0.15).
            label = s["label"]
            if method.startswith("ours"):
                phi_vals = m_sub["phi"].dropna().unique()
                if len(phi_vals) >= 1:
                    phi_for_panel = float(phi_vals[0])
                    label = rf"{s['label']} ($\varphi={phi_for_panel:.2f}$)"

            ax.plot(
                agg["coreset_size_mean"], agg["ratio_mean"],
                marker=s["marker"], lw=1.8, color=s["color"],
                label=label, markersize=5,
            )
            ax.fill_between(
                agg["coreset_size_mean"],
                agg["ratio_mean"] - agg["ratio_std"],
                agg["ratio_mean"] + agg["ratio_std"],
                alpha=0.15, color=s["color"],
            )

        ax.grid(True, which="both", alpha=0.25, ls="--", lw=0.5)
        ax.set_xlabel("Coreset size $m$", fontsize=10)
        if j == 0:
            ax.set_ylabel("Empirical worst-case cost ratio", fontsize=10)
        ax.set_title(
            f"{DATASET_TITLES.get(dataset, dataset)} ($k={k_label}$)",
            fontsize=11,
        )
        ax.set_xmargin(0.05)
        # then explicitly set the limits if you want asymmetry:
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0]+10, xlim[1] + 40)  # 50 extra units on the right
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Companion table
# ----------------------------------------------------------------------
def build_companion_table(
    df: pd.DataFrame, reference_m: int = COMPANION_REFERENCE_M,
) -> pd.DataFrame:
    """One row per (dataset, method) at requested_m = reference_m."""
    sub = df[df["requested_m"] == reference_m]
    if sub.empty:
        # Pick the closest requested_m per (dataset, method).
        def _pick_closest(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            g["_delta"] = (g["requested_m"] - reference_m).abs()
            min_delta = g["_delta"].min()
            return g[g["_delta"] == min_delta].drop(columns="_delta")

        sub = (
            df.groupby(["dataset", "method"], group_keys=False)
              .apply(_pick_closest)
        )

    agg = sub.groupby(["dataset", "method"]).agg(
        requested_m=("requested_m", "mean"),
        coreset_size=("coreset_size", "mean"),
        phi=("phi", "mean"),
        construction_time_s=("construction_time_s", "mean"),
        construction_distance_queries=("construction_distance_queries", "mean"),
        construction_quadruplet_queries=("construction_quadruplet_queries", "mean"),
        n_reps=("cost_ratio", "count"),
    ).reset_index()

    order_key = {m: i for i, m in enumerate(METHODS)}
    agg["_order"] = agg["method"].map(order_key)
    agg = agg.sort_values(["dataset", "_order"]).drop(columns="_order")
    return agg


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir", type=Path, default=DEFAULT_CSV_DIR,
        help="Directory containing the per-method CSVs.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Where to write the plot and companion table.",
    )
    parser.add_argument(
        "--reference-m", type=int, default=COMPANION_REFERENCE_M,
        help=f"Companion-table reference m. Default {COMPANION_REFERENCE_M}.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Plot test-mode CSVs and write outputs with _test suffix.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_csvs(args.csv_dir, test=args.test)
    suffix = "_test" if args.test else ""
    print(f"Loaded {len(df)} rows from {args.csv_dir} "
          f"(mode={'test' if args.test else 'full'})")

    print_tester_banner(df)

    panel_path = args.out_dir / f"exp_panels{suffix}.png"
    plot_panels(df, panel_path)
    print(f"Wrote {panel_path}")

    companion = build_companion_table(df, reference_m=args.reference_m)
    companion_path = args.out_dir / f"companion_table{suffix}.csv"
    companion.to_csv(companion_path, index=False)
    print(f"Wrote {companion_path}  ({len(companion)} rows)")

    print_tester_banner(df)


if __name__ == "__main__":
    main()