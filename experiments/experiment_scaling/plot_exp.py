"""Plot Experiment results.

Reads ``census_<mode>_exp<E>[_test].csv`` from ``results/csv``. Produces:

  - ``exp_panels_exp<E>[_test].png``         — log x, linear y
  - ``exp_panels_exp<E>[_test]_loglog.png``  — log x, log y (slopes)

The x-axis is ALWAYS log-scaled because the n grid is geometric
(BASE_N · 2^i); a linear x compresses the small-n end into a sliver.
The companion log-log PNG additionally puts y on log so scaling
exponents read off as straight-line slopes.

Each PNG is a 1x2 grid (cols = modes), one curve per k inside each
panel. y is post-Alg-G distance queries; x is n. Mean line + ±1 std
band across reps. Both axes use a "K"/"M" tick format
(15625 → "16K", 1_000_000 → "1M") so labels stay short.

The ``alg_tester`` and ``mode`` columns are required — plot crashes if
absent. A banner indicating the tester state and per-panel
alg_g_size range is printed to stdout (only).

Usage
-----
::

    python -m experiments.experiment_scaling.plot_exp --exp 7
    python -m experiments.experiment_scaling.plot_exp --exp 5 --modes semi
    python -m experiments.experiment_scaling.plot_exp --test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


DEFAULT_CSV_DIR = Path(__file__).parent / "results" / "csv"
DEFAULT_OUT_DIR = Path(__file__).parent / "results"

DATASET_NAME = "census"
MODES = ["semi", "full"]

# k -> color/marker. Extends easily if K_GRID grows.
K_COLORS = {
    5:   "#440154",
    10:  "#3b528b",
    20:  "#2a788e",
    25:  "#21918c",
    50:  "#5ec962",
    100: "#fde725",
}
K_MARKERS = {5: "o", 10: "s", 20: "v", 25: "^", 50: "D", 100: "v"}


REQUIRED_COLS = (
    "mode", "n", "k", "rep",
    "construction_distance_queries",
    "alg_g_size", "alg_g_was_reused",
    "alg_tester",
)


# ----------------------------------------------------------------------
# Tick formatting helpers
# ----------------------------------------------------------------------
def _format_thousands(v: float) -> str:
    """Format a numeric value as '12K' / '1.2M' / '500'.

    Integers in the target unit are rendered without a decimal
    point: 1_000_000 -> "1M", 15_625 -> "15.6K", 166_000 -> "166K".
    """
    v = float(v)
    if v >= 1_000_000:
        out = v / 1_000_000
        s = f"{out:.1f}M"
        return s.replace(".0M", "M")
    if v >= 1_000:
        out = v / 1_000
        s = f"{out:.1f}K"
        return s.replace(".0K", "K")
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.2f}"


def _kfmt(x, _pos=None) -> str:
    """matplotlib FuncFormatter wrapper for ``_format_thousands``."""
    return _format_thousands(x)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def csv_filename(mode: str, exp: int, test: bool) -> str:
    if test:
        return f"{DATASET_NAME}_{mode}_test.csv"
    return f"{DATASET_NAME}_{mode}_exp{exp}.csv"


def load_all_csvs(
    csv_dir: Path,
    modes: list[str],
    exp: int,
    test: bool,
) -> pd.DataFrame:
    """Concatenate every census_<mode>_exp<E>[_test].csv that exists."""
    frames = []
    missing: list[str] = []
    no_required_col: list[tuple[str, str]] = []

    for mode in modes:
        name = csv_filename(mode, exp, test)
        path = csv_dir / name
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path)
        for col in REQUIRED_COLS:
            if col not in df.columns:
                no_required_col.append((name, col))
                break
        else:
            frames.append(df)

    if no_required_col:
        msg = "\n".join(
            f"  - {name}: missing column '{col}'"
            for name, col in no_required_col
        )
        raise ValueError(
            "The following CSVs are missing required columns and "
            "cannot be plotted:\n" + msg
            + "\nRegenerate them with the current run_exp.py."
        )

    if missing:
        print("=" * 70)
        print("Missing CSV(s):")
        for name in missing:
            print(f"  - {name}")
        print("Plotting continues with available data.")
        print("=" * 70)

    if not frames:
        raise FileNotFoundError(
            f"no CSVs found under {csv_dir} matching the requested "
            f"(modes={modes}, exp={exp}, test={test})."
        )
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Stdout banner
# ----------------------------------------------------------------------
def print_banner(df: pd.DataFrame, suffix: str) -> None:
    print("#" * 70)
    print(f"# Plot suffix: {suffix!r}")

    states = df["alg_tester"].dropna().unique()
    if len(states) == 1:
        s = bool(states[0])
        print(f"# ALG-TESTER: {'ON' if s else 'OFF'}")
    elif len(states) == 0:
        print("# ALG-TESTER: N/A (no rows)")
    else:
        print(f"# ALG-TESTER: MIXED {sorted(map(bool, states))}")
        print("# Some rows ran with the tester ON, others OFF.")

    print("# Per-panel summary:")
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        ks = sorted(int(k) for k in sub["k"].unique())
        ns = sorted(int(n) for n in sub["n"].unique())
        n_max = int(sub["n"].max())
        n_reps = int(sub.groupby(["k", "n"])["rep"].nunique().max())
        sizes = sub["alg_g_size"].astype(int)
        print(
            f"#   {mode:<4s}  n_max={n_max:<8d}  k={ks}  "
            f"n_grid={ns}  reps≤{n_reps}  "
            f"|C| in [{int(sizes.min())}, {int(sizes.max())}]"
        )
    print("#" * 70)


# ----------------------------------------------------------------------
# Per-panel aggregation
# ----------------------------------------------------------------------
def _aggregate_for_k(panel_sub: pd.DataFrame, k: int) -> pd.DataFrame:
    k_sub = panel_sub[panel_sub["k"] == k]
    if k_sub.empty:
        return k_sub
    # Per-n: mean across reps + sample std (ddof=1). Std is NaN when
    # only one rep is present at a given n; fillna(0) so fill_between
    # collapses to the line rather than producing a hole.
    agg = k_sub.groupby("n").agg(
        dq_mean=("construction_distance_queries", "mean"),
        dq_std=("construction_distance_queries", "std"),
        n_reps=("rep", "nunique"),
    ).reset_index().sort_values("n")
    agg["dq_std"] = agg["dq_std"].fillna(0.0)
    return agg


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_panels(df: pd.DataFrame, out_path: Path, *, log_y: bool) -> None:
    """Render the (modes) panel grid.

    The x-axis is ALWAYS log-scaled — the n grid is geometric
    (BASE_N · 2^i), so a linear x squashes the small-n values into a
    sliver. ``log_y`` toggles the y-axis between linear (``False``,
    the default panel) and log (``True``, the log-log companion that
    surfaces scaling slopes).
    """
    modes_present = [m for m in MODES if m in df["mode"].unique()]
    if not modes_present:
        raise ValueError("no data to plot.")

    n_cols = len(modes_present)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5.8 * n_cols, 4.6), squeeze=False,
    )
    axes = axes[0]

    for j, mode in enumerate(modes_present):
        ax = axes[j]
        panel_sub = df[df["mode"] == mode]
        ks_in_panel = sorted(int(k) for k in panel_sub["k"].unique())

        for k in ks_in_panel:
            color = K_COLORS.get(k, "#444444")
            marker = K_MARKERS.get(k, "x")
            agg = _aggregate_for_k(panel_sub, k)
            if agg.empty:
                continue
            ax.plot(
                agg["n"], agg["dq_mean"],
                marker=marker, lw=1.8, color=color,
                label=f"$k={k}$", markersize=6,
            )
            ax.fill_between(
                agg["n"],
                agg["dq_mean"] - agg["dq_std"],
                agg["dq_mean"] + agg["dq_std"],
                alpha=0.18, color=color,
            )
        ax.set_title("Census 1990")
        ax.grid(True, which="both", alpha=0.25, ls="--", lw=0.5)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        # X ticks at actual swept n values, formatted with K/M units
        # so labels are short ("16K", "500K", "1M") instead of full
        # integers with thousands separators.
        ns_sorted = sorted(int(n) for n in panel_sub["n"].unique())
        if ns_sorted:
            ax.set_xticks(ns_sorted)
            ax.set_xticklabels(
                [_format_thousands(n) for n in ns_sorted],
                rotation=30, ha="right",
            )
            ax.minorticks_off()

        # Y axis: same K/M tick format. Auto-chosen tick positions
        # work fine for both linear and log y; the FuncFormatter only
        # rewrites how each tick value is rendered.
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))

        ax.set_xlabel("Dataset size $n$", fontsize=10)
        if j == 0:
            ax.set_ylabel("Distance queries", fontsize=10)

        # Per-panel legend (was a fig-level legend below the figure
        # before). Compact, frame-less, top-left where it usually
        # doesn't collide with curves rising left-to-right.
        ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Top-level: produce a (linear, log-log) pair
# ----------------------------------------------------------------------
def plot_for_run(
    csv_dir: Path,
    out_dir: Path,
    modes: list[str],
    exp: int,
    test: bool,
) -> list[Path]:
    df = load_all_csvs(csv_dir, modes, exp, test=test)
    suffix = "_test" if test else f"_exp{exp}"
    print(f"Loaded {len(df)} rows from {csv_dir} (suffix={suffix!r})")
    print_banner(df, suffix)

    out_paths: list[Path] = []

    linear = out_dir / f"exp_scaling_panels{suffix}.png"
    plot_panels(df, linear, log_y=False)
    print(f"Wrote {linear}")
    out_paths.append(linear)

    loglog = out_dir / f"exp_scaling_panels{suffix}_loglog.png"
    plot_panels(df, loglog, log_y=True)
    print(f"Wrote {loglog}")
    out_paths.append(loglog)

    print_banner(df, suffix)
    return out_paths


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir", type=Path, default=DEFAULT_CSV_DIR,
        help="Directory containing the per-(mode, exp) CSVs.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Where to write the plots.",
    )
    parser.add_argument(
        "--exp", type=int, default=None,
        choices=range(1, 8), metavar="E",
        help=("--exp value of the run to plot (required unless --test). "
              "Reads census_<mode>_exp<E>.csv for each mode."),
    )
    parser.add_argument(
        "--modes", nargs="+", default=None, choices=MODES,
        help=f"Modes to plot. Default: all of {MODES}.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=("Plot test-mode CSVs (census_<mode>_test.csv). --exp "
              "ignored."),
    )
    args = parser.parse_args()

    if not args.test and args.exp is None:
        parser.error("--exp is required unless --test is set.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = args.modes if args.modes else list(MODES)
    plot_for_run(
        args.csv_dir, args.out_dir,
        modes=modes,
        exp=args.exp if args.exp is not None else 0,
        test=args.test,
    )


if __name__ == "__main__":
    main()