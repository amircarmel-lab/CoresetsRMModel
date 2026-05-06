"""Experiment 3 runner: noise robustness via two quadruplet oracles.

Two oracles per (dataset, rep), one row per (oracle_kind, phi) pair:

  interpolated  — MixedQuadrupletOracle. Routes each query to either the
                  cheap embedding (noisy) or the expensive embedding
                  (exact) with probability beta = 1 - phi/phi_real,
                  giving expected effective phi.
  synthetic     — EuclideanQuadrupletOracle on the expensive embedding,
                  with i.i.d. flip rate phi (cached for persistence).

The phi grid is generated dynamically per dataset:
  full mode (--full): 0.0, 0.025, ..., k*0.025 < phi_real, then phi_real
  semi mode (--semi): 0.0, 0.05,  ..., k*0.05  < phi_real, then phi_real

Exactly one of --full / --semi must be supplied (except in --merge mode).

Three execution shapes, selected by which flags are present:

  Whole experiment (no --rep, no --merge):
    Runs every (dataset, rep) cell sequentially; iterates over all phi
    values and both oracle kinds inside each cell. Writes the canonical
    <dataset>[_test].csv directly.

  Per-rep mode (--rep N):
    Runs ONE rep across all selected datasets. Writes
    <dataset>[_test]_rep<N>.csv. Designed for cluster fan-out: launch
    5 jobs with --rep 0..4, then run --merge.

  Merge (--merge):
    Concatenates <dataset>[_test]_rep*.csv into the canonical CSV.

ALG-TESTER:
  Off by default. Enable with --tester-on (slows alg_g substantially).
  Smoke runs (--test) ALWAYS run with the tester off.

CLI examples::

    python -m experiments.experiment3_phi.run_exp3 --full --rep 0
    python -m experiments.experiment3_phi.run_exp3 --semi --rep 0 --tester-on
    python -m experiments.experiment3_phi.run_exp3 --full --merge
    python -m experiments.experiment3_phi.run_exp3 --semi --test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import coreset_rm.alg_g as alg_g_module
from coreset_rm.oracles import (
    EuclideanDistanceOracle,
    EuclideanQuadrupletOracle, MixedQuadrupletOracle,
)
from coreset_rm.rng import RNG

from documentation import ours
from documentation import stl10, twentynewsgroups
from documentation import (
    evaluate_strong_coreset_coords,
    measure_empirical_phi,
)
from documentation import append_row



# ----------------------------------------------------------------------
# Fixed parameters
# ----------------------------------------------------------------------
M_MULT_PHASE2 = 25      # coreset size = M_MULT_PHASE2 * k
Z = 2

K_PER_DATASET = {
    "text": 20,
    "image": 10,
}

SUBSAMPLE_SEED = 42

ALL_DATASETS = ["text", "image"]
ALL_ORACLES = ["interpolated", "synthetic"]

PHI_STEP_FULL = 0.025
PHI_STEP_SEMI = 0.05

RESULTS_DIR = Path(__file__).parent / "results"
CSV_DIR = RESULTS_DIR / "csv"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class ExpConfig:
    mode: str             # 'test' | 'full' | 'semi'
    subsample_n: Optional[int]
    phi_step: float
    n_reps: int
    n_kmpp_seeds: int
    phi_real_n_samples: int


def make_config(mode: str) -> ExpConfig:
    """mode in {'test', 'full', 'semi'}."""
    if mode == "test":
        return ExpConfig(
            mode="test",
            subsample_n=1000,
            phi_step=0.05,
            n_reps=2,
            n_kmpp_seeds=5,
            phi_real_n_samples=10_000,
        )
    if mode == "full":
        return ExpConfig(
            mode="full",
            subsample_n=None,
            phi_step=PHI_STEP_FULL,
            n_reps=10,
            n_kmpp_seeds=20,
            phi_real_n_samples=10_000,
        )
    if mode == "semi":
        return ExpConfig(
            mode="semi",
            subsample_n=None,
            phi_step=PHI_STEP_SEMI,
            n_reps=5,
            n_kmpp_seeds=20,
            phi_real_n_samples=10_000,
        )
    raise ValueError(f"unknown mode: {mode!r}")


def build_phi_grid(phi_real: float, step: float) -> list[float]:
    """0.0, step, 2*step, ... up to (strictly less than) phi_real,
    plus phi_real as the final point.

    Examples::
        phi_real=0.259, step=0.025  -> [0.0, 0.025, ..., 0.25, 0.259]
        phi_real=0.415, step=0.025  -> [0.0, 0.025, ..., 0.40, 0.415]
        phi_real=0.415, step=0.05   -> [0.0, 0.05, ..., 0.40, 0.415]
    """
    if phi_real <= 0.0:
        return [0.0]
    grid = []
    val = 0.0
    while val < phi_real:
        grid.append(round(val, 6))
        val += step
    grid.append(round(phi_real, 6))
    return grid


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(results_dir: Path, stamp: str, mode: str, tag: str = "") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    name = f"exp3_{mode}_{tag}_{stamp}.log" if tag else f"exp3_{mode}_{stamp}.log"
    log_path = results_dir / name

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


def log_tester_banner(logger: logging.Logger) -> None:
    state = "ON" if alg_g_module.USE_ALG_TESTER else "OFF"
    logger.info("#" * 70)
    logger.info("# ALG-TESTER: %s", state)
    logger.info("#" * 70)


def log_total_elapsed_banner(logger: logging.Logger, elapsed_s: float) -> None:
    """End-of-run banner showing total wall-clock time, formatted as
    seconds, minutes, and HH:MM:SS for quick scanning."""
    h = int(elapsed_s // 3600)
    m = int((elapsed_s % 3600) // 60)
    s = int(elapsed_s % 60)
    logger.info("#" * 70)
    logger.info("# TOTAL ELAPSED TIME")
    logger.info("#   seconds:  %.1f", elapsed_s)
    logger.info("#   minutes:  %.2f", elapsed_s / 60)
    logger.info("#   hours:    %.3f", elapsed_s / 3600)
    logger.info("#   hh:mm:ss: %02d:%02d:%02d", h, m, s)
    logger.info("#" * 70)


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------
def _subsample_pair(
    cheap: np.ndarray, expensive: np.ndarray, n: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(cheap.shape[0], size=n, replace=False)
    idx.sort()
    return (
        np.ascontiguousarray(cheap[idx]),
        np.ascontiguousarray(expensive[idx]),
    )


def load_text(n: Optional[int], logger: logging.Logger):
    logger.info("Loading 20-Newsgroups...")
    bundle = twentynewsgroups.load()
    cheap = bundle.metadata["cheap_coords"]
    expensive = bundle.metadata["expensive_coords"]
    full_n = cheap.shape[0]
    logger.info("  loaded: n=%d cheap_dim=%d expensive_dim=%d",
                full_n, cheap.shape[1], expensive.shape[1])
    if n is not None and n < full_n:
        cheap, expensive = _subsample_pair(cheap, expensive, n, SUBSAMPLE_SEED)
        logger.info("  subsampled to n=%d (SUBSAMPLE_SEED=%d, hardcoded)",
                    n, SUBSAMPLE_SEED)
    return cheap.astype(np.float64), expensive.astype(np.float64)


def load_image(n: Optional[int], logger: logging.Logger):
    logger.info("Loading STL-10...")
    bundle = stl10.load()
    cheap = bundle.metadata["cheap_coords"]
    expensive = bundle.metadata["expensive_coords"]
    full_n = cheap.shape[0]
    logger.info("  loaded: n=%d cheap_dim=%d expensive_dim=%d",
                full_n, cheap.shape[1], expensive.shape[1])
    if n is not None and n < full_n:
        cheap, expensive = _subsample_pair(cheap, expensive, n, SUBSAMPLE_SEED)
        logger.info("  subsampled to n=%d (SUBSAMPLE_SEED=%d, hardcoded)",
                    n, SUBSAMPLE_SEED)
    return cheap.astype(np.float64), expensive.astype(np.float64)


DATASET_LOADERS: dict[str, Callable] = {
    "text": load_text,
    "image": load_image,
}


# ----------------------------------------------------------------------
# CSV schema
# ----------------------------------------------------------------------
MAIN_CSV_FIELDS = [
    "mode", "dataset", "oracle_kind",
    "phi_target", "phi_real", "beta", "rep_seed",
    "n", "k", "m", "z",
    "cost_ratio",
    "construction_time_s",
    "construction_distance_queries",
    "construction_quadruplet_queries",
    "routed_to_expensive",
    "routed_to_cheap",
    "alg_tester",
]


# ----------------------------------------------------------------------
# One coreset-build + evaluate (per oracle_kind)
# ----------------------------------------------------------------------
def run_one(
    *,
    dataset_name: str,
    mode: str,
    oracle_kind: str,
    phi_target: float,
    phi_real: float,
    rep_seed: int,
    cheap: np.ndarray,
    expensive: np.ndarray,
    k: int,
    m: int,
    n_kmpp_seeds: int,
    logger: logging.Logger,
) -> dict:
    """Build one coreset under one oracle_kind, evaluate, return a row dict."""

    if oracle_kind == "interpolated":
        if phi_real <= 0.0:
            beta = 1.0
        else:
            beta = max(0.0, min(1.0, 1.0 - phi_target / phi_real))
        quad_oracle = MixedQuadrupletOracle(
            cheap_coords=cheap, expensive_coords=expensive,
            beta=beta, seed=rep_seed,
        )
        beta_for_csv = beta
    elif oracle_kind == "synthetic":
        # i.i.d. flip rate = phi_target on the expensive embedding
        # (cached for persistence inside EuclideanQuadrupletOracle).
        beta = float("nan")  # not applicable
        quad_oracle = EuclideanQuadrupletOracle(
            coords=expensive, phi=phi_target, seed=rep_seed,
        )
        beta_for_csv = beta
    else:
        raise ValueError(f"unknown oracle_kind: {oracle_kind!r}")

    dist_oracle = EuclideanDistanceOracle(expensive)
    root_rng = RNG(seed=None)

    cached = ours.prepare(
        coords=expensive, k=k,
        dist_oracle=dist_oracle, quad_oracle=quad_oracle,
        root_rng=root_rng.spawn("ours"),
    )
    r = ours.build(coords=expensive, cached=cached, m=m)

    ratio = evaluate_strong_coreset_coords(
        coreset_coords=r.coreset_coords,
        coreset_weights=r.coreset_weights,
        full_coords=expensive, k=k,
        n_kmpp_seeds=n_kmpp_seeds,
        base_seed=rep_seed * 1000,
    )

    # Routing stats only meaningful for the interpolated oracle.
    if oracle_kind == "interpolated":
        routed_expensive = quad_oracle.routed_to_expensive
        routed_cheap = quad_oracle.query_count - routed_expensive
    else:
        routed_expensive = 0
        routed_cheap = 0

    row = {
        "mode": mode,
        "dataset": dataset_name,
        "oracle_kind": oracle_kind,
        "phi_target": float(phi_target),
        "phi_real": float(phi_real),
        "beta": float(beta_for_csv),
        "rep_seed": int(rep_seed),
        "n": int(expensive.shape[0]),
        "k": int(k),
        "m": int(m),
        "z": int(Z),
        "cost_ratio": float(ratio),
        "construction_time_s": float(r.construction_time_s),
        "construction_distance_queries": int(r.distance_queries),
        "construction_quadruplet_queries": int(r.quadruplet_queries),
        "routed_to_expensive": int(routed_expensive),
        "routed_to_cheap": int(routed_cheap),
        "alg_tester": bool(alg_g_module.USE_ALG_TESTER),
    }
    logger.info(
        "[%s] phi_target=%.4f beta=%.4f seed=%d "
        "ratio=%.4g time=%.2fs dq=%d qq=%d (exp=%d cheap=%d)",
        oracle_kind, phi_target, beta_for_csv, rep_seed, ratio,
        r.construction_time_s,
        r.distance_queries, r.quadruplet_queries,
        routed_expensive, routed_cheap,
    )
    return row


# ----------------------------------------------------------------------
# Per-cell driver: one (dataset, rep), all phi × oracle_kind inside
# ----------------------------------------------------------------------
def run_cell(
    dataset: str,
    rep: int,
    cfg: ExpConfig,
    logger: logging.Logger,
    csv_path: Path,
) -> int:
    cheap, expensive = DATASET_LOADERS[dataset](cfg.subsample_n, logger)
    n = cheap.shape[0]
    k = K_PER_DATASET[dataset]
    m = M_MULT_PHASE2 * k

    logger.info("=" * 70)
    logger.info(
        "cell: dataset=%s rep=%d n=%d k=%d coreset_size=%d (=%d*k) z=%d",
        dataset, rep, n, k, m, M_MULT_PHASE2, Z,
    )
    logger.info("=" * 70)

    if m >= n:
        logger.warning(
            "m=%d >= n=%d; coreset is the whole dataset.",
            m, n,
        )

    phi_real = measure_empirical_phi(
        cheap, expensive,
        n_samples=cfg.phi_real_n_samples, seed=777,
    )
    logger.info(
        "[%s rep=%d] phi_real (n_samples=%d) = %.4f",
        dataset, rep, cfg.phi_real_n_samples, phi_real,
    )

    phi_grid = build_phi_grid(phi_real, cfg.phi_step)
    n_phi = len(phi_grid)
    n_total = n_phi * len(ALL_ORACLES)
    logger.info(
        "[%s rep=%d] phi grid (step=%.3f, |grid|=%d): %s",
        dataset, rep, cfg.phi_step, n_phi,
        ["{:.4f}".format(p) for p in phi_grid],
    )
    logger.info(
        "[%s rep=%d] %d phi × %d oracle kinds = %d runs to do",
        dataset, rep, n_phi, len(ALL_ORACLES), n_total,
    )

    n_rows = 0
    t_start = time.perf_counter()
    cell_idx = 0
    for phi_target in phi_grid:
        for oracle_kind in ALL_ORACLES:
            cell_idx += 1
            logger.info(
                "[%s rep=%d %d/%d] oracle=%s phi_target=%.4f  "
                "(elapsed=%.0fs)",
                dataset, rep, cell_idx, n_total, oracle_kind, phi_target,
                time.perf_counter() - t_start,
            )
            row = run_one(
                dataset_name=dataset,
                mode=cfg.mode,
                oracle_kind=oracle_kind,
                phi_target=phi_target,
                phi_real=phi_real,
                rep_seed=rep,
                cheap=cheap, expensive=expensive,
                k=k, m=m,
                n_kmpp_seeds=cfg.n_kmpp_seeds,
                logger=logger,
            )
            append_row(csv_path, {fld: row[fld] for fld in MAIN_CSV_FIELDS})
            n_rows += 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[%s rep=%d] cell done: %d rows in %.1fs (%.2fh) -> %s",
        dataset, rep, n_rows, elapsed, elapsed / 3600, csv_path,
    )
    return n_rows


# ----------------------------------------------------------------------
# Single-cell mode
# ----------------------------------------------------------------------
def _csv_suffix_for_mode(mode: str) -> str:
    """Empty string for full and semi (both write canonical CSVs).
    '_test' for test mode."""
    if mode == "test":
        return "_test"
    return ""


def run_single_cell(
    dataset: str,
    rep: int,
    cfg: ExpConfig,
    logger: logging.Logger,
) -> Path:
    suffix = _csv_suffix_for_mode(cfg.mode)
    csv_path = CSV_DIR / f"{dataset}{suffix}_rep{rep}.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
        logger.info("deleted existing %s", csv_path.name)

    run_cell(dataset, rep, cfg, logger, csv_path)
    return csv_path


def run_per_rep(
    rep: int,
    datasets: list[str],
    cfg: ExpConfig,
    logger: logging.Logger,
) -> list[Path]:
    written: list[Path] = []
    t_start = time.perf_counter()
    for dataset in datasets:
        p = run_single_cell(dataset, rep, cfg, logger)
        written.append(p)
        logger.info(
            "[rep=%d] %s done at elapsed=%.1fs (%.2fh)",
            rep, dataset, time.perf_counter() - t_start,
            (time.perf_counter() - t_start) / 3600,
        )
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(
        "rep=%d done: %d CSV(s) in %.1fs (%.2fh)",
        rep, len(written), elapsed, elapsed / 3600,
    )
    return written


# ----------------------------------------------------------------------
# Whole-dataset driver
# ----------------------------------------------------------------------
def run_dataset(
    dataset: str,
    cfg: ExpConfig,
    logger: logging.Logger,
) -> Path:
    suffix = _csv_suffix_for_mode(cfg.mode)
    main_csv_path = CSV_DIR / f"{dataset}{suffix}.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if main_csv_path.exists():
        main_csv_path.unlink()
        logger.info("deleted existing %s", main_csv_path.name)

    total_rows = 0
    t_start = time.perf_counter()
    for rep in range(cfg.n_reps):
        total_rows += run_cell(dataset, rep, cfg, logger, main_csv_path)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[%s] all reps done: %d rows in %.1fs (%.2fh) -> %s",
        dataset, total_rows, elapsed, elapsed / 3600, main_csv_path,
    )
    return main_csv_path


# ----------------------------------------------------------------------
# Merge per-rep CSVs into canonical CSVs
# ----------------------------------------------------------------------
def merge_per_rep_csvs(
    datasets: list[str],
    suffix: str,
    logger: logging.Logger,
) -> list[Path]:
    written: list[Path] = []
    for dataset in datasets:
        per_rep = sorted(
            CSV_DIR.glob(f"{dataset}{suffix}_rep*.csv"),
            key=lambda p: int(p.stem.replace(f"{dataset}{suffix}_rep", "")),
        )
        if not per_rep:
            logger.warning(
                "merge: no per-rep CSVs for %s%s under %s",
                dataset, suffix, CSV_DIR,
            )
            continue

        canonical = CSV_DIR / f"{dataset}{suffix}.csv"
        if canonical.exists():
            canonical.unlink()
        logger.info(
            "merge: %s%s.csv <- %d per-rep file(s)",
            dataset, suffix, len(per_rep),
        )

        for src in per_rep:
            with src.open("r", encoding="utf-8") as f:
                header = f.readline()
                if not header:
                    logger.warning("  skip empty %s", src.name)
                    continue
                first_write = not canonical.exists()
                with canonical.open("a", encoding="utf-8") as out:
                    if first_write:
                        out.write(header)
                    for line in f:
                        out.write(line)
            logger.info("  +%s", src.name)
        written.append(canonical)
        logger.info("  -> %s", canonical)

    return written


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Experiment 3 runner: noise robustness via two "
                     "quadruplet oracles (interpolated + synthetic)."),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=ALL_DATASETS,
        help=f"Datasets to run. Default: all of {ALL_DATASETS}.",
    )
    parser.add_argument(
        "--rep", type=int, default=None,
        help=("Per-rep mode: run ONE rep across all selected datasets."),
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Concatenate per-rep CSVs into the canonical CSV.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=("Smoke run: subsampled n=1000, 2 reps, 5 k-means++ seeds, "
              "step=0.05. ALG-TESTER forced OFF in test mode."),
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full mode: phi step = 0.025. Required unless --semi or --test.",
    )
    parser.add_argument(
        "--semi", action="store_true",
        help="Semi mode: phi step = 0.05. Required unless --full or --test.",
    )
    parser.add_argument(
        "--tester-on", action="store_true",
        help="Enable ALG-TESTER inside alg_g (slow). Ignored in --test.",
    )
    args = parser.parse_args()

    # --- Validate --full / --semi / --test ----------------------------
    if args.test and (args.full or args.semi):
        parser.error("--test cannot be combined with --full or --semi")
    if args.full and args.semi:
        parser.error("pass exactly one of --full / --semi (not both)")
    # --merge needs to know which suffix to look for. We accept --full/--semi
    # alongside --merge to determine the canonical CSV name (both write the
    # same canonical filename, but knowing the mode is helpful for logging).
    if not (args.test or args.full or args.semi or args.merge):
        parser.error("must pass exactly one of --full / --semi / --test "
                     "(or --merge with one of them)")

    if args.test:
        mode = "test"
    elif args.full:
        mode = "full"
    elif args.semi:
        mode = "semi"
    else:
        # --merge alone: default to 'full' for the suffix logic; this only
        # affects log filenames, not the data files (canonical full and
        # semi share the same suffix).
        mode = "full"

    cfg = make_config(mode)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    datasets = args.datasets if args.datasets else list(ALL_DATASETS)

    # --- Mode C: merge ------------------------------------------------
    if args.merge:
        t_start = time.perf_counter()
        suffix = _csv_suffix_for_mode(mode)
        log_path = setup_logging(RESULTS_DIR, stamp, mode, tag="merge")
        logger = logging.getLogger(f"exp3_{mode}_merge")
        logger.info("merge mode (suffix=%r)", suffix)
        logger.info("  datasets: %s", datasets)
        written = merge_per_rep_csvs(
            datasets=datasets, suffix=suffix, logger=logger,
        )
        logger.info("merge done. wrote %d canonical CSV(s):", len(written))
        for p in written:
            logger.info("  %s", p)
        log_total_elapsed_banner(logger, time.perf_counter() - t_start)
        return

    # Tester forced OFF in test mode regardless of --tester-on.
    if mode == "test":
        alg_g_module.USE_ALG_TESTER = False
    else:
        alg_g_module.USE_ALG_TESTER = bool(args.tester_on)

    # --- Mode B: per-rep ---------------------------------------------
    if args.rep is not None:
        tag = f"rep{args.rep}"
        log_path = setup_logging(RESULTS_DIR, stamp, mode, tag=tag)
        logger = logging.getLogger(f"exp3_{mode}_{tag}")
        logger.info("=" * 70)
        logger.info(
            "Experiment 3 per-rep — rep=%d mode=%s — %s",
            args.rep, mode, stamp,
        )
        logger.info("=" * 70)
        logger.info("datasets:  %s", datasets)
        logger.info("cfg:       %s", asdict(cfg))
        log_tester_banner(logger)
        logger.info("log:       %s", log_path)

        t_start = time.perf_counter()
        written = run_per_rep(args.rep, datasets, cfg, logger)
        elapsed = time.perf_counter() - t_start
        logger.info("=" * 70)
        logger.info(
            "per-rep done in %.1fs (%.2fh). Wrote %d CSV(s):",
            elapsed, elapsed / 3600, len(written),
        )
        for p in written:
            logger.info("  %s", p)
        log_tester_banner(logger)
        log_total_elapsed_banner(logger, elapsed)
        logger.info("=" * 70)
        return

    # --- Mode A: whole experiment -------------------------------------
    log_path = setup_logging(RESULTS_DIR, stamp, mode)
    logger = logging.getLogger(f"exp3_{mode}")

    logger.info("=" * 70)
    logger.info("Experiment 3 (%s mode) — %s", mode, stamp)
    logger.info("=" * 70)
    logger.info("datasets:  %s", datasets)
    logger.info("cfg:       %s", asdict(cfg))
    log_tester_banner(logger)
    logger.info("log:       %s", log_path)

    t_start = time.perf_counter()
    written: list[Path] = []
    for dataset in datasets:
        main_csv = run_dataset(dataset, cfg, logger)
        written.append(main_csv)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(
        "All done in %.1fs (%.2fh). Wrote %d CSVs:",
        elapsed, elapsed / 3600, len(written),
    )
    for p in written:
        logger.info("  %s", p)
    log_tester_banner(logger)
    log_total_elapsed_banner(logger, elapsed)
    logger.info("Plot with:  python -m experiments.experiment3_phi.plot_exp3")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()