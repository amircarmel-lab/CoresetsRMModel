"""Experiment 5 runner: cost ratio vs coreset size, single (n, k) point.

A focused variant of Experiment 2: same baselines, same m-sweep shape,
but a single dataset configuration per (dataset, method) pair —
n=30,000, k=30, with the ``ours`` method using phi=0.15. Three
datasets in this experiment: 20-Newsgroups, STL-10, US Census 1990.
Each is subsampled at construction time to a hardcoded n via
``SUBSAMPLE_SEED`` at module level so every rep sees the same subset.

The ``ours`` baseline uses :class:`EuclideanQuadrupletOracle` on the
expensive embedding (text/image) or the dataset coords (census), with
the ``USE_TESTER_CACHE`` default left ON — at n=30K the cache is
small (tens of MB) and amortizes the per-comparison cost of phi=0.15
majority voting.

Three execution shapes, like exp2:

  Whole experiment (no --rep, no --merge):
    Sweeps every (method, dataset, rep, m) cell sequentially in one
    process. Writes the canonical <method>_<dataset>.csv directly.

  Per-rep mode (--rep N):
    ONE rep across all selected methods × datasets × m. Writes
    <method>_<dataset>_rep<N>.csv per pair. Designed for cluster /
    local fan-out.

  Merge (--merge):
    Concatenates <method>_<dataset>_rep*.csv into the canonical CSV.

ALG-TESTER:
  Off by default. Enable with --tester-on (slows alg_g substantially).
  Smoke runs (--test) ALWAYS run with the tester off regardless of the
  flag. Each row is tagged with the active alg_tester state.

CLI examples::

    # whole experiment, sequential, single process
    python -m experiments.experiment5_focused.run_exp5

    # one rep, all methods × datasets (parallel-friendly across reps)
    python -m experiments.experiment5_focused.run_exp5 --rep 0

    # merge per-rep CSVs
    python -m experiments.experiment5_focused.run_exp5 --merge

    # rerun bico end-to-end without ours
    python -m experiments.experiment5_focused.run_exp5 --methods bico

    # smoke (n=2000, 2 reps, m in {200, 500, 1000}, tester forced off)
    python -m experiments.experiment5_focused.run_exp5 --test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import coreset_rm.alg_g as alg_g_module
from coreset_rm.oracles import (
    EuclideanDistanceOracle,
    EuclideanQuadrupletOracle, MixedQuadrupletOracle,
)
from coreset_rm.rng import RNG

from documentation import bico, lightweight, ours, sensitivity
from documentation import census, stl10, twentynewsgroups
from documentation import (
    evaluate_strong_coreset_coords,
    measure_empirical_phi,
)
from documentation import append_row

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
Z = 2

K_PER_DATASET_FULL = {"text": 20, "image": 10, "census": 30}
K_PER_DATASET_TEST = {"text": 20, "image": 10, "census": 30}

M_SWEEP_FULL = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
M_SWEEP_TEST = [200, 500, 1000]

N_REPS_FULL = 10
N_REPS_TEST = 2

N_KMPP_SEEDS_FULL = 20
N_KMPP_SEEDS_TEST = 5

# Fixed subsample size per dataset.
SUBSAMPLE_N_FULL = 25_000
SUBSAMPLE_N_TEST = 2_000
SUBSAMPLE_SEED = 42        # constant across reps

# Phi for the ours method. Realised on text/image via the
# MixedQuadrupletOracle (cheap/expensive routing with beta tuned to
# the dataset's empirical phi_real). On census there is no
# cheap/expensive split, so we fall back to the synthetic
# EuclideanQuadrupletOracle with i.i.d. flip rate phi.
PHI_OURS = 0.15

# Seed for measure_empirical_phi() inside prepare_ctx. Distinct from
# exp3's seed=777 so independent measurements produced here don't
# correlate with exp3's; phi_real should still match because the
# function is over the whole dataset, not a subsample.
PHI_REAL_SEED = 888

# Sample count for measure_empirical_phi(). Same magnitude as exp3.
PHI_REAL_N_SAMPLES = 10_000

ALL_METHODS = ["lightweight", "sensitivity", "bico", "ours_phi015"]
ALL_DATASETS = ["text", "image", "census"]

RESULTS_DIR = Path(__file__).parent / "results"
CSV_DIR = RESULTS_DIR / "csv"
ALG_G_TIMES_PATH = RESULTS_DIR / "alg_g_times.txt"


CSV_FIELDS = [
    "method", "dataset", "k", "rep",
    "requested_m", "actual_m", "coreset_size",
    "n",
    "phi",
    "cost_ratio",
    "construction_time_s",
    "construction_distance_queries",
    "construction_quadruplet_queries",
    "alg_tester",
]


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class ExpConfig:
    mode: str             # "test" or "full"
    k_per_dataset: dict[str, int]
    m_sweep: list[int]
    n_reps: int
    n_kmpp_seeds: int
    subsample_n: int


def make_config(mode: str) -> ExpConfig:
    if mode == "test":
        return ExpConfig(
            mode="test",
            k_per_dataset=dict(K_PER_DATASET_TEST),
            m_sweep=list(M_SWEEP_TEST),
            n_reps=N_REPS_TEST,
            n_kmpp_seeds=N_KMPP_SEEDS_TEST,
            subsample_n=SUBSAMPLE_N_TEST,
        )
    if mode == "full":
        return ExpConfig(
            mode="full",
            k_per_dataset=dict(K_PER_DATASET_FULL),
            m_sweep=list(M_SWEEP_FULL),
            n_reps=N_REPS_FULL,
            n_kmpp_seeds=N_KMPP_SEEDS_FULL,
            subsample_n=SUBSAMPLE_N_FULL,
        )
    raise ValueError(f"unknown mode: {mode!r}")


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(results_dir: Path, stamp: str, mode: str, tag: str = "") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    name = f"exp5_{mode}_{tag}_{stamp}.log" if tag else f"exp5_{mode}_{stamp}.log"
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


def log_elapsed_banner(logger: logging.Logger, elapsed_seconds: float) -> None:
    h = int(elapsed_seconds // 3600)
    m = int((elapsed_seconds % 3600) // 60)
    s = int(elapsed_seconds % 60)
    logger.info("#" * 20)
    logger.info("# hh:mm:ss: %02d:%02d:%02d", h, m, s)
    logger.info("#" * 20)


# ----------------------------------------------------------------------
# Dataset loading & per-dataset coords extraction
#
# Each loader returns ``(expensive, cheap_or_None)``:
#   - ``expensive`` is the geometry used by the algorithm (and by
#     EuclideanDistanceOracle for distance queries).
#   - ``cheap`` is the second embedding only present for text and
#     image; census has no cheap/expensive split, so it returns
#     ``None``. The MixedQuadrupletOracle is used when ``cheap`` is
#     not None; otherwise the synthetic EuclideanQuadrupletOracle
#     with i.i.d. flips is used.
# ----------------------------------------------------------------------
def _subsample_idx(full_n: int, n: int, seed: int) -> np.ndarray:
    """Pick `n` row indices from `[0, full_n)` deterministically."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(full_n, size=n, replace=False)
    idx.sort()
    return idx


def _subsample_pair(
    expensive: np.ndarray, cheap: np.ndarray, n: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample both arrays with the SAME row indices so cheap[i]
    and expensive[i] still describe the same item."""
    full_n = expensive.shape[0]
    if full_n <= n:
        return (
            np.ascontiguousarray(expensive),
            np.ascontiguousarray(cheap),
        )
    idx = _subsample_idx(full_n, n, seed)
    return (
        np.ascontiguousarray(expensive[idx]),
        np.ascontiguousarray(cheap[idx]),
    )


def _subsample_single(coords: np.ndarray, n: int, seed: int) -> np.ndarray:
    full_n = coords.shape[0]
    if full_n <= n:
        return np.ascontiguousarray(coords)
    idx = _subsample_idx(full_n, n, seed)
    return np.ascontiguousarray(coords[idx])


def _load_text_coords(
    n: int, logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """20-Newsgroups: returns (expensive, cheap)."""
    logger.info("Loading 20-Newsgroups...")
    bundle = twentynewsgroups.load()
    expensive = bundle.metadata["expensive_coords"]
    cheap = bundle.metadata["cheap_coords"]
    full_n = expensive.shape[0]
    logger.info(
        "  loaded: full_n=%d expensive_dim=%d cheap_dim=%d",
        full_n, expensive.shape[1], cheap.shape[1],
    )
    if n < full_n:
        expensive, cheap = _subsample_pair(expensive, cheap, n, SUBSAMPLE_SEED)
        logger.info("  subsampled to n=%d (SUBSAMPLE_SEED=%d)", n, SUBSAMPLE_SEED)
    return (
        np.ascontiguousarray(expensive, dtype=np.float64),
        np.ascontiguousarray(cheap, dtype=np.float64),
    )


def _load_image_coords(
    n: int, logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """STL-10: returns (expensive, cheap)."""
    logger.info("Loading STL-10...")
    bundle = stl10.load()
    expensive = bundle.metadata["expensive_coords"]
    cheap = bundle.metadata["cheap_coords"]
    full_n = expensive.shape[0]
    logger.info(
        "  loaded: full_n=%d expensive_dim=%d cheap_dim=%d",
        full_n, expensive.shape[1], cheap.shape[1],
    )
    if n < full_n:
        expensive, cheap = _subsample_pair(expensive, cheap, n, SUBSAMPLE_SEED)
        logger.info("  subsampled to n=%d (SUBSAMPLE_SEED=%d)", n, SUBSAMPLE_SEED)
    return (
        np.ascontiguousarray(expensive, dtype=np.float64),
        np.ascontiguousarray(cheap, dtype=np.float64),
    )


def _load_census_coords(
    n: int, logger: logging.Logger,
) -> tuple[np.ndarray, None]:
    """Census 1990: returns (coords, None) — no cheap/expensive split."""
    logger.info("Loading Census 1990...")
    bundle = census.load()
    coords = None
    if hasattr(bundle, "coords") and bundle.coords is not None:
        coords = bundle.coords
    elif hasattr(bundle, "metadata"):
        for key in ("coords", "expensive_coords", "X", "features", "data"):
            if key in bundle.metadata:
                coords = bundle.metadata[key]
                break
    if coords is None:
        raise KeyError(
            "could not locate coords on census bundle; inspect "
            "bundle.metadata.keys() and adjust _load_census_coords()"
        )
    full_n = coords.shape[0]
    logger.info("  loaded: full_n=%d dim=%d", full_n, coords.shape[1])
    if n < full_n:
        coords = _subsample_single(coords, n, SUBSAMPLE_SEED)
        logger.info("  subsampled to n=%d (SUBSAMPLE_SEED=%d)", n, SUBSAMPLE_SEED)
    return np.ascontiguousarray(coords, dtype=np.float64), None


DATASET_LOADERS: dict[
    str, Callable[[int, logging.Logger], tuple[np.ndarray, Optional[np.ndarray]]]
] = {
    "text": _load_text_coords,
    "image": _load_image_coords,
    "census": _load_census_coords,
}


def load_dataset(
    name: str, cfg: ExpConfig, logger: logging.Logger,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    return DATASET_LOADERS[name](cfg.subsample_n, logger)


# ----------------------------------------------------------------------
# alg_g timings trace
# ----------------------------------------------------------------------
def log_alg_g_time(
    method: str, dataset: str, k: int, rep: int, phi: float, time_s: float,
) -> None:
    ALG_G_TIMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"{method}  {dataset}  k={k:<3d} rep={rep:<2d} phi={phi:.3f} "
        f"tester={'ON' if alg_g_module.USE_ALG_TESTER else 'OFF':<3s} "
        f"time={time_s:.3f}\n"
    )
    with ALG_G_TIMES_PATH.open("a") as f:
        f.write(line)


# ----------------------------------------------------------------------
# Per-method build dispatch
# ----------------------------------------------------------------------
@dataclass
class BuildCtx:
    coords: np.ndarray
    k: int
    dist_oracle: EuclideanDistanceOracle
    quad_oracle: Optional[object]    # EuclideanQuadrupletOracle or MixedQuadrupletOracle
    root_rng: RNG
    cached_ours: object = field(default=None)
    cached_sensitivity: object = field(default=None)
    # Effective noise rate of the quadruplet oracle this ctx will
    # exercise. NaN for methods that don't use a quad oracle
    # (lightweight, sensitivity, bico). 0.15 for census ours
    # (synthetic). phi_real for text/image ours (Mixed cheap-only).
    # Recorded into the CSV's `phi` column so the plotter can show
    # the per-dataset operating point.
    effective_phi: float = field(default=float("nan"))


def prepare_ctx(
    method: str, coords: np.ndarray, cheap: Optional[np.ndarray],
    k: int, rep: int,
    dataset_name: str, logger: logging.Logger,
) -> BuildCtx:
    """Build a BuildCtx for ``method``. ``coords`` is the geometry
    used by the algorithm and the distance oracle. ``cheap`` is the
    second embedding for text/image (used to construct the
    MixedQuadrupletOracle); None for census.
    """
    n = coords.shape[0]
    dist_oracle = EuclideanDistanceOracle(coords)
    root_rng = RNG(seed=None)

    if method == "ours_phi015":
        phi_target = PHI_OURS
        if cheap is not None:
            # text / image: route every quadruplet query to the cheap
            # embedding (beta=0). The effective noise rate is then
            # phi_real of the dataset itself — see
            # MixedQuadrupletOracle docstring, which states
            # "beta = 0 reduces to the cheap-only oracle (phi =
            # phi_real)". phi=0.15 is NOT the operating point for
            # text/image; it's the synthetic-noise level we use on
            # census. We measure phi_real here so it can be recorded
            # in the CSV `phi` column and surfaced in the plot legend.
            phi_real = measure_empirical_phi(
                cheap, coords,
                n_samples=PHI_REAL_N_SAMPLES, seed=PHI_REAL_SEED,
            )
            beta = 0.0
            quad_oracle = MixedQuadrupletOracle(
                cheap_coords=cheap, expensive_coords=coords,
                beta=beta, seed=rep,
            )
            effective_phi = float(phi_real)
            logger.info(
                "  oracle: Mixed (cheap-only) on %s  beta=%.2f  "
                "phi_real=%.4f (effective phi)",
                dataset_name, beta, phi_real,
            )
        else:
            # census: synthetic i.i.d.-flip oracle on the coords.
            quad_oracle = EuclideanQuadrupletOracle(
                coords, phi=phi_target, seed=rep,
            )
            effective_phi = float(phi_target)
            logger.info(
                "  oracle: Euclidean (synthetic) on %s  phi=%.3f",
                dataset_name, phi_target,
            )

        t0 = time.perf_counter()
        cached = ours.prepare(
            coords=coords, k=k,
            dist_oracle=dist_oracle, quad_oracle=quad_oracle,
            root_rng=root_rng.spawn(method),
        )
        alg_g_time = time.perf_counter() - t0
        log_alg_g_time(method, dataset_name, k, rep, effective_phi, alg_g_time)
        logger.info(
            "  alg_g (via prepare) %s on %s k=%d rep=%d phi=%.3f "
            "tester=%s: %.3fs",
            method, dataset_name, k, rep, effective_phi,
            "ON" if alg_g_module.USE_ALG_TESTER else "OFF",
            alg_g_time,
        )
        return BuildCtx(
            coords=coords, k=k,
            dist_oracle=dist_oracle, quad_oracle=quad_oracle,
            root_rng=root_rng, cached_ours=cached,
            effective_phi=effective_phi,
        )

    if method == "sensitivity":
        cached = sensitivity.prepare(
            n=n, k=k, dist_oracle=dist_oracle,
            root_rng=root_rng.spawn("sensitivity"),
        )
        return BuildCtx(
            coords=coords, k=k,
            dist_oracle=dist_oracle, quad_oracle=None,
            root_rng=root_rng, cached_sensitivity=cached,
        )

    return BuildCtx(
        coords=coords, k=k,
        dist_oracle=dist_oracle, quad_oracle=None,
        root_rng=root_rng,
    )


def build_one(method: str, ctx: BuildCtx, m: int):
    if method == "lightweight":
        return lightweight.build(
            coords=ctx.coords, m=m,
            root_rng=ctx.root_rng.spawn(f"lightweight_m{m}"),
        )
    if method == "sensitivity":
        return sensitivity.build(
            coords=ctx.coords, cached=ctx.cached_sensitivity, m=m,
        )
    if method == "bico":
        return bico.build(
            coords=ctx.coords, k=ctx.k, m=m,
            root_rng=ctx.root_rng.spawn(f"bico_m{m}"),
        )
    if method == "ours_phi015":
        return ours.build(
            coords=ctx.coords, cached=ctx.cached_ours, m=m,
        )
    raise ValueError(f"unknown method: {method!r}")


# ----------------------------------------------------------------------
# Inner loop: one (method, dataset, rep) cell — iterates m
# ----------------------------------------------------------------------
def run_mrep_cell(
    method: str, dataset: str, coords: np.ndarray,
    cheap: Optional[np.ndarray],
    k: int, rep: int, cfg: ExpConfig,
    csv_path: Path, logger: logging.Logger,
    cell_idx: int, total_cells: int,
) -> int:
    """Run one (method, dataset, rep) — iterates m and appends rows.
    Returns rows written.
    """
    ctx = prepare_ctx(
        method=method, coords=coords, cheap=cheap, k=k, rep=rep,
        dataset_name=dataset, logger=logger,
    )
    n = coords.shape[0]
    rows = 0
    tester_state = bool(alg_g_module.USE_ALG_TESTER)
    for m in cfg.m_sweep:
        cell_idx += 1
        r = build_one(method, ctx, m)
        actual_m = int(r.coreset_coords.shape[0])

        ratio = evaluate_strong_coreset_coords(
            coreset_coords=r.coreset_coords,
            coreset_weights=r.coreset_weights,
            full_coords=coords, k=k,
            n_kmpp_seeds=cfg.n_kmpp_seeds,
            base_seed=rep * 1000 + k,
        )
        row = {
            "method": method, "dataset": dataset,
            "k": int(k), "rep": int(rep),
            "requested_m": int(m),
            "actual_m": actual_m,
            "coreset_size": actual_m,
            "n": int(n),
            "phi": float(ctx.effective_phi),
            "cost_ratio": float(ratio),
            "construction_time_s": float(r.construction_time_s),
            "construction_distance_queries": int(r.distance_queries),
            "construction_quadruplet_queries": int(r.quadruplet_queries),
            "alg_tester": tester_state,
        }
        append_row(csv_path, {fld: row[fld] for fld in CSV_FIELDS})
        rows += 1
        logger.info(
            "  [%d/%d] k=%d rep=%d req_m=%-4d act_m=%-4d "
            "ratio=%.4g dq=%-8d qq=%-8d time=%.3fs",
            cell_idx, total_cells, k, rep, m, actual_m, ratio,
            r.distance_queries, r.quadruplet_queries,
            r.construction_time_s,
        )
    return rows


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------
def run_method_dataset_rep(
    method: str, dataset: str, rep: int,
    cfg: ExpConfig, logger: logging.Logger,
) -> Path:
    """Run one (method, dataset, rep) — all m. Writes per-rep CSV."""
    suffix = "_test" if cfg.mode == "test" else ""
    csv_path = CSV_DIR / f"{method}_{dataset}{suffix}_rep{rep}.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
        logger.info("deleted existing %s", csv_path.name)

    coords, cheap = load_dataset(dataset, cfg, logger)
    n = coords.shape[0]
    k = cfg.k_per_dataset[dataset]
    logger.info("=" * 70)
    logger.info(
        "cell: method=%s dataset=%s rep=%d n=%d k=%d m_sweep=%s "
        "cheap=%s",
        method, dataset, rep, n, k, cfg.m_sweep,
        "yes" if cheap is not None else "no",
    )
    logger.info("=" * 70)

    t_start = time.perf_counter()
    total_cells = len(cfg.m_sweep)
    if max(cfg.m_sweep) > n:
        logger.warning(
            "  m_max=%d > n=%d; m values above n will be clipped "
            "by baselines that require m <= n.", max(cfg.m_sweep), n,
        )
    n_rows = run_mrep_cell(
        method=method, dataset=dataset, coords=coords, cheap=cheap,
        k=k, rep=rep, cfg=cfg, csv_path=csv_path,
        logger=logger, cell_idx=0, total_cells=total_cells,
    )
    elapsed = time.perf_counter() - t_start
    logger.info(
        "%s/%s rep=%d done: %d rows in %.1fs (%.2fh) -> %s",
        method, dataset, rep, n_rows, elapsed, elapsed / 3600, csv_path,
    )
    return csv_path


def run_per_rep(
    rep: int, methods: list[str], datasets: list[str],
    cfg: ExpConfig, logger: logging.Logger,
) -> list[Path]:
    """Per-rep mode: run every (method, dataset) for one rep."""
    written: list[Path] = []
    t_start = time.perf_counter()
    for method in methods:
        for dataset in datasets:
            p = run_method_dataset_rep(method, dataset, rep, cfg, logger)
            written.append(p)
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(
        "rep=%d done: %d CSV(s) in %.1fs (%.2fh)",
        rep, len(written), elapsed, elapsed / 3600,
    )
    return written


def run_method_on_dataset(
    method: str, dataset: str, cfg: ExpConfig, logger: logging.Logger,
) -> Path:
    """Whole-experiment mode (one method × one dataset, all reps × m)."""
    suffix = "_test" if cfg.mode == "test" else ""
    csv_path = CSV_DIR / f"{method}_{dataset}{suffix}.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
        logger.info("deleted existing %s", csv_path.name)

    coords, cheap = load_dataset(dataset, cfg, logger)
    n = coords.shape[0]
    k = cfg.k_per_dataset[dataset]
    logger.info("=" * 70)
    logger.info(
        "method=%s dataset=%s  n=%d k=%d reps=%d m_sweep=%s cheap=%s",
        method, dataset, n, k, cfg.n_reps, cfg.m_sweep,
        "yes" if cheap is not None else "no",
    )
    logger.info("=" * 70)

    t_start = time.perf_counter()
    n_rows = 0
    total_cells = cfg.n_reps * len(cfg.m_sweep)
    cell_idx = 0
    for rep in range(cfg.n_reps):
        if max(cfg.m_sweep) > n:
            logger.warning(
                "  m_max=%d > n=%d; m values above n will be clipped.",
                max(cfg.m_sweep), n,
            )
        n_rows += run_mrep_cell(
            method=method, dataset=dataset, coords=coords, cheap=cheap,
            k=k, rep=rep, cfg=cfg, csv_path=csv_path,
            logger=logger, cell_idx=cell_idx, total_cells=total_cells,
        )
        cell_idx += len(cfg.m_sweep)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "done %s on %s: %d rows in %.1fs -> %s",
        method, dataset, n_rows, elapsed, csv_path,
    )
    return csv_path


# ----------------------------------------------------------------------
# Merge per-rep CSVs into canonical CSVs
# ----------------------------------------------------------------------
def merge_per_rep_csvs(
    methods: list[str], datasets: list[str], test: bool,
    logger: logging.Logger,
) -> list[Path]:
    suffix = "_test" if test else ""
    written: list[Path] = []
    for method in methods:
        for dataset in datasets:
            stem = f"{method}_{dataset}{suffix}"
            per_rep = sorted(
                CSV_DIR.glob(f"{stem}_rep*.csv"),
                key=lambda p: int(p.stem.replace(f"{stem}_rep", "")),
            )
            if not per_rep:
                logger.warning(
                    "merge: no per-rep CSVs for %s under %s", stem, CSV_DIR,
                )
                continue

            canonical = CSV_DIR / f"{stem}.csv"
            if canonical.exists():
                canonical.unlink()
            logger.info(
                "merge: %s.csv <- %d per-rep file(s)",
                stem, len(per_rep),
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
        description=("Experiment 5: cost ratio vs coreset size at "
                     "n=30K, k=30, with ours_phi015. Datasets: text, "
                     "image, census."),
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, choices=ALL_METHODS,
        help=f"Methods to run. Default: all of {ALL_METHODS}.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=ALL_DATASETS,
        help=f"Datasets to run. Default: all of {ALL_DATASETS}.",
    )
    parser.add_argument(
        "--rep", type=int, default=None,
        help=("Per-rep mode: ONE rep across all selected methods × "
              "datasets × m. Writes <method>_<dataset>_rep<N>.csv "
              "per pair."),
    )
    parser.add_argument(
        "--merge", action="store_true",
        help=("Concatenate <method>_<dataset>[_test]_rep*.csv into "
              "the canonical CSVs."),
    )
    parser.add_argument(
        "--test", action="store_true",
        help=("Smoke run: subsampled n=2000, 2 reps, m in {200, 500, "
              "1000}. ALG-TESTER forced OFF in test mode."),
    )
    parser.add_argument(
        "--tester-on", action="store_true",
        help=("Enable ALG-TESTER inside alg_g (slow). Affects only "
              "ours_phi015. Ignored in --test mode."),
    )
    args = parser.parse_args()

    mode = "test" if args.test else "full"
    cfg = make_config(mode)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    methods = args.methods if args.methods else list(ALL_METHODS)
    datasets = args.datasets if args.datasets else list(ALL_DATASETS)

    t_main_start = time.perf_counter()

    # --- Mode C: merge ------------------------------------------------
    if args.merge:
        log_path = setup_logging(RESULTS_DIR, stamp, mode, tag="merge")
        logger = logging.getLogger(f"exp5_{mode}_merge")
        logger.info("merge mode (test=%s)", args.test)
        logger.info("  methods:  %s", methods)
        logger.info("  datasets: %s", datasets)
        written = merge_per_rep_csvs(
            methods=methods, datasets=datasets,
            test=args.test, logger=logger,
        )
        logger.info("merge done. wrote %d canonical CSV(s):", len(written))
        for p in written:
            logger.info("  %s", p)
        log_elapsed_banner(logger, time.perf_counter() - t_main_start)
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
        logger = logging.getLogger(f"exp5_{mode}_{tag}")
        logger.info("=" * 70)
        logger.info(
            "Experiment 5 per-rep — rep=%d mode=%s — %s",
            args.rep, mode, stamp,
        )
        logger.info("=" * 70)
        logger.info("methods:   %s", methods)
        logger.info("datasets:  %s", datasets)
        logger.info("cfg:       %s", cfg)
        log_tester_banner(logger)
        logger.info("log:       %s", log_path)

        t_start = time.perf_counter()
        written = run_per_rep(args.rep, methods, datasets, cfg, logger)
        elapsed = time.perf_counter() - t_start
        logger.info("=" * 70)
        logger.info(
            "per-rep done in %.1fs (%.2fh). Wrote %d CSV(s):",
            elapsed, elapsed / 3600, len(written),
        )
        for p in written:
            logger.info("  %s", p)
        log_tester_banner(logger)
        log_elapsed_banner(logger, time.perf_counter() - t_main_start)
        logger.info("=" * 70)
        return

    # --- Mode A: whole experiment -------------------------------------
    log_path = setup_logging(RESULTS_DIR, stamp, mode)
    logger = logging.getLogger(f"exp5_{mode}")

    logger.info("=" * 70)
    logger.info("Experiment 5 (%s mode) — %s", mode, stamp)
    logger.info("=" * 70)
    logger.info("methods:   %s", methods)
    logger.info("datasets:  %s", datasets)
    logger.info("cfg:       %s", cfg)
    log_tester_banner(logger)
    logger.info("log:       %s", log_path)
    logger.info("alg_g txt: %s", ALG_G_TIMES_PATH)

    t_start = time.perf_counter()
    written: list[Path] = []
    for method in methods:
        for dataset in datasets:
            csv_path = run_method_on_dataset(method, dataset, cfg, logger)
            written.append(csv_path)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("All done in %.1fs. Wrote %d CSVs:", elapsed, len(written))
    for p in written:
        logger.info("  %s", p)
    log_tester_banner(logger)
    log_elapsed_banner(logger, time.perf_counter() - t_main_start)


if __name__ == "__main__":
    main()
