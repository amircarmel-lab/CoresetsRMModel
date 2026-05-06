"""Experiment runner: distance-query scaling vs n on Census 1990.

See module docstring history; only the salient additions for this
revision are listed here.

--n-only N1 [N2 ...] (per-rep mode only)
  Restrict the sweep to the supplied n values (all must already be in
  the n_grid implied by --exp). The existing per-rep CSV is NOT
  deleted; rows whose n matches --n-only are dropped (idempotent
  re-run), then new rows are appended. Use to extend an existing run
  with one or more additional n values without redoing prior cells.

CLI examples (full mode, --exp 7, extend existing rep0 with n=1M):

    python -m experiments.experiment_scaling.run_exp \\
        --full --exp 7 --rep 0 --n-only 1000000
"""
from __future__ import annotations

import argparse
import csv as _csv
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import coreset_rm.alg_g as alg_g_module
from coreset_rm.alg_g import run_alg_g, AlgGOutput
from coreset_rm.oracles import (
    EuclideanDistanceOracle,
    EuclideanQuadrupletOracle,
)
from coreset_rm.rng import RNG
from experiments.baselines import ours
from experiments.datasets import census
from experiments.persistence.persistence import append_row

# ----------------------------------------------------------------------
# Fixed parameters
# ----------------------------------------------------------------------
M_MULT_PHASE2 = 25
PHI = 0.0
BASE_N = 15_625
EXP_MIN, EXP_MAX = 1, 7

N_GRID_TEST = [500, 1000, 2000]
K_GRID_FULL = [5, 10, 20]
K_GRID_TEST = [5, 10]

N_REPS = 10
SHUFFLE_SEED = 42

ALL_MODES = ["semi", "full"]
DATASET_NAME = "census"

RESULTS_DIR = Path(__file__).parent / "results"
CSV_DIR = RESULTS_DIR / "csv"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class ExpConfig:
    mode: str
    test: bool
    exp: int
    n_grid: list[int]
    k_grid: list[int]
    n_reps: int


def make_config(mode: str, test: bool, exp: int) -> ExpConfig:
    if test:
        return ExpConfig(
            mode=mode, test=True, exp=exp,
            n_grid=list(N_GRID_TEST),
            k_grid=list(K_GRID_TEST),
            n_reps=N_REPS,
        )
    n_grid = [BASE_N * (2 ** i) for i in range(exp + 1)]
    return ExpConfig(
        mode=mode, test=False, exp=exp,
        n_grid=n_grid,
        k_grid=list(K_GRID_FULL),
        n_reps=N_REPS,
    )


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(
    results_dir: Path, stamp: str, mode: str, scope: str,
    exp: Optional[int] = None, tag: str = "",
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"exp_{mode}_{scope}"]
    if scope == "full" and exp is not None:
        parts.append(f"exp{exp}")
    if tag:
        parts.append(tag)
    parts.append(stamp)
    log_path = results_dir / ("_".join(parts) + ".log")

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
# Dataset loading
# ----------------------------------------------------------------------
def load_census(logger: logging.Logger) -> np.ndarray:
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
            "bundle.metadata.keys() and adjust load_census()"
        )
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    logger.info("  loaded: n=%d dim=%d", coords.shape[0], coords.shape[1])
    return coords


def shuffle_coords(coords: np.ndarray, logger: logging.Logger) -> np.ndarray:
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(coords.shape[0])
    out = np.ascontiguousarray(coords[perm])
    logger.info(
        "shuffled coords with SHUFFLE_SEED=%d (constant across reps)",
        SHUFFLE_SEED,
    )
    return out


# ----------------------------------------------------------------------
# Grid resolution
# ----------------------------------------------------------------------
def resolve_n_max(n_grid: list[int], full_n: int, logger: logging.Logger) -> int:
    n_max_requested = max(n_grid)
    if n_max_requested > full_n:
        logger.warning(
            "n_max in n_grid is %d but census has only %d points; "
            "clamping n_max to %d.",
            n_max_requested, full_n, full_n,
        )
        return full_n
    return n_max_requested


def resolve_n_grid(n_grid: list[int], n_max: int) -> list[int]:
    out = sorted(set(n for n in n_grid if n < n_max))
    out.append(n_max)
    return out


# ----------------------------------------------------------------------
# CSV schema
# ----------------------------------------------------------------------
CSV_FIELDS = [
    "mode", "exp", "n", "k", "rep", "n_max", "phi",
    "alg_g_size", "alg_g_was_reused",
    "construction_distance_queries",
    "phase1_quadruplet_queries",
    "alg_g_quadruplet_queries",
    "alg_g_time_s", "pipeline_time_s",
    "alg_tester",
]


# ----------------------------------------------------------------------
# Idempotent purge for --n-only re-runs
# ----------------------------------------------------------------------
def purge_n_rows(
    csv_path: Path, n_values: list[int], logger: logging.Logger,
) -> int:
    """Drop rows from csv_path whose `n` is in n_values. Returns the
    number of rows dropped. Leaves header + non-matching rows in place.
    """
    if not csv_path.exists():
        return 0
    targets = {int(n) for n in n_values}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        kept = []
        n_dropped = 0
        for row in reader:
            try:
                n_val = int(row.get("n", ""))
            except (TypeError, ValueError):
                kept.append(row)
                continue
            if n_val in targets:
                n_dropped += 1
            else:
                kept.append(row)
        fieldnames = reader.fieldnames
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    logger.info(
        "purged %d row(s) from %s where n in %s (kept %d)",
        n_dropped, csv_path.name, sorted(targets), len(kept),
    )
    return n_dropped


# ----------------------------------------------------------------------
# Sub-execution: --semi
# ----------------------------------------------------------------------
def run_one_n_semi(
    *,
    k: int, rep: int, exp: int,
    n_sub: int, n_max: int,
    coords_shuffled: np.ndarray,
    alg_g_output_max: AlgGOutput,
    alg_g_size: int, alg_g_time: float, alg_g_quad_queries: int,
) -> dict:
    dist_oracle = EuclideanDistanceOracle(coords_shuffled)
    quad_oracle = EuclideanQuadrupletOracle(
        coords_shuffled, phi=PHI, seed=rep,
    )

    alg_g_output_sub = AlgGOutput(
        C=alg_g_output_max.C,
        M=alg_g_output_max.M[:n_sub],
        weights=alg_g_output_max.weights,
    )

    root_rng = RNG(seed=None)

    cached = ours.prepare(
        coords=coords_shuffled[:n_sub], k=k,
        dist_oracle=dist_oracle, quad_oracle=quad_oracle,
        root_rng=root_rng.spawn("ours"),
        algoutput=alg_g_output_sub,
    )
    m = M_MULT_PHASE2 * k
    r = ours.build(coords=coords_shuffled, cached=cached, m=m)

    is_n_max = (n_sub == n_max)
    return {
        "mode": "semi",
        "exp": int(exp),
        "n": int(n_sub),
        "k": int(k),
        "rep": int(rep),
        "n_max": int(n_max),
        "phi": float(PHI),
        "alg_g_size": int(alg_g_size),
        "alg_g_was_reused": bool(not is_n_max),
        "construction_distance_queries": int(r.distance_queries),
        "phase1_quadruplet_queries": int(r.quadruplet_queries),
        "alg_g_quadruplet_queries": int(alg_g_quad_queries if is_n_max else 0),
        "alg_g_time_s": float(alg_g_time if is_n_max else 0.0),
        "pipeline_time_s": float(r.construction_time_s),
        "alg_tester": bool(alg_g_module.USE_ALG_TESTER),
    }


# ----------------------------------------------------------------------
# Sub-execution: --full
# ----------------------------------------------------------------------
def run_one_n_full(
    *,
    k: int, rep: int, exp: int,
    n_sub: int, n_max: int,
    coords_shuffled: np.ndarray,
) -> dict:
    coords_sub = np.ascontiguousarray(coords_shuffled[:n_sub])

    dist_oracle = EuclideanDistanceOracle(coords_sub)
    quad_oracle = EuclideanQuadrupletOracle(coords_sub, phi=PHI, seed=rep)

    root_rng = RNG(seed=None)

    t0 = time.perf_counter()
    alg_g_output = run_alg_g(
        n=n_sub, k=k,
        quad_oracle=quad_oracle,
        root_rng=root_rng.spawn("alg_g"),
    )
    alg_g_time = time.perf_counter() - t0
    alg_g_size = len(alg_g_output.C)
    alg_g_quad_queries = int(quad_oracle.query_count)

    cached = ours.prepare(
        coords=coords_sub, k=k,
        dist_oracle=dist_oracle, quad_oracle=quad_oracle,
        root_rng=root_rng.spawn("ours"),
        algoutput=alg_g_output,
    )
    m = M_MULT_PHASE2 * k
    r = ours.build(coords=coords_sub, cached=cached, m=m)

    return {
        "mode": "full",
        "exp": int(exp),
        "n": int(n_sub),
        "k": int(k),
        "rep": int(rep),
        "n_max": int(n_max),
        "phi": float(PHI),
        "alg_g_size": int(alg_g_size),
        "alg_g_was_reused": False,
        "construction_distance_queries": int(r.distance_queries),
        "phase1_quadruplet_queries": int(r.quadruplet_queries),
        "alg_g_quadruplet_queries": int(alg_g_quad_queries),
        "alg_g_time_s": float(alg_g_time),
        "pipeline_time_s": float(r.construction_time_s),
        "alg_tester": bool(alg_g_module.USE_ALG_TESTER),
    }


# ----------------------------------------------------------------------
# Per-rep cell driver — iterates k × n
# ----------------------------------------------------------------------
def run_cell(
    rep: int,
    cfg: ExpConfig,
    logger: logging.Logger,
    csv_path: Path,
    coords_shuffled_full: np.ndarray,
) -> int:
    """Run all (k, n) cells for one rep. Caller manages csv_path."""
    full_n = coords_shuffled_full.shape[0]
    n_max = resolve_n_max(cfg.n_grid, full_n, logger)
    n_grid = resolve_n_grid(cfg.n_grid, n_max)
    coords_shuffled = np.ascontiguousarray(coords_shuffled_full[:n_max])

    logger.info("=" * 70)
    logger.info(
        "cell: dataset=%s mode=%s exp=%d rep=%d n_max=%d "
        "n_grid=%s k_grid=%s phi=%.3f",
        DATASET_NAME, cfg.mode, cfg.exp, rep, n_max,
        n_grid, cfg.k_grid, PHI,
    )
    logger.info("=" * 70)

    n_rows = 0
    cell_idx = 0
    total_cells = len(cfg.k_grid) * len(n_grid)
    t_start = time.perf_counter()

    for k in cfg.k_grid:
        if cfg.mode == "semi":
            n_rows += _run_k_semi(
                k=k, rep=rep, cfg=cfg,
                coords_shuffled=coords_shuffled,
                n_max=n_max, n_grid=n_grid,
                csv_path=csv_path, logger=logger,
                cell_idx=cell_idx, total_cells=total_cells,
                t_start=t_start,
            )
        else:
            n_rows += _run_k_full(
                k=k, rep=rep, cfg=cfg,
                coords_shuffled=coords_shuffled,
                n_max=n_max, n_grid=n_grid,
                csv_path=csv_path, logger=logger,
                cell_idx=cell_idx, total_cells=total_cells,
                t_start=t_start,
            )
        cell_idx += len(n_grid)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[mode=%s exp=%d rep=%d] cell done: %d rows in %.1fs (%.2fh) -> %s",
        cfg.mode, cfg.exp, rep, n_rows, elapsed, elapsed / 3600, csv_path,
    )
    return n_rows


def _run_k_semi(
    *,
    k: int, rep: int, cfg: ExpConfig,
    coords_shuffled: np.ndarray,
    n_max: int, n_grid: list[int],
    csv_path: Path, logger: logging.Logger,
    cell_idx: int, total_cells: int, t_start: float,
) -> int:
    dist_oracle_max = EuclideanDistanceOracle(coords_shuffled)
    quad_oracle_max = EuclideanQuadrupletOracle(
        coords_shuffled, phi=PHI, seed=rep,
    )
    root_rng_alg_g = RNG(seed=None)

    t0 = time.perf_counter()
    alg_g_output_max = run_alg_g(
        n=n_max, k=k,
        quad_oracle=quad_oracle_max,
        root_rng=root_rng_alg_g.spawn(f"alg_g_semi_k{k}_rep{rep}"),
    )
    alg_g_time = time.perf_counter() - t0
    alg_g_size = len(alg_g_output_max.C)
    alg_g_quad_queries = int(quad_oracle_max.query_count)
    logger.info(
        "[mode=semi exp=%d k=%d rep=%d] alg_g done on n_max=%d: |C|=%d "
        "qq=%d time=%.2fs",
        cfg.exp, k, rep, n_max, alg_g_size, alg_g_quad_queries, alg_g_time,
    )

    rows = 0
    for n_sub in n_grid:
        cell_idx += 1
        row = run_one_n_semi(
            k=k, rep=rep, exp=cfg.exp,
            n_sub=n_sub, n_max=n_max,
            coords_shuffled=coords_shuffled,
            alg_g_output_max=alg_g_output_max,
            alg_g_size=alg_g_size,
            alg_g_time=alg_g_time,
            alg_g_quad_queries=alg_g_quad_queries,
        )
        append_row(csv_path, {fld: row[fld] for fld in CSV_FIELDS})
        rows += 1
        logger.info(
            "  [%d/%d] mode=semi k=%-2d rep=%d n=%-7d "
            "dq=%-10d qq=%-9d time=%.2fs |C|=%d reused=%s "
            "(elapsed=%.0fs)",
            cell_idx, total_cells, k, rep, n_sub,
            row["construction_distance_queries"],
            row["phase1_quadruplet_queries"],
            row["pipeline_time_s"],
            row["alg_g_size"],
            row["alg_g_was_reused"],
            time.perf_counter() - t_start,
        )
    return rows


def _run_k_full(
    *,
    k: int, rep: int, cfg: ExpConfig,
    coords_shuffled: np.ndarray,
    n_max: int, n_grid: list[int],
    csv_path: Path, logger: logging.Logger,
    cell_idx: int, total_cells: int, t_start: float,
) -> int:
    rows = 0
    for n_sub in n_grid:
        cell_idx += 1
        row = run_one_n_full(
            k=k, rep=rep, exp=cfg.exp,
            n_sub=n_sub, n_max=n_max,
            coords_shuffled=coords_shuffled,
        )
        append_row(csv_path, {fld: row[fld] for fld in CSV_FIELDS})
        rows += 1
        logger.info(
            "  [%d/%d] mode=full k=%-2d rep=%d n=%-7d "
            "dq=%-10d alg_g_qq=%-9d phase1_qq=%-9d "
            "alg_g_time=%.2fs pipeline_time=%.2fs |C|=%d "
            "(elapsed=%.0fs)",
            cell_idx, total_cells, k, rep, n_sub,
            row["construction_distance_queries"],
            row["alg_g_quadruplet_queries"],
            row["phase1_quadruplet_queries"],
            row["alg_g_time_s"], row["pipeline_time_s"],
            row["alg_g_size"],
            time.perf_counter() - t_start,
        )
    return rows


# ----------------------------------------------------------------------
# CSV path helpers
# ----------------------------------------------------------------------
def csv_filename(mode: str, exp: int, test: bool, rep: Optional[int] = None) -> str:
    if test:
        base = f"{DATASET_NAME}_{mode}_test"
    else:
        base = f"{DATASET_NAME}_{mode}_exp{exp}"
    if rep is not None:
        base = f"{base}_rep{rep}"
    return base + ".csv"


# ----------------------------------------------------------------------
# Per-rep mode (parallel-friendly)
# ----------------------------------------------------------------------
def run_single_rep(
    rep: int,
    cfg: ExpConfig,
    logger: logging.Logger,
    coords_shuffled_full: np.ndarray,
    n_only: Optional[list[int]] = None,
) -> Path:
    """Run exactly one rep across all (k, n) cells. Writes per-rep CSV.

    When ``n_only`` is provided, the existing CSV is NOT deleted; rows
    matching any n in n_only are dropped and replaced (idempotent).
    Without n_only, the CSV is reset (delete-and-rewrite) as before.
    """
    csv_path = CSV_DIR / csv_filename(cfg.mode, cfg.exp, cfg.test, rep=rep)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    if n_only is None:
        if csv_path.exists():
            csv_path.unlink()
            logger.info("deleted existing %s", csv_path.name)
    else:
        if csv_path.exists():
            purge_n_rows(csv_path, n_only, logger)
        else:
            logger.warning(
                "--n-only specified but %s does not exist; "
                "creating fresh CSV with only the specified n value(s).",
                csv_path.name,
            )

    run_cell(rep, cfg, logger, csv_path, coords_shuffled_full)
    return csv_path


# ----------------------------------------------------------------------
# Whole-experiment driver
# ----------------------------------------------------------------------
def run_whole(
    cfg: ExpConfig,
    logger: logging.Logger,
    coords_shuffled_full: np.ndarray,
) -> Path:
    csv_path = CSV_DIR / csv_filename(cfg.mode, cfg.exp, cfg.test)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
        logger.info("deleted existing %s", csv_path.name)

    total_rows = 0
    t_start = time.perf_counter()
    for rep in range(cfg.n_reps):
        total_rows += run_cell(rep, cfg, logger, csv_path, coords_shuffled_full)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[mode=%s exp=%d] all reps done: %d rows in %.1fs (%.2fh) -> %s",
        cfg.mode, cfg.exp, total_rows, elapsed, elapsed / 3600, csv_path,
    )
    return csv_path


# ----------------------------------------------------------------------
# Merge per-rep CSVs into the canonical CSV
# ----------------------------------------------------------------------
def merge_per_rep_csvs(
    mode: str, exp: int, test: bool, logger: logging.Logger,
) -> Optional[Path]:
    canonical_name = csv_filename(mode, exp, test)
    stem = canonical_name[:-4]
    per_rep = sorted(
        CSV_DIR.glob(f"{stem}_rep*.csv"),
        key=lambda p: int(p.stem.replace(f"{stem}_rep", "")),
    )
    if not per_rep:
        logger.warning(
            "merge: no per-rep CSVs matching %s_rep*.csv under %s",
            stem, CSV_DIR,
        )
        return None

    canonical = CSV_DIR / canonical_name
    if canonical.exists():
        canonical.unlink()
    logger.info(
        "merge: %s <- %d per-rep file(s)", canonical_name, len(per_rep),
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
    logger.info("  -> %s", canonical)
    return canonical


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Experiment  runner: distance-query scaling vs n "
                     "on Census, swept by k. Modes: --semi or --full. "
                     "n grid is set by --exp E in 1..7."),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--semi", dest="mode", action="store_const", const="semi",
    )
    mode_group.add_argument(
        "--full", dest="mode", action="store_const", const="full",
    )
    parser.add_argument(
        "--exp", type=int, required=True,
        choices=range(EXP_MIN, EXP_MAX + 1), metavar="E",
    )
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tester-on", action="store_true")
    parser.add_argument(
        "--n-only", type=int, nargs="+", default=None, metavar="N",
        help=("Restrict the sweep to only these n values (must be a "
              "subset of the n_grid implied by --exp). Requires --rep. "
              "The existing per-rep CSV is preserved; rows matching "
              "any --n-only value are dropped and re-appended "
              "(idempotent re-run). Use to extend an existing run "
              "with one or more additional n tiers."),
    )
    args = parser.parse_args()

    # --n-only validation
    if args.n_only is not None:
        if args.rep is None:
            parser.error("--n-only requires --rep")
        if args.merge:
            parser.error("--n-only is not compatible with --merge")
        if args.test:
            parser.error("--n-only is not compatible with --test")

    mode = args.mode
    test = bool(args.test)
    exp = int(args.exp)
    cfg = make_config(mode, test, exp)
    scope = "test" if test else "full"
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Restrict n_grid if --n-only was supplied. Every requested n must
    # already be in the implied n_grid; we do not invent new tiers.
    if args.n_only is not None:
        implied = set(cfg.n_grid)
        bad = [n for n in args.n_only if n not in implied]
        if bad:
            parser.error(
                f"--n-only values {bad} are not in the n_grid for "
                f"--exp {exp} (which is {sorted(implied)})"
            )
        cfg = ExpConfig(
            mode=cfg.mode, test=cfg.test, exp=cfg.exp,
            n_grid=sorted(set(args.n_only)),
            k_grid=cfg.k_grid, n_reps=cfg.n_reps,
        )

    t_main_start = time.perf_counter()

    # --- Merge --------------------------------------------------------
    if args.merge:
        log_path = setup_logging(
            RESULTS_DIR, stamp, mode, scope, exp=exp, tag="merge",
        )
        logger = logging.getLogger(f"exp_{mode}_{scope}_merge")
        logger.info("merge mode=%s exp=%d test=%s", mode, exp, test)
        canonical = merge_per_rep_csvs(
            mode=mode, exp=exp, test=test, logger=logger,
        )
        if canonical is not None:
            logger.info("merge done: %s", canonical)
        else:
            logger.info("merge done: no canonical CSV produced.")
        log_elapsed_banner(logger, time.perf_counter() - t_main_start)
        return

    if test:
        alg_g_module.USE_ALG_TESTER = False
    else:
        alg_g_module.USE_ALG_TESTER = bool(args.tester_on)

    alg_g_module.USE_TESTER_CACHE = False

    # --- Per-rep ------------------------------------------------------
    if args.rep is not None:
        tag = f"rep{args.rep}"
        if args.n_only is not None:
            tag = f"{tag}_nonly"
        log_path = setup_logging(
            RESULTS_DIR, stamp, mode, scope, exp=exp, tag=tag,
        )
        logger = logging.getLogger(f"exp_{mode}_{scope}_{tag}")
        logger.info("=" * 70)
        logger.info(
            "Experiment  per-rep — rep=%d mode=%s exp=%d scope=%s — %s",
            args.rep, mode, exp, scope, stamp,
        )
        logger.info("=" * 70)
        logger.info("cfg:       %s", asdict(cfg))
        if args.n_only is not None:
            logger.info(
                "n-only mode: appending to existing CSV; "
                "n values to (re)run: %s", sorted(set(args.n_only)),
            )
        log_tester_banner(logger)
        logger.info("USE_TESTER_CACHE: %s", alg_g_module.USE_TESTER_CACHE)
        logger.info("log:       %s", log_path)

        coords_full = load_census(logger)
        coords_shuffled_full = shuffle_coords(coords_full, logger)

        t_start = time.perf_counter()
        csv_path = run_single_rep(
            args.rep, cfg, logger, coords_shuffled_full,
            n_only=args.n_only,
        )
        elapsed = time.perf_counter() - t_start
        logger.info("=" * 70)
        logger.info(
            "per-rep done in %.1fs (%.2fh). Wrote: %s",
            elapsed, elapsed / 3600, csv_path,
        )
        log_tester_banner(logger)
        log_elapsed_banner(logger, time.perf_counter() - t_main_start)
        return

    # --- Whole experiment --------------------------------------------
    log_path = setup_logging(RESULTS_DIR, stamp, mode, scope, exp=exp)
    logger = logging.getLogger(f"exp_{mode}_{scope}")

    logger.info("=" * 70)
    logger.info(
        "Experiment  (mode=%s, exp=%d, scope=%s) — %s",
        mode, exp, scope, stamp,
    )
    logger.info("=" * 70)
    logger.info("cfg:       %s", asdict(cfg))
    log_tester_banner(logger)
    logger.info("USE_TESTER_CACHE: %s", alg_g_module.USE_TESTER_CACHE)
    logger.info("log:       %s", log_path)

    coords_full = load_census(logger)
    coords_shuffled_full = shuffle_coords(coords_full, logger)

    t_start = time.perf_counter()
    csv_path = run_whole(cfg, logger, coords_shuffled_full)
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(
        "All done in %.1fs (%.2fh). Wrote: %s",
        elapsed, elapsed / 3600, csv_path,
    )
    log_tester_banner(logger)
    log_elapsed_banner(logger, time.perf_counter() - t_main_start)


if __name__ == "__main__":
    main()