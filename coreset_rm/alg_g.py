"""Alg-G: Coreset+ construction via a noisy quadruplet oracle.

Practical implementation of Alg-G from Raychaudhury, Esmailpour, Galhotra,
Sintos (2026), "Metric k-clustering using only Weak Comparison Oracles"
(arXiv:2601.19333), Appendix B.

The algorithmic structure tracks Section B.1 / B.2. Concrete numerical
choices follow the authors' reference notebook (`ICLR2026_Experiments`)
rather than the paper's analytical constants:

  - Sample sizes are small constants in k, not k·polylog n.
  - Coreset C contains only S^(1) samples plus the final tail (S^(2)
    points are auxiliary — used only to define the kernel/guard rank
    windows for filtering, never added to C).
  - Filtering keeps the bottom-3/4 of V_i \\ S_i by max_s Pcount_s(v),
    not the paper's threshold {v : max_s Pcount_s(v) < ⌊m_win/2⌋}.
  - Termination uses a constant threshold, not a log³n formula.
  - Safe-set fraction is 1/2 (not 1/4): V''_i is the top |V_i|/2 of V'_i
    by approximate-NN rank.

Programming choices we keep over the authors':

  - Batched oracle queries via QuadrupletOracle.compare_batch when
    available (purely amortizes Python overhead; query count unchanged).
  - Alg-Tester memoization within a round (sound by oracle persistence,
    Section 2 of the paper).
  - Per-round CorePrime precompute for Alg-Tester (one trim per pair,
    not per comparator call).
  - Iterative AdvSort (avoids Python recursion limits).
  - Seeded RNG via the project's RNG/spawn pattern.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from coreset_rm.config import D_DISLOCATION, PipelineConfig
from coreset_rm.oracles import QuadrupletOracle
from coreset_rm.prob_sort import prob_sort_edges
from coreset_rm.rng import RNG


# ----------------------------------------------------------------------
# Constants — match the authors' reference notebook.
# ----------------------------------------------------------------------
# Sample sizes (notebook lines: `s1_size = min(n_i, max(6, 3*k))` and
# `s2_size = min(n_i, max(24, 8*k, 80))`). Constant in n; only k drives
# them. These are dramatically smaller than the paper's k·polylog n
# bounds and are the main reason their |C| stays small.
S1_FLOOR = 6
S1_K_MULT = 3
S2_FLOOR = 80
S2_K_MULT = 8
S2_BASE_FLOOR = 24

# Kernel/guard size (notebook: `cgsize = max(12, int(1.2 * ell))` where
# ell = log_2(n_i)). Decoupled from the paper's dislocation parameter D.
CGSIZE_FLOOR = 12
CGSIZE_LOG_MULT = 1.2

# Guard placement offset (notebook: `Delta = 5`). Small additive gap
# between kernel and guard; guard width is cgsize/3.
DELTA = 5
GUARD_WIDTH_DIV = 3

# Filter retention: keep the BOTTOM fraction of V_i \ S_i by max-Pcount.
# Authors use 0.75 (drop top 25% closest-to-some-sample).
FILTER_KEEP_FRACTION = 0.75

# Safe-set fraction: notebook config sets SAFE_PREFIX_DIV = 2, meaning
# V''_i = top |V_i|/2 of V'_i by approximate-NN rank.
SAFE_SET_FRACTION = 0.5

# Termination threshold (notebook: THRESHOLD = 100). Constant, not
# log-scaled. Final V_tail goes into C with identity mapping.
TERMINATION_THRESHOLD = 100

# Recursion cap. Authors use O_log(n) = floor(log2(n)) + 1.
MAX_ROUNDS_MULT = 1
MAX_ROUNDS_OFFSET = 1

# If False, Alg-Tester collapses to a single direct oracle query per
# compare. Breaks the formal µ=1 emulation (paper Lemma B.3) and the
# 4-sorted guarantee (Lemma 2.2), but is much faster and works well at
# moderate noise. Authors' notebook always runs with the full tester.
USE_ALG_TESTER = True

# If True, cache Alg-Tester results within a round, keyed by ordered
# edge pair. Sound by oracle persistence.
USE_TESTER_CACHE = True

# If True, use QuadrupletOracle.compare_batch when the oracle exposes
# it. Pure Python-level speedup; query count is unchanged.
USE_BATCH_ORACLE = True

VERBOSE = False


# ----------------------------------------------------------------------
# Oracle batch helper
# ----------------------------------------------------------------------
def _compare_batch(
    quad_oracle: QuadrupletOracle,
    queries: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Run many compare(a,b,c,d) calls; return a bool array.

    Uses compare_batch if the oracle supports it; otherwise falls back
    to a Python loop. Query count is unchanged.
    """
    if USE_BATCH_ORACLE and hasattr(quad_oracle, "compare_batch"):
        return np.asarray(quad_oracle.compare_batch(queries), dtype=bool)
    out = np.empty(len(queries), dtype=bool)
    for i, (a, b, c, d) in enumerate(queries):
        out[i] = quad_oracle.compare(a, b, c, d)
    return out


# ----------------------------------------------------------------------
# Output dataclass (signature unchanged)
# ----------------------------------------------------------------------
@dataclass
class AlgGOutput:
    """Output of Alg-G.

    C:       point indices selected as representatives. Contains only
             S^(1) samples accumulated across rounds plus the final
             V_tail (deduplicated). S^(2) is NOT in C.
    M:       length-n list. M[i] ∈ {0, ..., |C|-1} is the index into C
             of the representative point i is mapped to.
    weights: length-|C| array. weights[j] = |{i : M[i] == j}|.
    """
    C: List[int]
    M: List[int]
    weights: np.ndarray


# ----------------------------------------------------------------------
# Per-run parameters
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Params:
    s1_size: int           # |S^(1)| target
    s2_size: int           # |S^(2)| target
    cgsize: int            # kernel size = guard offset start
    guard_width: int       # guard window width
    Delta: int             # gap between kernel and guard
    D: int                 # ProbSort dislocation bound (carried through, unused here)
    termination_threshold: int
    max_rounds: int


def _compute_params(
    n: int,
    k: int,
    config: PipelineConfig | None = None,
) -> _Params:
    """Compute per-run sample sizes and window widths.

    Follows the authors' notebook formulas:
      s1_size = max(6, 3·k)
      s2_size = max(24, 8·k, 80)
      cgsize  = max(12, ⌈1.2·log₂ n⌉)
      guard offset = cgsize + Delta, guard width = cgsize // 3
      termination = 100
      max_rounds  = ⌊log₂ n⌋ + 1
    """
    log_n = max(1, int(math.ceil(math.log2(max(n, 2)))))
    cgsize = max(CGSIZE_FLOOR, int(CGSIZE_LOG_MULT * log_n))
    guard_width = max(1, cgsize // GUARD_WIDTH_DIV)

    s1_size = max(S1_FLOOR, S1_K_MULT * k)
    s2_size = max(S2_BASE_FLOOR, S2_K_MULT * k, S2_FLOOR)

    if config is None:
        D = D_DISLOCATION
    else:
        D = config.D

    return _Params(
        s1_size=s1_size,
        s2_size=s2_size,
        cgsize=cgsize,
        guard_width=guard_width,
        Delta=DELTA,
        D=D,
        termination_threshold=TERMINATION_THRESHOLD,
        max_rounds=MAX_ROUNDS_MULT * log_n + MAX_ROUNDS_OFFSET,
    )


# ----------------------------------------------------------------------
# Entry point — signature unchanged
# ----------------------------------------------------------------------
def run_alg_g(
    n: int,
    k: int,
    quad_oracle: QuadrupletOracle,
    root_rng: RNG,
    config: PipelineConfig | None = None,
) -> AlgGOutput:
    """Construct a Coreset+ for (k, p)-clustering using only quadruplet queries."""
    _validate_inputs(n, k)

    params = _compute_params(n, k, config)
    rng = root_rng.spawn("alg_g").generator()

    if VERBOSE:
        print(
            f"[alg_g] n={n} k={k}  s1={params.s1_size} s2={params.s2_size}  "
            f"cgsize={params.cgsize} guard_w={params.guard_width}  "
            f"term={params.termination_threshold} max_rounds={params.max_rounds}  "
            f"tester={USE_ALG_TESTER} cache={USE_TESTER_CACHE} batch={USE_BATCH_ORACLE}"
        )

    M_point: List[int] = [-1] * n
    C_set: set = set()
    V_i: List[int] = list(range(n))

    for round_idx in range(params.max_rounds):
        if len(V_i) <= params.termination_threshold:
            break
        if VERBOSE:
            print(f"[alg_g round {round_idx + 1}] |V_i| = {len(V_i):,d}")
        V_i_new, round_updates = _run_round(V_i, quad_oracle, rng, params)
        if len(V_i_new) == len(V_i):
            # No progress this round (degenerate sample / empty V'). Stop.
            if VERBOSE:
                print("  [stop] no progress")
            break
        V_i = V_i_new
        for point, target in round_updates.items():
            M_point[point] = target
            C_set.add(target)

    # Tail: V_i becomes part of C with identity mapping.
    for v in V_i:
        M_point[v] = v
        C_set.add(v)

    if VERBOSE:
        print(
            f"[alg_g] done. |C|={len(C_set)} "
            f"total quads={quad_oracle.query_count:,d}"
        )

    return _build_output(M_point, C_set, n)


# ----------------------------------------------------------------------
# One round (steps 1–7 of Section B.1, authors' variant)
# ----------------------------------------------------------------------
def _run_round(
    V_i: List[int],
    quad_oracle: QuadrupletOracle,
    rng: np.random.Generator,
    params: _Params,
) -> Tuple[List[int], Dict[int, int]]:
    """Run one round. Returns (V_{i+1}, mapping_updates).

    `mapping_updates` keys are points finalized this round; values are
    their assigned representatives. Only S^(1) samples and V''_i points
    are added to the global mapping by run_alg_g.
    """
    q0_round = quad_oracle.query_count
    t0_round = time.perf_counter()

    # ---- Step 1: sample S^(1), S^(2). ----
    V_arr = np.asarray(V_i, dtype=np.int64)
    s1_n = min(params.s1_size, len(V_arr))
    s2_n = min(params.s2_size, len(V_arr))
    S1_raw = rng.choice(V_arr, size=s1_n, replace=False)
    S2_raw = rng.choice(V_arr, size=s2_n, replace=False)
    S1 = [int(x) for x in S1_raw]
    S2 = [int(x) for x in S2_raw]
    S_union = set(S1) | set(S2)

    if VERBOSE:
        print(f"  [step1 sample]           |S1|={len(S1)} |S2|={len(S2)}")

    if len(S1) < 2 or len(S2) < 2:
        # Degenerate sample. S^(1) still goes into C; S^(2) stays in
        # the active set (matches step-7 semantics: S^(2) is never
        # removed from V_i). Removing S^(2) here without recording it
        # in `updates` would orphan those points (no M_point entry, no
        # presence in next-round V_i, no tail-loop coverage).
        updates = {s: s for s in S1}
        S1_set = set(S1)
        V_new = [v for v in V_i if v not in S1_set]
        return V_new, updates

    # ---- Step 2: ProbSort X_i = E(S^(1), S^(2)). ----
    q0 = quad_oracle.query_count
    t0 = time.perf_counter()
    edges_X: List[Tuple[int, int]] = [(s, w) for s in S1 for w in S2]
    pi_X = prob_sort_edges(edges_X, quad_oracle)
    rank_X: Dict[Tuple[int, int], int] = {e: i for i, e in enumerate(pi_X)}
    if VERBOSE:
        print(
            f"  [step2 prob_sort_edges]  |edges|={len(edges_X):>7,d}  "
            f"quads={quad_oracle.query_count - q0:>8,d}  "
            f"time={time.perf_counter() - t0:5.2f}s"
        )

    # ---- Step 3: kernel and guard windows per s. ----
    t0 = time.perf_counter()
    kernel, guard = _extract_kernel_guard(
        S1, pi_X, params.cgsize, params.guard_width, params.Delta
    )
    if VERBOSE:
        avg_k = sum(len(v) for v in kernel.values()) / max(1, len(kernel))
        avg_g = sum(len(v) for v in guard.values()) / max(1, len(guard))
        print(
            f"  [step3 kernel/guard]     avg|kernel|={avg_k:.0f} "
            f"avg|guard|={avg_g:.0f}  "
            f"time={time.perf_counter() - t0:5.2f}s"
        )

    # ---- Step 4: filter to V'_i (bottom 3/4 by max-Pcount). ----
    q0 = quad_oracle.query_count
    t0 = time.perf_counter()
    V_prime = _filter_active_set(V_i, S_union, S1, guard, quad_oracle)
    if VERBOSE:
        denom = max(1, len(V_i) - len(S_union))
        print(
            f"  [step4 filter]           |V'|={len(V_prime):>6,d}  "
            f"|V'|/|V\\S|={len(V_prime) / denom:.2f}  "
            f"quads={quad_oracle.query_count - q0:>8,d}  "
            f"time={time.perf_counter() - t0:5.2f}s"
        )

    if not V_prime:
        # Nothing to map this round; S^(1) goes into C, S^(2) stays
        # in the active set (matches step-7 semantics).
        updates = {s: s for s in S1}
        S1_set = set(S1)
        V_new = [v for v in V_i if v not in S1_set]
        return V_new, updates

    # ---- Step 5: precompute CorePrime, then AdvSort Y_i. ----
    q0 = quad_oracle.query_count
    t0 = time.perf_counter()

    # CorePrime[s1][s2] is the trimmed kernel of the trim-side for the
    # ordered pair (s1, s2). Computed once per round, used inside Alg-Tester.
    CorePrime = _build_core_prime(S1, kernel, rank_X, params.cgsize, params.Delta)

    edges_Y: List[Tuple[int, int]] = [(s, v) for s in S1 for v in V_prime]
    tester_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}

    def comparator(e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
        if USE_TESTER_CACHE:
            hit = tester_cache.get((e1, e2))
            if hit is not None:
                return hit
            flipped = tester_cache.get((e2, e1))
            if flipped is not None:
                return not flipped
        result = _alg_tester(e1, e2, kernel, CorePrime, quad_oracle)
        if USE_TESTER_CACHE:
            tester_cache[(e1, e2)] = result
        return result

    pi_Y = _adv_sort(edges_Y, comparator, rng)
    rank_Y: Dict[Tuple[int, int], int] = {e: i for i, e in enumerate(pi_Y)}
    if VERBOSE:
        print(
            f"  [step5 adv_sort+tester]  |edges|={len(edges_Y):>7,d}  "
            f"cached={len(tester_cache)}  "
            f"quads={quad_oracle.query_count - q0:>8,d}  "
            f"time={time.perf_counter() - t0:5.2f}s"
        )

    # f_v = first incident edge in π_Y for each v.
    f_v: Dict[int, Tuple[int, int]] = {}
    for edge in pi_Y:
        v = edge[1]
        if v not in f_v:
            f_v[v] = edge

    # ---- Step 6: safe-set V''_i and per-round mapping. ----
    safe_size = max(1, int(len(V_i) * SAFE_SET_FRACTION))
    V_prime_by_nn = sorted(V_prime, key=lambda v: rank_Y[f_v[v]])
    V_double_prime = V_prime_by_nn[:safe_size]

    # Only S^(1) self-maps into C; S^(2) is NOT a center.
    updates: Dict[int, int] = {s: s for s in S1}
    for v in V_double_prime:
        s_endpoint = f_v[v][0]  # always an S^(1) point by construction
        updates[v] = s_endpoint

    # ---- Step 7: V_{i+1} = V_i \ (V''_i ∪ S^(1)). ----
    # Note: S^(2) points are NOT removed. They re-enter the active set
    # and may be sampled, mapped, or end up in V_tail in later rounds.
    removed = set(S1) | set(V_double_prime)
    V_new = [v for v in V_i if v not in removed]

    if VERBOSE:
        print(
            f"  [round total]            |V_i|={len(V_i):>6,d} -> "
            f"|V_i+1|={len(V_new):>6,d}  "
            f"quads={quad_oracle.query_count - q0_round:>8,d}  "
            f"time={time.perf_counter() - t0_round:5.2f}s"
        )

    return V_new, updates


# ----------------------------------------------------------------------
# Kernel / guard extraction
# ----------------------------------------------------------------------
def _extract_kernel_guard(
    S1: Sequence[int],
    pi_X: Sequence[Tuple[int, int]],
    cgsize: int,
    guard_width: int,
    Delta: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """For each s ∈ S^(1), pull its sub-order from π_X and slice
    kernel = top cgsize, guard = ranks (cgsize+Δ, cgsize+Δ+guard_width].

    Authors' notebook semantics (1-indexed in the notebook; we use the
    equivalent slicing in 0-indexed arrays).
    """
    per_s: Dict[int, List[int]] = {s: [] for s in S1}
    for (u, w) in pi_X:
        if u in per_s:
            per_s[u].append(w)

    kernel: Dict[int, List[int]] = {}
    guard: Dict[int, List[int]] = {}
    guard_lo = cgsize + Delta
    guard_hi = cgsize + Delta + guard_width
    for s, ws in per_s.items():
        kernel[s] = ws[:cgsize]
        guard[s] = ws[guard_lo:guard_hi]
    return kernel, guard


# ----------------------------------------------------------------------
# Filter — bottom 3/4 by max Pcount (authors' rank-based variant).
# ----------------------------------------------------------------------
def _filter_active_set(
    V_i: Sequence[int],
    S_union: set,
    S1: Sequence[int],
    guard: Dict[int, List[int]],
    quad_oracle: QuadrupletOracle,
) -> List[int]:
    """V'_i = bottom-FILTER_KEEP_FRACTION of V_i \\ S_i by max-Pcount.

    Pcount_s(v) = #{g ∈ Guard(s) : Q̃({s,v}, {s,g}) returns "d(s,v) ≤ d(s,g)"}.
    HIGH max-Pcount means v is close to some sample → likely a noisy/boundary
    point we don't want to map this round. We sort all non-sample v by
    max-Pcount ascending and keep the bottom 3/4.

    No short-circuit: we need the actual values, not just "above/below
    a threshold," because the filter is rank-based.

    Implementation note: queries are batched per (v, s) pair so the
    inner Python loop over guards is folded into one numpy call.
    """
    candidates = [int(v) for v in V_i if v not in S_union]
    if not candidates:
        return []

    # max_pcount[i] is max_s Pcount_s(v) for candidates[i].
    max_pcount = np.zeros(len(candidates), dtype=np.int64)

    # Iterate over s, compute Pcount_s(v) for all v in one batch each,
    # take running max.
    for s in S1:
        guard_s = guard.get(s, ())
        if not guard_s:
            continue
        gs_list = list(guard_s)
        # Build all (s, v, s, g) queries: candidates × guards.
        queries = [
            (s, v, s, g) for v in candidates for g in gs_list
        ]
        results = _compare_batch(quad_oracle, queries)
        # Reshape to (|candidates|, |guards|), sum across guards.
        pcount_s = results.reshape(len(candidates), len(gs_list)).sum(axis=1)
        np.maximum(max_pcount, pcount_s, out=max_pcount)

    # Bottom FILTER_KEEP_FRACTION by max_pcount (ascending).
    keep_n = int(len(candidates) * FILTER_KEEP_FRACTION)
    if keep_n <= 0:
        return []
    order = np.argsort(max_pcount, kind="stable")
    keep_idx = order[:keep_n]
    return [candidates[i] for i in keep_idx]


# ----------------------------------------------------------------------
# Per-round CorePrime precompute (used by Alg-Tester)
# ----------------------------------------------------------------------
def _build_core_prime(
    S1: Sequence[int],
    kernel: Dict[int, List[int]],
    rank_X: Dict[Tuple[int, int], int],
    cgsize: int,
    Delta: int,
) -> Dict[int, Dict[int, List[int]]]:
    """For every ordered pair (s1, s2) with s1 < s2 in S^(1), determine
    which side carries e* in π_Z and pre-trim the OTHER side's kernel.

    `CorePrime[s1][s2]` returns the trimmed kernel that Alg-Tester uses
    when comparing two edges with primary samples (s1, s2). One direction
    of each pair gets the trimmed list; the other gets an empty list
    (since by symmetry only the trim-side carries the vote).

    For pairs `(s, s)` (same primary sample, "Case b" of Section B.2),
    the kernel is used as-is — handled inline in `_alg_tester`.
    """
    CorePrime: Dict[int, Dict[int, List[int]]] = {s: {} for s in S1}

    for i, s1 in enumerate(S1):
        ker_s1 = kernel.get(s1, [])
        for s2 in S1[i + 1:]:
            ker_s2 = kernel.get(s2, [])

            # Build π_Z = the order over E(s1, ker(s1)) ∪ E(s2, ker(s2))
            # induced by π_X (only the ranks we need; sort by rank).
            z_ranks: List[Tuple[Tuple[int, int], int]] = []
            for w in ker_s1:
                rk = rank_X.get((s1, w))
                if rk is not None:
                    z_ranks.append(((s1, w), rk))
            for w in ker_s2:
                rk = rank_X.get((s2, w))
                if rk is not None:
                    z_ranks.append(((s2, w), rk))

            if len(z_ranks) < 2:
                CorePrime[s1][s2] = []
                CorePrime[s2][s1] = []
                continue

            z_ranks.sort(key=lambda t: t[1])
            pi_Z = [edge for edge, _ in z_ranks]
            e_star_s = pi_Z[-1][0]

            # The side containing e* is the "main" side; the other is
            # "trim". We pre-trim the trim-side's kernel by removing
            # entries whose π_Z rank lies in the boundary band
            # [2*cgsize - Delta, 2*cgsize] (1-indexed in the paper /
            # notebook). In 0-indexed terms: positions in
            # [2*cgsize - Delta - 1, 2*cgsize - 1).
            pos_of = {edge: pos for pos, edge in enumerate(pi_Z)}
            lo = 2 * cgsize - Delta - 1
            hi = 2 * cgsize - 1

            if e_star_s == s1:
                # s1 main, s2 trim
                trimmed = []
                for w in ker_s2:
                    pos = pos_of.get((s2, w))
                    if pos is None:
                        continue
                    if not (lo <= pos < hi):
                        trimmed.append(w)
                CorePrime[s1][s2] = trimmed
                CorePrime[s2][s1] = []
            else:
                # s2 main, s1 trim
                trimmed = []
                for w in ker_s1:
                    pos = pos_of.get((s1, w))
                    if pos is None:
                        continue
                    if not (lo <= pos < hi):
                        trimmed.append(w)
                CorePrime[s2][s1] = trimmed
                CorePrime[s1][s2] = []

    return CorePrime


# ----------------------------------------------------------------------
# Alg-Tester (Section B.2) — uses precomputed CorePrime.
# ----------------------------------------------------------------------
def _alg_tester(
    e1: Tuple[int, int],
    e2: Tuple[int, int],
    kernel: Dict[int, List[int]],
    CorePrime: Dict[int, Dict[int, List[int]]],
    quad_oracle: QuadrupletOracle,
) -> bool:
    """Emulate a µ=1 adversarial oracle judging d(e1) ≤ d(e2).

    Three cases:
      1. s1 not in kernel-table or s2 not in kernel-table: trivial True.
         (Defensive; should not happen in normal flow.)
      2. s1 == s2: use kernel(s1) directly, no trimming. Vote with
         compare(s1, v1, vc, v2) for vc ∈ kernel(s1).
      3. s1 != s2: use CorePrime[s1][s2] if non-empty (means s1 is the
         main side); else use CorePrime[s2][s1] (s2 is main side).

    With USE_ALG_TESTER = False, collapse to one direct oracle query.
    """
    s1, v1 = e1
    s2, v2 = e2

    if not USE_ALG_TESTER:
        return quad_oracle.compare(s1, v1, s2, v2)

    if s1 not in kernel or s2 not in kernel:
        return True

    if s1 == s2:
        kernel_prime = kernel.get(s1, [])
        if not kernel_prime:
            return True
        # Majority: count vc ∈ kernel(s1) for which oracle says
        # d(s1, v1) > d(vc, v2), i.e., compare returns False.
        queries = [(s1, v1, vc, v2) for vc in kernel_prime]
        results = _compare_batch(quad_oracle, queries)
        t_count = int((~results).sum())
        return not (t_count > len(kernel_prime) // 2)

    # s1 != s2.
    cp_12 = CorePrime.get(s1, {}).get(s2, [])
    cp_21 = CorePrime.get(s2, {}).get(s1, [])

    if cp_12:
        # s1 is main; trim-side kernel is cp_12 (subset of kernel(s2)).
        # Vote: count vc where d(s1,v1) > d(vc,v2).
        queries = [(s1, v1, vc, v2) for vc in cp_12]
        results = _compare_batch(quad_oracle, queries)
        t_count = int((~results).sum())
        return not (t_count > len(cp_12) // 2)
    if cp_21:
        # s2 is main; trim-side kernel is cp_21 (subset of kernel(s1)).
        # Vote: count vc where d(s2,v2) > d(vc,v1).
        queries = [(s2, v2, vc, v1) for vc in cp_21]
        results = _compare_batch(quad_oracle, queries)
        t_count = int((~results).sum())
        return t_count > len(cp_21) // 2

    return True


# ----------------------------------------------------------------------
# AdvSort: iterative randomized quicksort over edges with a comparator.
# ----------------------------------------------------------------------
def _adv_sort(
    items: Sequence,
    comparator,
    rng: np.random.Generator,
) -> list:
    """Randomized quicksort using `comparator(a, b) -> bool` (a ≤ b).

    Iterative (explicit stack) to sidestep Python's recursion limit on
    the |edges_Y| inputs we pass in — these can be in the tens of
    thousands at moderate n.
    """
    arr = list(items)
    n_items = len(arr)
    if n_items <= 1:
        return arr
    stack: List[Tuple[int, int]] = [(0, n_items - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        pivot_idx = int(rng.integers(lo, hi + 1))
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if comparator(arr[j], pivot):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        p = i + 1
        stack.append((lo, p - 1))
        stack.append((p + 1, hi))
    return arr


# ----------------------------------------------------------------------
# Output assembly
# ----------------------------------------------------------------------
def _build_output(
    M_point: List[int],
    C_set: Iterable[int],
    n: int,
) -> AlgGOutput:
    """Build AlgGOutput from per-point targets and the C set."""
    C = sorted(C_set)
    idx_of = {c: j for j, c in enumerate(C)}
    M = [0] * n
    for i in range(n):
        target = M_point[i]
        if target < 0 or target not in idx_of:
            raise RuntimeError(f"point {i} has no representative in C")
        M[i] = idx_of[target]
    weights = np.zeros(len(C), dtype=np.float64)
    for j in M:
        weights[j] += 1.0
    return AlgGOutput(C=C, M=M, weights=weights)


def _validate_inputs(n: int, k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")