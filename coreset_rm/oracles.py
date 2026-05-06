"""Oracle protocols and Euclidean implementations.

The pipeline works on integer point indices only. Coordinates live
privately inside the oracle constructors; nothing else sees them.

Persistence for the noisy quadruplet oracle is implicit: the noise for
each distinct pair-of-edges is determined by a deterministic hash of
(seed, canonical_key). No per-query storage.
"""
from __future__ import annotations
from typing import Protocol, Sequence
import numpy as np


# ----------------------------------------------------------------------
#  Protocols
# ----------------------------------------------------------------------

class QuadrupletOracle(Protocol):
    """Noisy comparison oracle.

    compare(a, b, c, d) returns True iff the oracle says d(a,b) <= d(c,d).
    Correct with probability at least 1 - phi. Persistent: the same query
    always returns the same answer. Pair-order antisymmetry holds:
    compare(a, b, c, d) == not compare(c, d, a, b).
    """
    query_count: int
    def compare(self, a: int, b: int, c: int, d: int) -> bool: ...


class DistanceOracle(Protocol):
    """Exact distance oracle."""
    query_count: int
    def distance(self, u: int, v: int) -> float: ...
    def distance_one_to_many(self, u: int, vs: Sequence[int]) -> np.ndarray: ...


# ----------------------------------------------------------------------
#  Hash helper: deterministic 64-bit hash for implicit persistence
# ----------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_PHI_SCALE = float(1 << 64)


def _splitmix64(x: int) -> int:
    """splitmix64 finalizer. Fast deterministic 64-bit avalanche."""
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _hash_canonical(seed: int, a: int, b: int, c: int, d: int) -> int:
    """Hash a canonical (seed, four-int) tuple to a 64-bit value."""
    h = seed & _MASK64
    for v in (a, b, c, d):
        h = _splitmix64((h ^ (v & _MASK64)) & _MASK64)
    return h


# ----------------------------------------------------------------------
#  Euclidean implementations
# ----------------------------------------------------------------------

class EuclideanDistanceOracle:
    """Exact Euclidean distance oracle backed by a coordinate array."""

    __slots__ = ("_coords", "query_count")

    def __init__(self, coords: np.ndarray):
        self._coords = np.ascontiguousarray(coords, dtype=np.float64)
        self.query_count = 0

    @property
    def n(self) -> int:
        return self._coords.shape[0]

    @property
    def d(self) -> int:
        return self._coords.shape[1]

    def distance(self, u: int, v: int) -> float:
        self.query_count += 1
        diff = self._coords[u] - self._coords[v]
        return float(np.sqrt(diff @ diff))

    def distance_one_to_many(self, u: int, vs: Sequence[int]) -> np.ndarray:
        vs_arr = np.asarray(vs, dtype=np.int64)
        self.query_count += int(vs_arr.size)
        diffs = self._coords[vs_arr] - self._coords[u]
        return np.sqrt(np.einsum("ij,ij->i", diffs, diffs))


class EuclideanQuadrupletOracle:
    """Noisy quadruplet oracle with hash-based persistence.

    For each canonical pair-of-edges, a deterministic hash of
    (seed, canonical_key) decides whether the true answer is flipped.
    Within-pair symmetry ({a,b} = {b,a}) and pair-order antisymmetry are
    enforced by canonicalization before hashing; no per-query storage.
    """

    __slots__ = ("_coords", "_seed", "_phi_threshold", "query_count")

    def __init__(self, coords: np.ndarray, phi: float, seed: int):
        self._coords = np.ascontiguousarray(coords, dtype=np.float64)
        self._seed = seed & _MASK64
        # Canonical hash < threshold flips the answer; gives marginal rate phi.
        self._phi_threshold = int(phi * _PHI_SCALE)
        self.query_count = 0

    def compare(self, a: int, b: int, c: int, d: int) -> bool:
        self.query_count += 1

        # Canonicalize each edge: (min, max).
        e1_lo, e1_hi = (a, b) if a <= b else (b, a)
        e2_lo, e2_hi = (c, d) if c <= d else (d, c)

        # Identical edges: d(e,e) ≤ d(e,e) is trivially true, noise-free.
        if (e1_lo, e1_hi) == (e2_lo, e2_hi):
            return True

        # Canonicalize pair order: lex-smaller edge first; flip the answer
        # if we swapped, to keep pair-order antisymmetry.
        if (e1_lo, e1_hi) <= (e2_lo, e2_hi):
            swapped = False
            k1_lo, k1_hi, k2_lo, k2_hi = e1_lo, e1_hi, e2_lo, e2_hi
        else:
            swapped = True
            k1_lo, k1_hi, k2_lo, k2_hi = e2_lo, e2_hi, e1_lo, e1_hi

        diff1 = self._coords[k1_lo] - self._coords[k1_hi]
        diff2 = self._coords[k2_lo] - self._coords[k2_hi]
        canonical_true = bool((diff1 @ diff1) <= (diff2 @ diff2))

        h = _hash_canonical(self._seed, k1_lo, k1_hi, k2_lo, k2_hi)
        flip = h < self._phi_threshold
        canonical_answer = canonical_true != flip

        return (not canonical_answer) if swapped else canonical_answer

# ----------------------------------------------------------------------
#  Dual-embedding quadruplet oracle (experiment instrumentation)
# ----------------------------------------------------------------------

class DualEmbeddingQuadrupletOracle:
    """Quadruplet oracle whose answers come from a 'cheap' embedding.

    Unlike :class:`EuclideanQuadrupletOracle`, the noise here is not
    synthetic. ``compare(a, b, c, d)`` returns the Euclidean comparison
    on the cheap embedding. Whenever the cheap embedding disagrees
    with the expensive embedding on the ordering, that counts as noise;
    the effective noise rate phi is a property of the embedding pair
    and can be measured empirically (e.g. via
    ``experiments.evaluation.measure_empirical_phi``).

    Persistence is automatic: the coords do not change during the
    oracle's lifetime, so the same query always returns the same
    answer. No seed, no hash, no per-query storage.

    The constructor takes both embeddings but stores only the cheap
    one. The expensive embedding is consumed purely for invariant
    checking (same number of points). This documents the oracle's
    identity as being tied to a pair of embeddings, even though only
    cheap is used to answer queries.

    Pair-order antisymmetry and within-pair symmetry (the {a,b}={b,a}
    convention) are enforced by canonicalization, matching the
    semantics of :class:`EuclideanQuadrupletOracle`.
    """

    __slots__ = ("_cheap", "query_count")

    def __init__(
        self,
        cheap_coords: np.ndarray,
        expensive_coords: np.ndarray,
    ):
        if cheap_coords.shape[0] != expensive_coords.shape[0]:
            raise ValueError(
                f"cheap and expensive coords must agree on n; "
                f"got {cheap_coords.shape[0]} vs "
                f"{expensive_coords.shape[0]}"
            )
        self._cheap = np.ascontiguousarray(cheap_coords, dtype=np.float64)
        self.query_count = 0

    @property
    def n(self) -> int:
        return self._cheap.shape[0]

    @property
    def d_cheap(self) -> int:
        return self._cheap.shape[1]

    def compare(self, a: int, b: int, c: int, d: int) -> bool:
        self.query_count += 1

        # Canonicalize each edge: (min, max).
        e1_lo, e1_hi = (a, b) if a <= b else (b, a)
        e2_lo, e2_hi = (c, d) if c <= d else (d, c)

        # Identical edges: d(e,e) <= d(e,e) is trivially true.
        if (e1_lo, e1_hi) == (e2_lo, e2_hi):
            return True

        # Canonicalize pair order: lex-smaller edge first; negate if we
        # swapped, to keep pair-order antisymmetry on exact equality.
        if (e1_lo, e1_hi) <= (e2_lo, e2_hi):
            swapped = False
            k1_lo, k1_hi, k2_lo, k2_hi = e1_lo, e1_hi, e2_lo, e2_hi
        else:
            swapped = True
            k1_lo, k1_hi, k2_lo, k2_hi = e2_lo, e2_hi, e1_lo, e1_hi

        diff1 = self._cheap[k1_lo] - self._cheap[k1_hi]
        diff2 = self._cheap[k2_lo] - self._cheap[k2_hi]
        canonical_answer = bool((diff1 @ diff1) <= (diff2 @ diff2))

        return (not canonical_answer) if swapped else canonical_answer


class MixedQuadrupletOracle:
    """Quadruplet oracle routing between cheap and expensive per query.


    Parameters
    ----------
    cheap_coords
        ``(n, d_cheap)`` array. The noisy embedding; matches phi_real
        against ``expensive_coords`` on random quadruplets.
    expensive_coords
        ``(n, d_expensive)`` array. Treated as ground truth — queries
        routed here are noise-free by construction.
    beta
        Routing probability toward expensive coords. ``beta = 0``
        reduces to the cheap-only oracle (phi = phi_real). ``beta = 1``
        reduces to the exact oracle (phi = 0).
    seed
        Seed for the routing hash. Different seeds give statistically
        independent routing patterns; each is itself deterministic.

    Attributes
    ----------
    query_count
        Total number of :meth:`compare` calls.
    routed_to_expensive
        Of those, how many were answered via ``expensive_coords``.
        Diagnostic — the expected value over random quadruplets is
        ``beta * query_count``.
    """

    __slots__ = (
        "_cheap", "_expensive", "_beta", "_seed", "_threshold",
        "query_count", "routed_to_expensive",
    )

    def __init__(
            self,
            cheap_coords: np.ndarray,
            expensive_coords: np.ndarray,
            beta: float,
            seed: int,
    ):
        if cheap_coords.shape[0] != expensive_coords.shape[0]:
            raise ValueError(
                f"cheap_coords and expensive_coords must agree on n; "
                f"got {cheap_coords.shape[0]} vs "
                f"{expensive_coords.shape[0]}"
            )
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1]; got {beta}")

        self._cheap = np.ascontiguousarray(cheap_coords, dtype=np.float64)
        self._expensive = np.ascontiguousarray(
            expensive_coords, dtype=np.float64
        )
        self._beta = float(beta)
        self._seed = int(seed)
        # Hash output is a uniform 64-bit int in [0, 2^64). Comparing
        # against int(beta * 2^64) gives a Bernoulli(beta) coin per
        # distinct canonical key. Endpoints behave correctly:
        #   beta = 0 -> threshold = 0        -> never routes expensive
        #   beta = 1 -> threshold = 2^64     -> always routes expensive
        self._threshold = int(self._beta * _PHI_SCALE)
        self.query_count = 0
        self.routed_to_expensive = 0

    @property
    def n(self) -> int:
        return self._cheap.shape[0]

    def compare(self, a: int, b: int, c: int, d: int) -> bool:
        self.query_count += 1

        # Canonicalize each edge: (min, max).
        e1_lo, e1_hi = (a, b) if a <= b else (b, a)
        e2_lo, e2_hi = (c, d) if c <= d else (d, c)

        # Identical edges: d(e,e) <= d(e,e) is trivially true. Also
        # avoids a zero diff and a spurious hash call.
        if (e1_lo, e1_hi) == (e2_lo, e2_hi):
            return True

        # Canonicalize pair order so that swapping (a,b) with (c,d)
        # maps to the same canonical key; negate the answer if we
        # swapped to preserve pair-order antisymmetry.
        if (e1_lo, e1_hi) <= (e2_lo, e2_hi):
            swapped = False
            k1_lo, k1_hi, k2_lo, k2_hi = e1_lo, e1_hi, e2_lo, e2_hi
        else:
            swapped = True
            k1_lo, k1_hi, k2_lo, k2_hi = e2_lo, e2_hi, e1_lo, e1_hi

        # Deterministic routing keyed on (seed, canonical_key).
        h = _hash_canonical(self._seed, k1_lo, k1_hi, k2_lo, k2_hi)
        if h < self._threshold:
            self.routed_to_expensive += 1
            coords = self._expensive
        else:
            coords = self._cheap

        diff1 = coords[k1_lo] - coords[k1_hi]
        diff2 = coords[k2_lo] - coords[k2_hi]
        canonical_answer = bool((diff1 @ diff1) <= (diff2 @ diff2))

        return (not canonical_answer) if swapped else canonical_answer
