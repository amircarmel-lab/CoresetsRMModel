# coreset_rm/config.py
"""Pipeline constants.

These are the four knobs the algorithm consumes. Everything else
(thresholds, ring assignments, d_max) is computed from the data at runtime.
"""
from dataclasses import dataclass

# Clustering exponent. Locked to 2 for k-means.
Z = 2

# Dislocation budget. In ProbSort terms this is the max rank error of the
# sorted order π; in our exact-sort-plus-noise stress test it is the
# perturbation size. Also determines the ambiguity-zone width (2*D+1)
# around each ring boundary in Step 4.
D_DISLOCATION = 30

# Rings per cluster. Ratio-2 bins covering [d_max^z / 2^L, d_max^z].
# Points below the smallest threshold go to the close set.
L_RINGS_PER_CLUSTER = 15

# Uniform samples drawn per ring in Step 5.
S_SAMPLES_PER_RING = 100

# Phase 2 target coreset size as a multiple of k. Final |S_2| = M_MULT * k.
M_MULT_PHASE2 = 40

@dataclass(frozen=True)
class PipelineConfig:
    D: int = D_DISLOCATION
    L: int = L_RINGS_PER_CLUSTER
    s_per_ring: int = S_SAMPLES_PER_RING
    m_mult_phase2: int = M_MULT_PHASE2