"""Top-level pipeline. Composes Phase 1 + Phase 2 and returns the final coreset."""
from __future__ import annotations
from dataclasses import dataclass

from coreset_rm.config import PipelineConfig
from coreset_rm.instrumentation import Metrics, track
from coreset_rm.oracles import DistanceOracle, QuadrupletOracle
from coreset_rm.phase1 import run_phase1_mock
from coreset_rm.phase1 import run_phase1
from coreset_rm.phase2 import run_phase2, default_m
from coreset_rm.rng import RNG
from coreset_rm.types import WeightedPointSet


@dataclass
class PipelineOutput:
    S2: WeightedPointSet          # the final coreset
    centers: list[int]            # the k centers from Phase 1 (C*)
    metrics: Metrics


def run_pipeline(
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    quad_oracle: QuadrupletOracle | None,
    seed: int | None = None,
    m: int | None = None,
    config: PipelineConfig | None = None
) -> PipelineOutput:
    """Run the full pipeline: Phase 1 → Phase 2.

    Args:
        n:           number of points.
        k:           number of clusters.
        dist_oracle: distance oracle.
        quad_oracle: quadruplet oracle.
        seed:        top-level seed.
        m:           Phase 2 target size. Defaults to M_MULT_PHASE2 * k.

    Returns:
        PipelineOutput with the final coreset S_2, the k centers, and
        per-step metrics.
    """
    if m is None:
        if config is None:
            m = default_m(k)
        else:
            m = config.m_mult_phase2 * k

    root_rng = RNG(seed=seed)
    metrics = Metrics()

    with track(metrics, "phase1", dist=dist_oracle, quad=quad_oracle):
        phase1_out = run_phase1(
            n=n, k=k,
            dist_oracle=dist_oracle,
            quad_oracle=quad_oracle,
            root_rng=root_rng,
            metrics=metrics,
            config=config
        )
    metrics.intermediate_S1_size = len(phase1_out.S1)

    with track(metrics, "phase2", dist=dist_oracle, quad=quad_oracle):
        S2 = run_phase2(
            phase1_out=phase1_out,
            m=m,
            rng=root_rng.spawn("phase2_sampling").generator(),
        )
    metrics.final_S2_size = len(S2)
    metrics.coreset_plus_size = n

    return PipelineOutput(S2=S2, centers=phase1_out.centers, metrics=metrics)