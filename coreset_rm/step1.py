"""Step 1 orchestrator: Alg-G → extract k centers → assign → ClusterPartition."""
from __future__ import annotations

from coreset_rm.alg_g import run_alg_g, AlgGOutput
from coreset_rm.config import PipelineConfig
from coreset_rm.extract_centers import extract_centers
from coreset_rm.oracles import DistanceOracle, QuadrupletOracle
from coreset_rm.rng import RNG
from coreset_rm.types import ClusterPartition


def run_step1(
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    quad_oracle: QuadrupletOracle,
    root_rng: RNG,
    config: PipelineConfig | None = None,
    algoutput: AlgGOutput | None = None,
) -> ClusterPartition:
    """Run Step 1: produce (C, M) via Alg-G, then extract k centers via k-means++."""
    if algoutput is None:
        alg_g_out = run_alg_g(
            n=n, k=k, quad_oracle=quad_oracle,
            root_rng=root_rng.spawn("step1_alg_g"),
            config=config,
        )
    else:
        alg_g_out = algoutput
    return extract_centers(
        alg_g_out=alg_g_out,
        n=n, k=k, dist_oracle=dist_oracle,
        root_rng=root_rng.spawn("step1_extract"),
    )