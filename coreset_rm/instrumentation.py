from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator

from coreset_rm.oracles import DistanceOracle, QuadrupletOracle


@dataclass
class Metrics:
    """Per-step counts and timings for a pipeline run."""
    quadruplet_queries: Dict[str, int] = field(default_factory=dict)
    distance_queries: Dict[str, int] = field(default_factory=dict)
    wall_time: Dict[str, float] = field(default_factory=dict)
    coreset_plus_size: int = 0
    intermediate_S1_size: int = 0
    final_S2_size: int = 0

    def to_dict(self) -> dict:
        return {
            "quadruplet_queries": dict(self.quadruplet_queries),
            "distance_queries": dict(self.distance_queries),
            "wall_time": dict(self.wall_time),
            "coreset_plus_size": self.coreset_plus_size,
            "intermediate_S1_size": self.intermediate_S1_size,
            "final_S2_size": self.final_S2_size,
        }

    def pretty(self) -> str:
        lines = ["Metrics:"]
        lines.append("  sizes: |C|={}, |S1|={}, |S2|={}".format(
            self.coreset_plus_size,
            self.intermediate_S1_size,
            self.final_S2_size,
        ))
        lines.append("  wall time (s):")
        for step, t in self.wall_time.items():
            lines.append(f"    {step:30s} {t:8.3f}")
        lines.append("  quadruplet queries:")
        for step, q in self.quadruplet_queries.items():
            lines.append(f"    {step:30s} {q:>12,d}")
        lines.append("  distance queries:")
        for step, q in self.distance_queries.items():
            lines.append(f"    {step:30s} {q:>12,d}")
        return "\n".join(lines)


@contextmanager
def track(
    metrics: Metrics,
    step: str,
    quad: QuadrupletOracle | None = None,
    dist: DistanceOracle | None = None,
) -> Iterator[None]:
    """Context manager: record wall time and oracle-query deltas for a step.

    Usage:
        with track(metrics, "step3_dmax", quad=quad, dist=dist):
            do_stuff()

    Reads oracle query counts before and after the block. If the same step
    name is used multiple times, counts accumulate.
    """
    q0 = quad.query_count if quad is not None else 0
    d0 = dist.query_count if dist is not None else 0
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        metrics.wall_time[step] = metrics.wall_time.get(step, 0.0) + dt
        if quad is not None:
            delta_q = quad.query_count - q0
            metrics.quadruplet_queries[step] = (
                metrics.quadruplet_queries.get(step, 0) + delta_q
            )
        if dist is not None:
            delta_d = dist.query_count - d0
            metrics.distance_queries[step] = (
                metrics.distance_queries.get(step, 0) + delta_d
            )