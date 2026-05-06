"""Seeded RNG with deterministic, name-addressed substream spawning.

One top-level seed per pipeline run. Components ask for named
substreams; each substream is derived deterministically from the parent
seed and the name, so substreams are reproducible regardless of the
order or count of sibling spawn() calls.
"""
from __future__ import annotations
import hashlib
import numpy as np


class RNG:
    """Seeded RNG with deterministic name-addressed children.

    Usage:
        root = RNG(seed=42)
        sampler_rng = root.spawn("step5_ring_sampling")
        gen = sampler_rng.generator()
        gen.choice(...)

    Conventions:
      - Call `spawn(name)` to get an independent substream.
      - Call `generator()` once per RNG and thread the result through.
        Calling `generator()` twice on the same RNG yields identical
        streams (SeedSequence is deterministic).
    """

    __slots__ = ("_seed_seq",)

    def __init__(self, seed: int | np.random.SeedSequence):
        if isinstance(seed, np.random.SeedSequence):
            self._seed_seq = seed
        else:
            self._seed_seq = np.random.SeedSequence(seed)

    def generator(self) -> np.random.Generator:
        """A numpy Generator drawn from this stream."""
        return np.random.default_rng(self._seed_seq)

    def spawn(self, name: str) -> "RNG":
        """Derive a named child RNG. Deterministic in (parent, name).

        Spawning the same name twice returns equivalent streams.
        """
        tag = int.from_bytes(
            hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest(),
            "little",
        )
        child_seq = np.random.SeedSequence(
            entropy=self._seed_seq.entropy,
            spawn_key=self._seed_seq.spawn_key + (tag,),
        )
        return RNG(child_seq)

    def integer_seed(self) -> int:
        """A 64-bit integer seed, for components that need a raw int
        (e.g. EuclideanQuadrupletOracle's persistence hash)."""
        (val,) = self._seed_seq.generate_state(1, dtype=np.uint64)
        return int(val)