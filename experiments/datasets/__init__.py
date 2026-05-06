"""Dataset loaders and shared bundle type for experiments.

Every dataset module exposes a ``load()`` function returning a :class:`DatasetBundle`.
Preprocessed matrices and neural embeddings are cached under
:data:`DEFAULT_CACHE` so that heavy one-time computations (BGE-large,
DINOv2) are not redone on every notebook restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Cache root: ``experiments/cache/``. Resolved relative to this file so the
# layout works regardless of where the repo lives on disk. ``experiments``
# sits as a sibling of ``coreset_rm`` at the project root.
#   __file__                       -> .../experiments/datasets/__init__.py
#   .parent                        -> .../experiments/datasets/
#   .parent.parent                 -> .../experiments/
_EXPERIMENTS_DIR: Path = Path(__file__).resolve().parent.parent
DEFAULT_CACHE: Path = _EXPERIMENTS_DIR / "cache"


@dataclass(frozen=True)
class DatasetBundle:
    """Return type shared by all dataset loaders.

    Attributes
    ----------
    coords
        Coordinates fed to the clustering pipeline, shape ``(n, d)``,
        dtype ``float32``. For tabular datasets this is the preprocessed
        feature matrix; for text/image datasets it is the embedding selected
        by the loader's ``embedding`` argument (the "expensive" one by
        convention).
    labels
        Integer labels, shape ``(n,)``, or ``None`` for unlabelled data.
        For partially labelled datasets (e.g. STL-10 unlabelled split) we
        use ``-1`` rather than ``None`` so downstream code can index
        uniformly.
    metadata
        Free-form dict. By convention, text/image loaders put the cheap
        embedding under ``metadata["cheap_coords"]`` so the dual-embedding
        quadruplet oracle can be constructed from the bundle alone.
    """

    coords: np.ndarray
    labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            raise ValueError(
                f"coords must be 2-D; got shape {self.coords.shape}"
            )
        if self.labels is not None:
            if self.labels.ndim != 1:
                raise ValueError(
                    f"labels must be 1-D; got shape {self.labels.shape}"
                )
            if self.labels.shape[0] != self.coords.shape[0]:
                raise ValueError(
                    f"coords has {self.coords.shape[0]} rows but labels has "
                    f"{self.labels.shape[0]}"
                )


def ensure_subdir(*parts: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Resolve and create (if missing) a subdirectory of the cache root.

    Example
    -------
    >>> ensure_subdir("embeddings", "stl10")   # -> .../cache/embeddings/stl10/
    """
    path = cache_dir.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["DatasetBundle", "DEFAULT_CACHE", "ensure_subdir"]