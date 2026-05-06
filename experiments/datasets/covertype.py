"""Covertype (581,012 points, 54 features, 7 classes) loader.

Source: UCI Machine Learning Repository via :mod:`ucimlrepo`
(``fetch_ucirepo(id=31)``). This matches the fetch path used in the
existing ``experiment_running_time_test`` diagnostic.

Preprocessing: scale all 54 features to ``[0, 1]`` with
:class:`sklearn.preprocessing.MinMaxScaler`. Without scaling, Elevation
(range ~1800-3850) dominates Euclidean distance and the clustering
effectively reduces to 1-D. MinMax caps every feature's per-column
contribution to squared distance at 1, so no single column can outvote
the others, while preserving the ``{0, 1}`` values of the wilderness
and soil one-hot columns (unlike StandardScaler, which would distort
them).

Labels are native 1-7; we shift to 0-6 so downstream code can index
freely.

The preprocessed matrix and labels are cached as separate ``.npy`` files
under ``experiments/cache/datasets/``. Subsequent calls read the cache
directly and skip the UCI fetch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.datasets import DEFAULT_CACHE, DatasetBundle, ensure_subdir

_UCI_ID = 31  # Covertype on UCI
_COORDS_FILENAME = "covertype.npy"
_LABELS_FILENAME = "covertype_labels.npy"


def load(
    embedding: str | None = None,  # noqa: ARG001 - unused for tabular data
    cache_dir: Path = DEFAULT_CACHE,
) -> DatasetBundle:
    """Load Covertype, preprocessing and caching on first call.

    Parameters
    ----------
    embedding
        Ignored. Present to match the loader contract shared with text
        and image datasets.
    cache_dir
        Root of the experiments cache. Defaults to
        ``experiments/cache/``.

    Returns
    -------
    DatasetBundle
        ``coords`` is ``(581012, 54) float32`` with every feature scaled
        to ``[0, 1]``. ``labels`` is ``(581012,) int64`` with values in
        ``{0, ..., 6}`` (shifted from the native UCI range
        ``{1, ..., 7}``).
    """
    datasets_dir = ensure_subdir("datasets", cache_dir=cache_dir)
    coords_path = datasets_dir / _COORDS_FILENAME
    labels_path = datasets_dir / _LABELS_FILENAME

    if coords_path.exists() and labels_path.exists():
        coords = np.load(coords_path)
        labels = np.load(labels_path)
    else:
        coords, labels = _fetch_and_preprocess()
        np.save(coords_path, coords)
        np.save(labels_path, labels)

    return DatasetBundle(
        coords=coords,
        labels=labels,
        metadata={
            "name": "covertype",
            "n_classes": 7,
            "feature_groups": {
                "continuous": list(range(0, 10)),
                "wilderness_area_onehot": list(range(10, 14)),
                "soil_type_onehot": list(range(14, 54)),
            },
        },
    )


def _fetch_and_preprocess() -> tuple[np.ndarray, np.ndarray]:
    """Fetch Covertype from UCI and MinMax-scale all features to [0, 1].

    Kept as a separate function so the caching path above stays trivially
    readable, and so these heavy imports are deferred until actually needed.
    """
    from sklearn.preprocessing import MinMaxScaler
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=_UCI_ID)
    # ucimlrepo returns pandas DataFrames. Covertype has a single target
    # column named "Cover_Type".
    X = dataset.data.features.to_numpy(dtype=np.float64)
    y = dataset.data.targets.to_numpy().ravel().astype(np.int64) - 1

    X_scaled = MinMaxScaler().fit_transform(X).astype(np.float32, copy=False)
    return X_scaled, y


__all__ = ["load"]


def _main() -> None:
    """Print a summary of the dataset. Useful after a fresh checkout to
    trigger the UCI fetch and to sanity-check the cached output.

    Run with::

        python -m experiments.datasets.covertype
    """
    import time

    t0 = time.perf_counter()
    bundle = load()
    elapsed = time.perf_counter() - t0

    coords, labels = bundle.coords, bundle.labels
    assert labels is not None  # Covertype is labelled

    print(f"loaded in {elapsed:.2f}s  "
          f"(cold fetch ~1-2 min; warm cache ~1-2 sec)")
    print(f"coords: shape={coords.shape}  dtype={coords.dtype}")
    print(f"labels: shape={labels.shape}  dtype={labels.dtype}  "
          f"range=[{labels.min()}, {labels.max()}]")
    print("class counts:")
    for cls, count in enumerate(np.bincount(labels)):
        print(f"    class {cls}: {count:>7,d}")
    print("per-column ranges:")
    col_min, col_max = coords.min(axis=0), coords.max(axis=0)
    print(f"    min-of-mins = {col_min.min():.4g}   "
          f"max-of-mins = {col_min.max():.4g}")
    print(f"    min-of-maxs = {col_max.min():.4g}   "
          f"max-of-maxs = {col_max.max():.4g}")
    print(f"metadata: {bundle.metadata}")


if __name__ == "__main__":
    _main()
