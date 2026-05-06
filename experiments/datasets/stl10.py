"""STL-10 train+test split (13,000 labelled images, 96x96 RGB) loader.

Source: :class:`torchvision.datasets.STL10`. We concatenate the
``"train"`` split (5,000 images) with the ``"test"`` split (8,000
images) to get 13,000 labelled samples across 10 classes. The
``"unlabeled"`` split (100,000 images without labels) is not used —
13,000 is sufficient for the clustering experiments and encodes
roughly 8x faster.

The raw archive (~2.6 GB, contains all three splits) downloads to
``experiments/cache/datasets/stl10_raw/`` on first use. Extracting the
archive produces individual binary files per split; torchvision then
reads them on demand.

The loader generates two sets of embeddings via
:mod:`experiments.embeddings.image`:

- cheap:     ``mobilenet_v3_small``  (576-dim)
- expensive: ``dinov2_large``        (1024-dim)

Both are L2-normalized and cached. DINOv2-base is intentionally
excluded — the plan calls ``dinov2_large`` the primary expensive
embedding; adding base costs more encode time without enabling a new
experiment.

First-run encoding on a modern CPU: mobilenet ~3 min, dinov2_large
~3 hours. Subsequent loads hit cache in a couple of seconds.

The returned bundle has ``coords`` set to the embedding named by the
``embedding`` argument (default ``"expensive"``). Both embeddings are
always accessible via ``metadata["cheap_coords"]`` and
``metadata["expensive_coords"]``, so the dual-embedding oracle can
read both regardless of the choice. Labels are integer class indices
in ``{0, ..., 9}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from documentation import DEFAULT_CACHE, DatasetBundle, ensure_subdir
from documentation import encode

CHEAP_MODEL = "mobilenet_v3_small"
EXPENSIVE_MODEL = "dinov2_large"

_EMBEDDING_CHOICES = {"cheap": CHEAP_MODEL, "expensive": EXPENSIVE_MODEL}

_LABELS_FILENAME = "labels.npy"
_CLASS_NAMES_FILENAME = "class_names.json"


def load(
    embedding: str | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> DatasetBundle:
    """Load STL-10 train+test with image embeddings, caching on first call.

    Parameters
    ----------
    embedding
        Which embedding to place in ``coords``. One of ``"cheap"`` or
        ``"expensive"``; defaults to ``"expensive"``. Both embeddings
        are always generated and cached regardless; the parameter only
        selects which one populates ``coords``.
    cache_dir
        Root of the experiments cache. Defaults to
        ``experiments/cache/``.

    Returns
    -------
    DatasetBundle
        ``coords`` is ``(13000, d) float32`` L2-normalized embeddings.
        ``labels`` is ``(13000,) int64`` with values in ``{0, ..., 9}``.
        ``metadata`` carries ``cheap_coords`` and ``expensive_coords``,
        model identifiers, ``class_names`` (list of 10 strings), and
        ``n_classes`` (10).

    Raises
    ------
    ValueError
        If ``embedding`` is not ``"cheap"`` or ``"expensive"``, or if
        partial cache files exist with inconsistent row counts (in
        which case the user should delete the cache and re-run).
    """
    if embedding is None:
        embedding = "expensive"
    if embedding not in _EMBEDDING_CHOICES:
        raise ValueError(
            f"embedding must be one of {sorted(_EMBEDDING_CHOICES)!r}; "
            f"got {embedding!r}"
        )

    emb_dir = ensure_subdir("embeddings", "stl10", cache_dir=cache_dir)
    cheap_path = emb_dir / f"{CHEAP_MODEL}.npy"
    expensive_path = emb_dir / f"{EXPENSIVE_MODEL}.npy"
    labels_path = emb_dir / _LABELS_FILENAME
    class_names_path = emb_dir / _CLASS_NAMES_FILENAME

    all_cached = (
        cheap_path.exists()
        and expensive_path.exists()
        and labels_path.exists()
        and class_names_path.exists()
    )

    if all_cached:
        labels = np.load(labels_path)
        class_names = json.loads(class_names_path.read_text())
        cheap_coords = np.load(cheap_path)
        expensive_coords = np.load(expensive_path)
        _validate_shapes(cheap_coords, expensive_coords, labels)
    else:
        # At least one cache missing. Download/load STL-10 raw (fast
        # when torchvision's binary cache exists) and let encode()
        # re-run only the models whose caches are actually missing.
        pil_sequence, labels, class_names = _fetch_stl10(cache_dir)
        np.save(labels_path, labels)
        class_names_path.write_text(json.dumps(class_names, indent=2))

        cheap_coords = encode(pil_sequence, CHEAP_MODEL, cheap_path)
        expensive_coords = encode(
            pil_sequence, EXPENSIVE_MODEL, expensive_path
        )
        _validate_shapes(cheap_coords, expensive_coords, labels)

    coords = cheap_coords if embedding == "cheap" else expensive_coords

    return DatasetBundle(
        coords=coords,
        labels=labels,
        metadata={
            "name": "stl10_train_test",
            "n_classes": 10,
            "class_names": class_names,
            "cheap_model": CHEAP_MODEL,
            "expensive_model": EXPENSIVE_MODEL,
            "cheap_coords": cheap_coords,
            "expensive_coords": expensive_coords,
        },
    )


def _fetch_stl10(cache_dir: Path):
    """Download (if needed) and concatenate STL-10 train + test splits.

    Returns ``(pil_sequence, labels, class_names)`` where
    ``pil_sequence`` is a ``Sequence`` of PIL images over 13,000
    samples, ``labels`` is a ``(13000,) int64`` array with values in
    ``{0, ..., 9}``, and ``class_names`` is the list of 10 class names
    supplied by torchvision.

    Note that on first call torchvision downloads the full ~2.6 GB
    archive (contains all three splits) even though we only use two.
    """
    from torchvision.datasets import STL10

    raw_dir = ensure_subdir("datasets", "stl10_raw", cache_dir=cache_dir)
    train = STL10(root=str(raw_dir), split="train", download=True)
    test = STL10(root=str(raw_dir), split="test", download=True)

    data = np.concatenate([train.data, test.data], axis=0)
    labels = np.concatenate([train.labels, test.labels]).astype(
        np.int64, copy=False
    )
    class_names = list(train.classes)  # same list for train and test

    return _STL10PILImages(data), labels, class_names


class _STL10PILImages:
    """Sequence adapter over a stacked STL-10 image array.

    torchvision stores STL-10 image data as ``(n, 3, 96, 96) uint8``
    CHW. This adapter keeps the big array as-is and produces PIL
    images on ``__getitem__`` so only one PIL image is alive at a time.
    The ``encode()`` function pulls images sequentially during batching,
    which keeps peak memory low.
    """

    def __init__(self, data: np.ndarray) -> None:
        if data.ndim != 4:
            raise RuntimeError(
                f"unexpected STL-10 data shape: {data.shape}"
            )
        # torchvision's STL10 stores (n, 3, 96, 96) CHW. Tolerate HWC in
        # case a future torchvision version changes the layout.
        if data.shape[1] == 3 and data.shape[-1] != 3:
            self._data = data  # CHW
            self._is_chw = True
        elif data.shape[-1] == 3:
            self._data = data  # HWC
            self._is_chw = False
        else:
            raise RuntimeError(
                f"can't interpret STL-10 data layout: {data.shape}"
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int):
        from PIL import Image

        arr = self._data[i]
        if self._is_chw:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            # Ensure contiguous: PIL copies regardless, but this keeps
            # us robust to future PIL versions that might assert.
            arr = np.ascontiguousarray(arr)
        return Image.fromarray(arr)


def _validate_shapes(
    cheap: np.ndarray, expensive: np.ndarray, labels: np.ndarray
) -> None:
    """Ensure the three cached artefacts agree on ``n``."""
    n_cheap, n_exp, n_lab = cheap.shape[0], expensive.shape[0], labels.shape[0]
    if not (n_cheap == n_exp == n_lab):
        raise ValueError(
            f"Cache mismatch for STL-10: cheap has {n_cheap} rows, "
            f"expensive has {n_exp} rows, labels has {n_lab} rows. "
            f"Delete all files under cache/embeddings/stl10/ and re-run."
        )


__all__ = ["load", "CHEAP_MODEL", "EXPENSIVE_MODEL"]


def _main() -> None:
    """Trigger cold generation (if needed) and print a summary.

    Run with::

        python -m experiments.datasets.stl10

    First run:

    - Downloads ~2.6 GB of raw STL-10 via torchvision (~5-15 min).
    - Encodes 13,000 images with mobilenet_v3_small (~3 min on CPU).
    - Encodes 13,000 images with dinov2_large (~3 hours on CPU).

    Subsequent runs load from cache in a couple of seconds.
    """
    import time

    print("Loading STL-10 train+test "
          "(first run encodes 13,000 images; expect ~3 hours total "
          "for dinov2_large)...",
          flush=True)
    t0 = time.perf_counter()
    bundle = load()
    elapsed = time.perf_counter() - t0

    coords = bundle.coords
    labels = bundle.labels
    assert labels is not None
    cheap = bundle.metadata["cheap_coords"]
    expensive = bundle.metadata["expensive_coords"]
    class_names = bundle.metadata["class_names"]

    print(f"\nloaded in {elapsed:.2f}s")
    print(f"coords (expensive by default):  shape={coords.shape}  "
          f"dtype={coords.dtype}")
    print(f"  cheap_coords:      shape={cheap.shape}  dtype={cheap.dtype}")
    print(f"  expensive_coords:  shape={expensive.shape}  "
          f"dtype={expensive.dtype}")
    print(f"labels:  shape={labels.shape}  dtype={labels.dtype}  "
          f"range=[{labels.min()}, {labels.max()}]")

    counts = np.bincount(labels, minlength=bundle.metadata["n_classes"])
    print(f"class distribution (expect train 500 + test 800 = 1300 per class):")
    for cls, count in enumerate(counts):
        print(f"    class {cls}  {count:>5d}   {class_names[cls]}")

    cheap_norms = np.linalg.norm(cheap, axis=1)
    exp_norms = np.linalg.norm(expensive, axis=1)
    print(f"\nnorms (should be ~1.0):")
    print(f"    cheap:      min={cheap_norms.min():.4f}  "
          f"max={cheap_norms.max():.4f}")
    print(f"    expensive:  min={exp_norms.min():.4f}  "
          f"max={exp_norms.max():.4f}")


if __name__ == "__main__":
    _main()