"""20 Newsgroups (~18,846 documents, 20 classes) loader.

Source: :func:`sklearn.datasets.fetch_20newsgroups`, with headers,
footers, and quoted replies stripped so the embedding model sees
content rather than boilerplate.

The loader generates two sets of embeddings on first call, using
:mod:`experiments.embeddings.text`:

- cheap:     ``TaylorAI/bge-micro-v2``     (384-dim)
- expensive: ``BAAI/bge-large-en-v1.5``    (1024-dim)

Both are L2-normalized and cached. On warm calls, sklearn is not
touched at all.

The returned bundle has ``coords`` set to the embedding named by the
``embedding`` argument (default ``"expensive"``). Both embeddings are
always accessible via ``metadata["cheap_coords"]`` and
``metadata["expensive_coords"]``, so the dual-embedding quadruplet
oracle can find what it needs regardless of which one is in ``coords``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.datasets import DEFAULT_CACHE, DatasetBundle, ensure_subdir
from experiments.embeddings.text import slugify_model_name, encode


CHEAP_MODEL = "TaylorAI/bge-micro-v2"
EXPENSIVE_MODEL = "BAAI/bge-large-en-v1.5"

_EMBEDDING_CHOICES = {"cheap": CHEAP_MODEL, "expensive": EXPENSIVE_MODEL}

_LABELS_FILENAME = "labels.npy"
_CLASS_NAMES_FILENAME = "class_names.json"


def load(
    embedding: str | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> DatasetBundle:
    """Load 20-Newsgroups with BGE embeddings, caching on first call.

    Parameters
    ----------
    embedding
        Which embedding to place in ``coords``. One of ``"cheap"`` or
        ``"expensive"``; defaults to ``"expensive"``. Both embeddings
        are always generated and cached regardless of this choice; the
        parameter only selects which one populates ``coords``.
    cache_dir
        Root of the experiments cache. Defaults to
        ``experiments/cache/``.

    Returns
    -------
    DatasetBundle
        ``coords`` is ``(n, d) float32`` L2-normalized embeddings.
        ``labels`` is ``(n,) int64`` with values in ``{0, ..., 19}``.
        ``metadata`` carries ``cheap_coords`` and ``expensive_coords``
        (both as ``(n, d) float32``), ``class_names`` (list of 20
        strings), and dataset identifiers.

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

    emb_dir = ensure_subdir(
        "embeddings", "twentynewsgroups", cache_dir=cache_dir
    )
    cheap_path = emb_dir / (slugify_model_name(CHEAP_MODEL) + ".npy")
    expensive_path = emb_dir / (slugify_model_name(EXPENSIVE_MODEL) + ".npy")
    labels_path = emb_dir / _LABELS_FILENAME
    class_names_path = emb_dir / _CLASS_NAMES_FILENAME

    all_cached = (
        cheap_path.exists()
        and expensive_path.exists()
        and labels_path.exists()
        and class_names_path.exists()
    )

    if all_cached:
        # Pure cache path: no sklearn, no model.
        labels = np.load(labels_path)
        class_names = json.loads(class_names_path.read_text())
        cheap_coords = np.load(cheap_path)
        expensive_coords = np.load(expensive_path)
        _validate_shapes(cheap_coords, expensive_coords, labels)
    else:
        # At least one cache is missing. Fetch texts once and let
        # encode() fill in whatever's stale. encode() is itself
        # cache-aware, so if (say) cheap is cached and expensive isn't,
        # we don't re-run cheap.
        texts, labels, class_names = _fetch_texts()
        np.save(labels_path, labels)
        class_names_path.write_text(json.dumps(class_names, indent=2))

        cheap_coords = encode(texts, CHEAP_MODEL, cheap_path)
        expensive_coords = encode(texts, EXPENSIVE_MODEL, expensive_path)
        _validate_shapes(cheap_coords, expensive_coords, labels)

    coords = cheap_coords if embedding == "cheap" else expensive_coords

    return DatasetBundle(
        coords=coords,
        labels=labels,
        metadata={
            "name": "twentynewsgroups",
            "n_classes": 20,
            "class_names": class_names,
            "cheap_model": CHEAP_MODEL,
            "expensive_model": EXPENSIVE_MODEL,
            "cheap_coords": cheap_coords,
            "expensive_coords": expensive_coords,
        },
    )


def _fetch_texts() -> tuple[list[str], np.ndarray, list[str]]:
    """Fetch 20-Newsgroups and return (texts, labels, class_names).

    Uses ``subset="all"`` to combine train+test (~18,846 documents) and
    strips headers, footers, and quoted replies so the embedder sees
    the actual message body.
    """
    from sklearn.datasets import fetch_20newsgroups

    bundle = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    texts = list(bundle.data)
    labels = bundle.target.astype(np.int64, copy=False)
    class_names = list(bundle.target_names)
    return texts, labels, class_names


def _validate_shapes(
    cheap: np.ndarray, expensive: np.ndarray, labels: np.ndarray
) -> None:
    """Ensure the three cached artefacts agree on ``n``.

    A mismatch means caches were regenerated against different fetches
    (likely because the user deleted only some of them). The cleanest
    recovery is to delete all and re-run, so we error loudly with that
    instruction.
    """
    n_cheap, n_exp, n_lab = cheap.shape[0], expensive.shape[0], labels.shape[0]
    if not (n_cheap == n_exp == n_lab):
        raise ValueError(
            f"Cache mismatch for 20-Newsgroups: cheap has {n_cheap} rows, "
            f"expensive has {n_exp} rows, labels has {n_lab} rows. Delete "
            f"all files under cache/embeddings/twentynewsgroups/ and "
            f"re-run."
        )


__all__ = ["load", "CHEAP_MODEL", "EXPENSIVE_MODEL"]


def _main() -> None:
    """Trigger cold generation (if needed) and print a summary.

    Run with::

        python -m experiments.datasets.twentynewsgroups

    First run on a fresh checkout:

    - Downloads ~14 MB of raw 20-Newsgroups via sklearn (~30s-1min).
    - Downloads ~60 MB bge-micro weights from HuggingFace (one-time).
    - Encodes 18,846 documents with bge-micro (~1-2 min on CPU).
    - Downloads ~1.3 GB bge-large weights from HuggingFace (one-time).
    - Encodes 18,846 documents with bge-large (~20-30 min on CPU).

    Subsequent runs load from cache in a couple of seconds.
    """
    import time

    print("Loading 20-Newsgroups "
          "(first run encodes ~18,846 docs; expect 20-30+ minutes)...",
          flush=True)
    t0 = time.perf_counter()
    bundle = load()
    elapsed = time.perf_counter() - t0

    coords = bundle.coords
    labels = bundle.labels
    assert labels is not None

    cheap = bundle.metadata["cheap_coords"]
    expensive = bundle.metadata["expensive_coords"]

    print(f"\nloaded in {elapsed:.2f}s")
    print(f"coords (expensive by default):  shape={coords.shape}  "
          f"dtype={coords.dtype}")
    print(f"  cheap_coords:      shape={cheap.shape}  dtype={cheap.dtype}")
    print(f"  expensive_coords:  shape={expensive.shape}  "
          f"dtype={expensive.dtype}")
    print(f"labels:  shape={labels.shape}  dtype={labels.dtype}  "
          f"range=[{labels.min()}, {labels.max()}]")
    print(f"class counts (top 3 and bottom 3):")
    counts = np.bincount(labels, minlength=20)
    class_names = bundle.metadata["class_names"]
    order = np.argsort(counts)[::-1]
    for i in list(order[:3]) + [-1] + list(order[-3:]):
        if i == -1:
            print("    ...")
            continue
        print(f"    class {i:2d}  {counts[i]:>5d}   {class_names[i]}")

    # Quick sanity: embeddings should be L2-normalized
    cheap_norms = np.linalg.norm(cheap, axis=1)
    exp_norms = np.linalg.norm(expensive, axis=1)
    print(f"\nnorms (should be ~1.0):")
    print(f"    cheap:      min={cheap_norms.min():.4f}  "
          f"max={cheap_norms.max():.4f}")
    print(f"    expensive:  min={exp_norms.min():.4f}  "
          f"max={exp_norms.max():.4f}")


if __name__ == "__main__":
    _main()
