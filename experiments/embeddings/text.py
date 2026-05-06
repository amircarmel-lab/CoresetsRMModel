"""BGE text-embedding generation and caching.

Exposes :func:`encode`, which takes a sequence of strings and produces
an ``(n, d) float32`` embedding array. Results are L2-normalized, so
Euclidean distance on the output is monotone with cosine distance (and
equal up to a constant factor). This is the canonical way to use BGE.

Any HuggingFace sentence-transformers model ID works; the two the
experiments currently use are:

- ``TaylorAI/bge-micro-v2``     cheap, 384-dim, ~17M params
- ``BAAI/bge-large-en-v1.5``    expensive, 1024-dim, ~335M params

Cache paths are supplied by the caller (typically a dataset loader),
so this module stays free of dataset-specific logic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np


def encode(
    texts: Sequence[str],
    model_name: str,
    cache_path: Path,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode ``texts`` with ``model_name``, caching to ``cache_path``.

    If ``cache_path`` exists and the cached array's row count matches
    ``len(texts)``, returns the cached array without touching the model.
    Otherwise loads the model, encodes on CPU with L2 normalization,
    writes ``cache_path``, and returns.

    On cache miss, prints one line per stage (import, model load,
    encode, cache write) with elapsed time. Useful for long runs like
    20-Newsgroups with bge-large, where each stage is visibly slow and
    it's otherwise unclear which one is taking the time.

    Parameters
    ----------
    texts
        Sequence of strings to encode. Length ``n``.
    model_name
        Any HuggingFace sentence-transformers model ID.
    cache_path
        Path to the ``.npy`` file. Parent directories are created as
        needed.
    batch_size
        Encoder batch size. ``32`` works for bge-micro on CPU; drop to
        ``16`` for bge-large on CPU if memory is tight.
    show_progress
        Passed through to sentence-transformers' progress bar.

    Returns
    -------
    np.ndarray
        ``(n, d) float32``, L2-normalized along axis 1.

    Raises
    ------
    ValueError
        If ``cache_path`` exists but its row count disagrees with
        ``len(texts)``. This signals a stale cache; delete the file
        and re-run.
    """
    n = len(texts)

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] != n:
            raise ValueError(
                f"Cached embeddings at {cache_path} have {cached.shape[0]} "
                f"rows but {n} texts were provided. Delete the cache and "
                f"re-run if the dataset has changed."
            )
        return cached

    # --- Cache miss. Print stage-by-stage timing so the caller can tell
    # which phase is slow on first run (library import is quick; model
    # download on first call can be many minutes over an unauthenticated
    # HuggingFace connection).
    print(f"  [encode] cache miss at {cache_path.name}", flush=True)

    t0 = time.perf_counter()
    print(f"  [encode] importing sentence-transformers (pulls torch)...",
          flush=True)
    # Deferred: sentence-transformers pulls torch, which is ~2s to import.
    from sentence_transformers import SentenceTransformer
    print(f"  [encode]   ... done in {time.perf_counter() - t0:.1f}s",
          flush=True)

    t0 = time.perf_counter()
    print(f"  [encode] loading model {model_name!r} "
          f"(first run downloads weights, can take many minutes)...",
          flush=True)
    model = SentenceTransformer(model_name, device="cpu")
    print(f"  [encode]   ... done in {time.perf_counter() - t0:.1f}s",
          flush=True)

    t0 = time.perf_counter()
    print(f"  [encode] encoding {n} texts (batch_size={batch_size})...",
          flush=True)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"  [encode]   ... done in {time.perf_counter() - t0:.1f}s, "
          f"shape={embeddings.shape}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  [encode] wrote cache to {cache_path}", flush=True)
    return embeddings


def slugify_model_name(model_name: str) -> str:
    """Turn a HuggingFace model ID into a filesystem-safe filename stem.

    Keeps the organization prefix so that e.g. ``BAAI/bge-large-en-v1.5``
    and (a hypothetical) ``OtherOrg/bge-large-en-v1.5`` don't collide.

    Example
    -------
    >>> slugify_model_name("BAAI/bge-large-en-v1.5")
    'BAAI__bge-large-en-v1.5'
    """
    return model_name.replace("/", "__")


__all__ = ["encode", "slugify_model_name"]


def _main() -> None:
    """Smoke test: encode three short sentences with both BGE models.

    Run with::

        python -m experiments.embeddings.text

    First run downloads weights (~60 MB for bge-micro-v2, ~1.3 GB for
    bge-large-en-v1.5) and takes a few minutes total on a typical
    connection. Subsequent runs hit the HuggingFace cache and finish
    in seconds.

    For each model, verifies that L2 normalization produces unit-norm
    rows and that a semantically-similar sentence pair is closer than
    a semantically-distant pair.
    """
    import tempfile

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above the sluggish hound.",
        "My grandmother's recipe calls for two cups of flour.",
    ]

    models = [
        ("TaylorAI/bge-micro-v2", "~60 MB"),
        ("BAAI/bge-large-en-v1.5", "~1.3 GB"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        total_start = time.perf_counter()
        for model_name, weights_size in models:
            cache_path = tmp_path / (slugify_model_name(model_name) + ".npy")
            print(f"\n--- {model_name}  (weights {weights_size}) ---")

            t0 = time.perf_counter()
            emb = encode(
                sentences,
                model_name=model_name,
                cache_path=cache_path,
                show_progress=False,
            )
            elapsed = time.perf_counter() - t0
            print(f"took {elapsed:.1f}s   shape={emb.shape}   "
                  f"dtype={emb.dtype}")

            norms = np.linalg.norm(emb, axis=1)
            print(f"row norms (should be ~1.0): {norms.round(4)}")

            cos_01 = float(emb[0] @ emb[1])
            cos_02 = float(emb[0] @ emb[2])
            print(f"cos(fox vs similar-fox)    = {cos_01:.4f}")
            print(f"cos(fox vs grandmother's)  = {cos_02:.4f}")

            assert 0.9999 < norms.min() <= norms.max() < 1.0001, \
                f"{model_name}: L2 normalization broken, norms={norms}"
            assert cos_01 > cos_02, \
                f"{model_name}: similar sentences should be closer"
        total_elapsed = time.perf_counter() - total_start

    print(f"\nsmoke test OK for both models  "
          f"(total wall-clock: {total_elapsed:.1f}s)")


if __name__ == "__main__":
    _main()