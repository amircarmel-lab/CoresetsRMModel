"""Image embedding generation and caching.

Exposes :func:`encode`, which takes a sequence of images (either PIL
Image objects or filesystem paths) and produces an ``(n, d) float32``
embedding array. Results are L2-normalized, matching
:mod:`experiments.embeddings.text`, so Euclidean distance on the output
is monotone with cosine distance.

Three model names are recognized:

- ``"mobilenet_v3_small"`` cheap, 576-dim (pre-classifier backbone
                           features), ~2.5M params, loaded via
                           :mod:`torchvision`.
- ``"dinov2_base"``        expensive, 768-dim (CLS token), ~86M params,
                           loaded via :mod:`transformers` from
                           HuggingFace ``facebook/dinov2-base``.
- ``"dinov2_large"``       expensive, 1024-dim (CLS token), ~300M
                           params, from ``facebook/dinov2-large``.

Everything runs on CPU. All three use the standard ImageNet
preprocessing chain (resize(256) -> center-crop(224) -> normalize with
ImageNet mean/std), so the transform is shared; only the model-load
and feature-extraction steps differ.

DINOv2 is loaded via HuggingFace rather than ``torch.hub`` because the
latter requires reaching ``github.com`` for repo metadata, which fails
on some corporate networks and airgapped environments. HuggingFace
downloads go through ``huggingface.co`` / ``cdn-lfs.huggingface.co``
and reuse the same cache our text embeddings populate.

Cache paths are supplied by the caller (typically a dataset loader),
so this module stays free of dataset-specific logic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_MODEL_NAMES = frozenset({
    "mobilenet_v3_small",
    "dinov2_base",
    "dinov2_large",
})


def encode(
    images: Sequence,
    model_name: str,
    cache_path: Path,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode ``images`` with ``model_name``, caching to ``cache_path``.

    If ``cache_path`` exists and its row count matches ``len(images)``,
    returns the cached array without touching the model. Otherwise
    loads the model, encodes on CPU with L2 normalization, writes
    ``cache_path``, and returns.

    Parameters
    ----------
    images
        Sequence of length ``n`` containing either PIL Image objects
        or filesystem paths (``str`` or ``pathlib.Path``). Type is
        detected from the first element; the whole sequence is assumed
        to be uniform.
    model_name
        One of ``"mobilenet_v3_small"``, ``"dinov2_base"``,
        ``"dinov2_large"``.
    cache_path
        Path to the ``.npy`` file. Parent directories are created as
        needed.
    batch_size
        Forward-pass batch size. ``32`` is fine for mobilenet and
        dinov2_base on CPU; drop to ``16`` for dinov2_large if memory
        is tight.
    show_progress
        If true, shows a tqdm progress bar during encoding.

    Returns
    -------
    np.ndarray
        ``(n, d) float32``, L2-normalized along axis 1.

    Raises
    ------
    ValueError
        If ``model_name`` is not in the registry, or if the cache
        exists but its row count disagrees with ``len(images)``.
    """
    if model_name not in _MODEL_NAMES:
        raise ValueError(
            f"model_name must be one of {sorted(_MODEL_NAMES)!r}; "
            f"got {model_name!r}"
        )

    n = len(images)

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] != n:
            raise ValueError(
                f"Cached embeddings at {cache_path} have {cached.shape[0]} "
                f"rows but {n} images were provided. Delete the cache and "
                f"re-run if the dataset has changed."
            )
        return cached

    # --- Cache miss. Print stage-by-stage timing so the caller can tell
    # which phase is slow on first run.
    print(f"  [encode] cache miss at {cache_path.name}", flush=True)

    t0 = time.perf_counter()
    print(f"  [encode] importing torch/torchvision...", flush=True)
    import torch
    from torchvision import transforms
    print(f"  [encode]   ... done in {time.perf_counter() - t0:.1f}s",
          flush=True)

    t0 = time.perf_counter()
    print(f"  [encode] loading model {model_name!r} "
          f"(first run downloads weights)...", flush=True)
    model = _load_model(model_name)
    model.eval()
    print(f"  [encode]   ... done in {time.perf_counter() - t0:.1f}s",
          flush=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    t0 = time.perf_counter()
    print(f"  [encode] encoding {n} images (batch_size={batch_size})...",
          flush=True)
    features = _encode_all(
        model, model_name, transform, images, batch_size, show_progress
    )
    print(f"  [encode]   ... done in {time.perf_counter() - t0:.1f}s, "
          f"shape={features.shape}", flush=True)

    # L2 normalize along the feature axis. Add a tiny epsilon to guard
    # against zero-norm rows (shouldn't happen for real images, but it's
    # cheap insurance).
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = (features / np.maximum(norms, 1e-12)).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    print(f"  [encode] wrote cache to {cache_path}", flush=True)
    return features


def _load_model(model_name: str):
    """Load the named model. Downloads weights on first use.

    - ``mobilenet_v3_small`` uses torchvision; weights come from
      ``download.pytorch.org``.
    - DINOv2 variants use :mod:`transformers` with the ``facebook/...``
      HuggingFace checkpoints, avoiding torch.hub's ``github.com``
      dependency. Downloads go to ``~/.cache/huggingface/``.
    """
    if model_name == "mobilenet_v3_small":
        from torchvision.models import (
            mobilenet_v3_small,
            MobileNet_V3_Small_Weights,
        )
        return mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )
    if model_name == "dinov2_base":
        from transformers import AutoModel

        return AutoModel.from_pretrained("facebook/dinov2-base")
    if model_name == "dinov2_large":
        from transformers import AutoModel

        return AutoModel.from_pretrained("facebook/dinov2-large")
    raise ValueError(f"unhandled model_name: {model_name!r}")


def _extract_features(model, model_name: str, batch):
    """Per-model feature extraction from a preprocessed batch tensor."""
    import torch

    with torch.no_grad():
        if model_name == "mobilenet_v3_small":
            # Pre-classifier backbone features (576-dim). The classifier
            # head (Linear 576->1024 -> Hardswish -> Dropout ->
            # Linear 1024->1000) is skipped entirely; the avgpool output
            # is the general-purpose embedding.
            x = model.features(batch)
            x = model.avgpool(x)
            return torch.flatten(x, 1)
        # DINOv2 via transformers: the HuggingFace Dinov2Model returns a
        # BaseModelOutputWithPooling whose pooler_output is the CLS
        # token after the final layernorm. This matches what torch.hub's
        # DINOv2 returns from model(x).
        return model(pixel_values=batch).pooler_output


def _encode_all(
    model,
    model_name: str,
    transform,
    images: Sequence,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    """Iterate over images in batches, accumulate features on CPU."""
    import torch

    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(total=len(images), desc=f"encoding {model_name}",
                        unit="img")
        except ImportError:
            pbar = None

    all_features: list[np.ndarray] = []
    batch_tensors: list = []

    def _flush(batch_list):
        if not batch_list:
            return
        stacked = torch.stack(batch_list)
        out = _extract_features(model, model_name, stacked).cpu().numpy()
        all_features.append(out)
        if pbar is not None:
            pbar.update(len(batch_list))

    for img in _iter_rgb(images):
        batch_tensors.append(transform(img))
        if len(batch_tensors) == batch_size:
            _flush(batch_tensors)
            batch_tensors = []
    _flush(batch_tensors)

    if pbar is not None:
        pbar.close()

    if not all_features:
        # Empty input. Return an empty 2-D array with the right dtype.
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(all_features, axis=0)


def _iter_rgb(images: Sequence):
    """Yield RGB PIL images from a sequence of PIL images or paths.

    The input type is detected from ``images[0]``; the whole sequence
    must be uniform. PIL images not already in ``RGB`` mode are
    converted; paths are opened and converted.
    """
    if len(images) == 0:
        return

    first = images[0]
    looks_like_pil = (
        hasattr(first, "mode")
        and hasattr(first, "convert")
        and hasattr(first, "size")
    )

    if looks_like_pil:
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            yield img
    else:
        from PIL import Image

        for path in images:
            yield Image.open(str(path)).convert("RGB")


__all__ = ["encode"]


def _main() -> None:
    """Smoke test: encode three random images with each supported model.

    Run with::

        python -m experiments.embeddings.image

    First run downloads weights (~20 MB for mobilenet_v3_small from
    pytorch.org, ~340 MB for dinov2_base and ~1.2 GB for dinov2_large
    from HuggingFace). Subsequent runs hit the local caches and finish
    in seconds.

    Uses random 96x96 PIL images (the native STL-10 resolution) and
    verifies shape, dtype, and L2 normalization for each model.
    """
    import tempfile

    from PIL import Image

    rng = np.random.default_rng(0)
    arrays = rng.integers(0, 256, size=(3, 96, 96, 3), dtype=np.uint8)
    pil_images = [Image.fromarray(a) for a in arrays]

    models = [
        ("mobilenet_v3_small", "~20 MB"),
        ("dinov2_base", "~340 MB"),
        ("dinov2_large", "~1.2 GB"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        total_start = time.perf_counter()
        for model_name, weights_size in models:
            cache_path = tmp_path / f"{model_name}.npy"
            print(f"\n--- {model_name}  (weights {weights_size}) ---")

            t0 = time.perf_counter()
            emb = encode(
                pil_images,
                model_name=model_name,
                cache_path=cache_path,
                show_progress=False,
            )
            elapsed = time.perf_counter() - t0
            print(f"took {elapsed:.1f}s   shape={emb.shape}   "
                  f"dtype={emb.dtype}")

            norms = np.linalg.norm(emb, axis=1)
            print(f"row norms (should be ~1.0): {norms.round(4)}")

            assert 0.9999 < norms.min() <= norms.max() < 1.0001, \
                f"{model_name}: L2 normalization broken, norms={norms}"
        total_elapsed = time.perf_counter() - total_start

    print(f"\nsmoke test OK for all three models  "
          f"(total wall-clock: {total_elapsed:.1f}s)")


if __name__ == "__main__":
    _main()