"""US Census 1990 (~2,458,285 rows, ~68 features) loader.

Source: UCI Machine Learning Repository via :mod:`ucimlrepo`
(``fetch_ucirepo(id=116)``). The raw file stores one person per row with
pre-coded integer attributes (age bucket, industry code, marital status
code, and so on). There is no natural label column, so this loader
returns ``labels=None``.

The raw dataset has ~2.46M rows. We **cap it at 1,000,000 rows** via a
uniform random subsample (seed fixed at module level so every machine
produces a bit-identical cache). The cap keeps memory and per-pipeline
runtime reasonable without changing the character of the data: Census
is sorted roughly by state, so a random subsample is far more
representative than the first 1M rows.

Note on the "distance" this induces. The categorical codes are not
ordinally meaningful (``marital_status=3`` is not "three times" the
marital-ness of ``marital_status=1``), but they are treated as numeric
here, matching standard practice in the clustering literature on this
dataset. The clustering effectively measures "which bucket does each
person fall in" rather than genuine demographic similarity.

Preprocessing:

1. Drop the row-id column (``caseid`` or equivalent).
2. Drop any zero-variance column (a column whose values never change
   contributes nothing to clustering but can cause numerical issues in
   MinMax).
3. Drop any unique-per-row column (a secondary ID, if one slipped
   through). These are identified by ``nunique() == n_rows``.
4. Randomly subsample to ``N_CAP`` rows with ``SUBSAMPLE_SEED``.
5. Scale remaining columns to ``[0, 1]`` with
   :class:`sklearn.preprocessing.MinMaxScaler`. This matches Covertype
   preprocessing and prevents columns with large code ranges from
   dominating Euclidean distance. The MinMax range is defined by the
   subsample, not the full population - the cap IS the dataset, as far
   as the experiments are concerned.

The preprocessed matrix is cached under
``experiments/cache/datasets/`` with ``N_CAP`` and ``SUBSAMPLE_SEED``
encoded in the filename, so changing either constant automatically
invalidates the old cache (no manual deletion needed). Subsequent
calls with unchanged constants read the cache directly and skip the
UCI fetch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.datasets import DEFAULT_CACHE, DatasetBundle, ensure_subdir

_UCI_ID = 116  # US Census Data (1990) on UCI

# Subsample cap. The raw dataset has ~2.46M rows; capping at 1M keeps
# memory and runtime reasonable while letting `--exp 6` runs of
# experiment 4 use the full available size (n_max = 15625 * 2^6 = 1M).
# The seed is module-level (not a caller parameter) so every machine
# produces bit-identical cached output.
N_CAP = 2_000_000
SUBSAMPLE_SEED = 0

# Cache filenames embed N_CAP and SUBSAMPLE_SEED so that bumping either
# constant produces a different cache path, automatically invalidating
# stale caches without anyone having to remember to delete the .npy.
_COORDS_FILENAME = f"census_n{N_CAP}_seed{SUBSAMPLE_SEED}.npy"
_DROPPED_COLUMNS_FILENAME = (
    f"census_n{N_CAP}_seed{SUBSAMPLE_SEED}_dropped_columns.json"
)


def load(
    embedding: str | None = None,  # noqa: ARG001 - unused for tabular data
    cache_dir: Path = DEFAULT_CACHE,
) -> DatasetBundle:
    """Load Census 1990, preprocessing and caching on first call.

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
        ``coords`` is ``(N_CAP, ~68) float32`` with every feature
        scaled to ``[0, 1]`` (column count depends on how many are
        dropped as ID-like or zero-variance; the dropped column names
        are recorded in ``metadata["dropped_columns"]``). ``labels`` is
        ``None``.
    """
    import json

    datasets_dir = ensure_subdir("datasets", cache_dir=cache_dir)
    coords_path = datasets_dir / _COORDS_FILENAME
    dropped_path = datasets_dir / _DROPPED_COLUMNS_FILENAME

    if coords_path.exists() and dropped_path.exists():
        coords = np.load(coords_path)
        dropped_columns = json.loads(dropped_path.read_text())
    else:
        coords, dropped_columns = _fetch_and_preprocess()
        np.save(coords_path, coords)
        dropped_path.write_text(json.dumps(dropped_columns, indent=2))

    return DatasetBundle(
        coords=coords,
        labels=None,
        metadata={
            "name": "census1990",
            "dropped_columns": dropped_columns,
            "n_cap": N_CAP,
            "subsample_seed": SUBSAMPLE_SEED,
        },
    )


def _fetch_and_preprocess() -> tuple[np.ndarray, dict]:
    """Fetch Census, drop non-feature columns, subsample to N_CAP,
    MinMax-scale to [0, 1].

    Returns the scaled ``(n, d) float32`` array and a dict describing
    which columns were dropped and why.
    """
    from sklearn.preprocessing import MinMaxScaler
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=_UCI_ID)
    # ucimlrepo returns a pandas DataFrame under .data.features. Census
    # has no target column, so .data.targets may be None or empty.
    df = dataset.data.features

    df_clean, dropped = _drop_non_feature_columns(df)

    # Subsample rows. If the source already has <= N_CAP rows, keep all
    # of them.
    if len(df_clean) > N_CAP:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        idx = rng.choice(len(df_clean), size=N_CAP, replace=False)
        idx.sort()  # sort so the cached order is deterministic and
                    # contiguous reads stay cache-friendly
        df_clean = df_clean.iloc[idx]

    X = df_clean.to_numpy(dtype=np.float64)
    X_scaled = MinMaxScaler().fit_transform(X).astype(np.float32, copy=False)
    return X_scaled, dropped


def _drop_non_feature_columns(df):
    """Drop caseid/ID-like, zero-variance, and unique-per-row columns.

    Pure function over a DataFrame: takes a DataFrame, returns
    ``(cleaned_df, dropped_info)`` where ``dropped_info`` is a dict
    mapping reason -> list of column names. Kept small and pure so it
    can be tested on synthetic DataFrames without any network fetch.
    """
    n_rows = len(df)
    dropped: dict[str, list[str]] = {
        "id_like_by_name": [],
        "zero_variance": [],
        "unique_per_row": [],
    }

    # 1. Drop by name. Census uses "caseid" but be liberal: anything
    #    that looks like an identifier column.
    id_name_patterns = ("caseid", "case_id", "id", "rownum", "index")
    for col in df.columns:
        name_lower = str(col).lower()
        if name_lower in id_name_patterns:
            dropped["id_like_by_name"].append(str(col))
    df = df.drop(columns=dropped["id_like_by_name"])

    # 2. Drop zero-variance columns. Use nunique() rather than std() to
    #    avoid dtype issues and to catch columns where every value is
    #    some constant non-zero integer code.
    nunique = df.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    dropped["zero_variance"] = [str(c) for c in zero_var]
    df = df.drop(columns=zero_var)

    # 3. Drop unique-per-row columns (a secondary ID that slipped
    #    through the name filter). On small synthetic samples any
    #    column can look unique-per-row by chance, so only flag this
    #    on reasonably-sized inputs.
    if n_rows >= 1000:
        nunique = df.nunique()
        unique_per_row = nunique[nunique == n_rows].index.tolist()
        dropped["unique_per_row"] = [str(c) for c in unique_per_row]
        df = df.drop(columns=unique_per_row)

    return df, dropped


__all__ = ["load"]


def _main() -> None:
    """Print a summary of the dataset. Run with::

        python -m experiments.datasets.census
    """
    import time

    t0 = time.perf_counter()
    bundle = load()
    elapsed = time.perf_counter() - t0

    coords = bundle.coords
    print(f"loaded in {elapsed:.2f}s  "
          f"(cold fetch several minutes; warm cache a few sec)")
    print(f"coords: shape={coords.shape}  dtype={coords.dtype}  "
          f"size={coords.nbytes / 1e6:.1f} MB")
    print(f"labels: {bundle.labels}")
    print(f"n_cap (module-level): {N_CAP}")
    print(f"subsample_seed (module-level): {SUBSAMPLE_SEED}")
    print("per-column ranges:")
    col_min, col_max = coords.min(axis=0), coords.max(axis=0)
    print(f"    min-of-mins = {col_min.min():.4g}   "
          f"max-of-mins = {col_min.max():.4g}")
    print(f"    min-of-maxs = {col_max.min():.4g}   "
          f"max-of-maxs = {col_max.max():.4g}")
    dropped = bundle.metadata["dropped_columns"]
    print("dropped columns:")
    for reason, cols in dropped.items():
        print(f"    {reason}: {cols if cols else '(none)'}")


if __name__ == "__main__":
    _main()