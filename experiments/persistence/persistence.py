"""Implementations for result persistence.

Storage format: plain CSV, one row per experiment run, append-only
across sessions. Readable in any text editor; trivial to diff and grep.

Schema is enforced per-file: the first call to `append_row` fixes the
header, and subsequent calls must supply the same keys in the same
order. Mismatches raise rather than silently padding with nulls —
silent coercion has caused us to lose data in the past.

Not safe for concurrent writers to the same file. If an experiment
parallelizes, each worker writes to its own file and plotting code
concatenates.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def append_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one row to a CSV at `path`.

    If the file does not exist, it is created (along with any missing
    parent directories) and the header is written from `row.keys()`.
    If the file exists, the first line is read and compared to
    `row.keys()`; a `ValueError` is raised on mismatch.

    Values are written via `csv.DictWriter`; anything that str() can
    handle round-trips through `load_results` as a string, number, or
    NaN depending on what pandas infers.

    Parameters
    ----------
    path : destination CSV path.
    row : ordered mapping of column name -> value. Insertion order
          matters — it becomes the column order in the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())
    if not fieldnames:
        raise ValueError("row is empty; nothing to append")

    existing_header = _read_header(path)

    if existing_header is None:
        write_header = True
    else:
        if existing_header != fieldnames:
            raise ValueError(
                f"Schema mismatch at {path}.\n"
                f"  Existing header: {existing_header}\n"
                f"  New row keys:    {fieldnames}\n"
                "Either align the schema or write to a different file."
            )
        write_header = False

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def load_results(path: str | Path) -> pd.DataFrame:
    """Read the CSV at `path` into a DataFrame.

    Types are inferred by pandas' default CSV reader. Numeric columns
    come back as int/float, string columns as object, empty cells as NaN.

    Raises `FileNotFoundError` if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no results file at {path}")
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
def _read_header(path: Path) -> list[str] | None:
    """Return the header as a list of column names, or None if the file
    doesn't exist or is empty.
    """
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None