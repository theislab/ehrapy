"""Design-matrix assembly for the causal module.

Treatment / outcome / covariate columns may live in either ``edata.X`` (or a layer) or ``edata.obs``.
The downstream estimators are sklearn-based and require an in-memory numpy design matrix; this module is therefore explicit about that constraint and rejects Dask- or sparse-backed ``edata.X`` with a clear error instead of pretending to support them by silently densifying.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


class Design(NamedTuple):
    """Design matrices extracted from an :class:`~ehrdata.EHRData` object.

    Attributes:
        T: 1D treatment vector, length n.
        Y: 1D outcome vector, length n.
        X: 2D covariate matrix after one-hot encoding, shape (n, p).
        feature_names: Column names of ``X`` (post one-hot expansion).
        index: Observation index of the rows that survived NaN filtering.
    """

    T: np.ndarray
    Y: np.ndarray
    X: np.ndarray
    feature_names: list[str]
    index: pd.Index


def _require_numpy_backing(arr, *, name: str) -> np.ndarray:
    """Validate that ``arr`` is a numpy ndarray (or squeezable 3D ndarray); error explicitly otherwise."""
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3:
            if arr.shape[2] != 1:
                raise ValueError(f"{name} has shape {arr.shape}; causal estimators expect 2D data.")
            return arr[:, :, 0]
        return arr
    raise NotImplementedError(
        f"Causal estimators require a numpy-backed EHRData; got {type(arr).__name__} for {name}. "
        "Materialise upstream (e.g. `edata.X = edata.X.compute()` for Dask or "
        "`edata.X = np.asarray(edata.X.todense())` for scipy.sparse) before calling the causal API."
    )


def _collect_columns(
    edata: EHRData,
    columns: Sequence[str],
    *,
    layer: str | None,
) -> pd.DataFrame:
    """Pull ``columns`` from either ``edata.var_names`` or ``edata.obs.columns`` into a single DataFrame.

    Var-side columns are read from ``edata.X`` (or the named layer).
    Obs-side columns are read from ``edata.obs`` and preserve their original dtype.
    """
    var_set = set(edata.var_names)
    obs_set = set(edata.obs.columns)
    missing = [c for c in columns if c not in var_set and c not in obs_set]
    if missing:
        raise KeyError(
            f"Columns {missing!r} not found in edata.var_names or edata.obs.columns. "
            "Treatment / outcome / covariates must reference existing variables or obs annotations."
        )

    var_cols = [c for c in columns if c in var_set]
    obs_cols = [c for c in columns if c in obs_set and c not in var_set]

    parts: list[pd.DataFrame] = []
    if var_cols:
        raw = edata.X if layer is None else edata.layers[layer]
        arr = _require_numpy_backing(raw, name=f"edata.layers[{layer!r}]" if layer else "edata.X")
        col_indices = [list(edata.var_names).index(c) for c in var_cols]
        parts.append(pd.DataFrame(arr[:, col_indices], columns=var_cols, index=edata.obs.index))
    if obs_cols:
        parts.append(edata.obs[obs_cols])

    df = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0] if parts else pd.DataFrame(index=edata.obs.index)
    return df.loc[:, list(columns)].copy()


def build_design(
    edata: EHRData,
    *,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    layer: str | None = None,
    dropna: bool = True,
) -> Design:
    """Assemble (T, Y, X) arrays from an :class:`~ehrdata.EHRData` object.

    Categorical covariates (object / category dtype) are one-hot encoded with the first level dropped.
    Boolean covariates are cast to float.
    Numeric covariates pass through.

    Args:
        edata: Central data object.
        treatment: Column name of the treatment variable.
        outcome: Column name of the outcome variable.
        covariates: Column names of the adjustment set; may be empty.
        layer: Layer to use for variables drawn from ``edata.X``.
        dropna: If ``True``, drop rows where any of T, Y, or covariates is NaN.

    Returns:
        A :class:`Design` named tuple.

    Raises:
        KeyError: If any of ``treatment``, ``outcome``, or ``covariates`` cannot be found.
        ValueError: If the design has fewer than two complete rows after NaN filtering.
        NotImplementedError: If ``edata.X`` (or the named layer) is not a numpy ndarray.
    """
    covariates = list(covariates)
    if treatment in covariates or outcome in covariates:
        raise ValueError("`treatment` and `outcome` must not appear in `covariates`.")
    if treatment == outcome:
        raise ValueError("`treatment` and `outcome` must differ.")

    df = _collect_columns(edata, [treatment, outcome, *covariates], layer=layer)

    if dropna:
        df = df.dropna(axis=0, how="any")

    if len(df) < 2:
        raise ValueError(f"Design matrix has only {len(df)} complete rows; need at least 2.")

    t_series = df[treatment]
    y_series = df[outcome]
    cov_df = df[covariates] if covariates else pd.DataFrame(index=df.index)

    T = pd.to_numeric(t_series, errors="raise").to_numpy(dtype=float)
    Y = pd.to_numeric(y_series, errors="raise").to_numpy(dtype=float)

    if cov_df.shape[1] == 0:
        X = np.zeros((len(df), 0), dtype=float)
        feature_names: list[str] = []
    else:
        cov_encoded = pd.get_dummies(cov_df, drop_first=True, dummy_na=False).astype(float)
        X = cov_encoded.to_numpy(dtype=float)
        feature_names = list(cov_encoded.columns)

    return Design(T=T, Y=Y, X=X, feature_names=feature_names, index=df.index)


def assert_binary_treatment(T: np.ndarray, treatment: str) -> None:
    """Validate that the treatment is binary (0/1)."""
    unique = np.unique(T[~np.isnan(T)])
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"Treatment '{treatment}' must be binary with values in {{0, 1}}; got {unique.tolist()!r}.")
    if len(unique) < 2:
        raise ValueError(f"Treatment '{treatment}' has only one observed level; cannot estimate an effect.")
