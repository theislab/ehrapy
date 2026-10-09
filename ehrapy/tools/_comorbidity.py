from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from ehrdata._logger import logger
from fast_array_utils.types import CSBase, DaskArray

from ehrapy._compat import _map_observation_blocks, _materialize
from ehrapy.get._get import _resolve_axis
from ehrapy.tools._comorbidity_tables import COMORBIDITY_INDICES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ehrdata import EHRData

ICD_VOCABULARIES: Mapping[str, frozenset[str]] = {
    "ICD10": frozenset({"ICD10", "ICD10CM", "ICD10GM", "ICD10CN"}),
    "ICD9": frozenset({"ICD9", "ICD9CM"}),
}
MAPPED_CODES = "icd_codes"


def _variable_codes(var: pd.DataFrame) -> pd.DataFrame:
    """ICD revision and undotted code of the codes of every variable and of the ICD codes mapped to it, indexed by the position of the variable."""
    codes = pd.DataFrame({"vocabulary": var["vocabulary"].array, "code": var["code"].array})
    if MAPPED_CODES in var.columns:
        mapped = (
            pd.Series(var[MAPPED_CODES].array, index=codes.index).astype("string").str.split("|").explode().dropna()
        )
        if len(mapped):
            mapped = mapped.str.split("/", n=1, expand=True).reindex(columns=[0, 1])
            codes = pd.concat([codes, mapped.set_axis(["vocabulary", "code"], axis=1)])
    revisions = {
        vocabulary: revision for revision, vocabularies in ICD_VOCABULARIES.items() for vocabulary in vocabularies
    }
    return pd.DataFrame(
        {
            "revision": codes["vocabulary"].astype("string").str.upper().map(revisions),
            "code": codes["code"].astype("string").str.replace(".", "", regex=False).str.upper(),
        },
        index=codes.index,
    )


@singledispatch
def _has_categories(X: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Whether every observation has a positive value at any timepoint in any variable of each category."""
    present = X > 0
    if X.ndim == 3:
        present = present.any(axis=2)
    return present.astype(np.float64) @ membership > 0


@_has_categories.register(CSBase)
def _(X: CSBase, membership: np.ndarray) -> np.ndarray:
    return np.asarray((X > 0).astype(np.float64) @ membership) > 0


@_has_categories.register(DaskArray)
def _(X: DaskArray, membership: np.ndarray) -> DaskArray:
    return _map_observation_blocks(
        X,
        _has_categories,
        membership,
        chunks=(X.chunks[0], (membership.shape[1],)),
        drop_axis=2 if X.ndim == 3 else [],
        meta=np.empty((0, 0), dtype=bool),
    )


def comorbidity_index(
    edata: EHRData,
    *,
    method: Literal["charlson", "elixhauser"] = "charlson",
    weights: Literal["charlson", "quan", "van_walraven"] | None = None,
    layer: str | None = None,
    tem_names: Any | Sequence[Any] | slice | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Score the comorbidity burden of every observation with the Charlson or Elixhauser comorbidity index.

    The variables are diagnosis codes, described by the `vocabulary` and `code` columns of `edata.var` as written by `ehrdata.annotate_codes`.
    Codes of the vocabularies `ICD9` and `ICD9CM`, and `ICD10`, `ICD10CM`, `ICD10GM` and `ICD10CN` are assigned to comorbidity categories by the ICD-9-CM and ICD-10 coding algorithms of :cite:p:`Quan2005`, matching codes with or without dots by their prefix.
    Variables of other vocabularies, such as SNOMED, count with the ICD codes mapped to them in `edata.var["icd_codes"]`, which `ehrdata.annotate_codes` writes with an OMOP CDM database, and belong to every category of any of these codes.
    Variables without an ICD code are ignored with a warning.
    An observation has a comorbidity if any of its variables in the category has a positive value, such as a presence flag or a count.
    For 3D data, a code counts at any timepoint in `tem_names`.

    If an observation has both the milder and the more severe form of a comorbidity, only the more severe form is kept.
    For the Charlson index, moderate or severe liver disease supersedes mild liver disease, diabetes with chronic complications supersedes diabetes without, and a metastatic solid tumor supersedes any malignancy.
    For the Elixhauser index, complicated hypertension and diabetes supersede their uncomplicated forms, and metastatic cancer supersedes a solid tumor without metastasis.

    The score is the sum of the weights of the comorbidities of an observation.
    The Charlson index is weighted by the original weights of :cite:p:`Charlson1987` or the updated weights of :cite:p:`Quan2011`.
    The Elixhauser index is weighted by the weights of :cite:p:`vanWalraven2009`, which weigh both forms of hypertension and diabetes with 0.

    Args:
        edata: Central data object.
        method: The comorbidity index to compute.
        weights: The weights of the comorbidity categories: `"charlson"` or `"quan"` for the Charlson index and `"van_walraven"` for the Elixhauser index.
            Defaults to `"charlson"` for the Charlson and `"van_walraven"` for the Elixhauser index.
        layer: The layer to read the codes from.
            If `None`, `edata.X` is used.
        tem_names: For 3D data, the timepoints of `edata.tem.index` to look for codes in, such as a look-back window.
            If `None`, all timepoints are used.
        key_added: The score is stored in `edata.obs[key_added]` and whether an observation has the comorbidity `category` in `edata.obs[f"{key_added}_{category}"]`.
            Defaults to `method`.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> codes = ["E11.9", "E11.22", "C78.0", "I21.0"]
        >>> var = pd.DataFrame({"vocabulary": "ICD10CM", "code": codes}, index=[f"ICD10CM/{code}" for code in codes])
        >>> X = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0]])
        >>> edata = ed.EHRData(X, obs=pd.DataFrame(index=["a", "b", "c"]), var=var)
        >>> ep.tl.comorbidity_index(edata)
        >>> edata.obs[["charlson", "charlson_diabetes_uncomplicated", "charlson_diabetes_complicated"]]
           charlson  charlson_diabetes_uncomplicated  charlson_diabetes_complicated
        a         2                             True                          False
        b         2                            False                           True
        c         6                            False                          False
    """
    if method not in COMORBIDITY_INDICES:
        raise ValueError(f"Unknown method {method!r}. Choose one of {list(COMORBIDITY_INDICES)}.")
    index = COMORBIDITY_INDICES[method]
    scheme: str = next(iter(index.weights)) if weights is None else weights
    if scheme not in index.weights:
        raise ValueError(f"The {method} index takes the weights {list(index.weights)}, not {scheme!r}.")
    if missing := {"vocabulary", "code"} - set(edata.var.columns):
        raise ValueError(
            f"edata.var lacks the columns {sorted(missing)}, which describe the code of every variable. "
            "Annotate variables named like `ICD10CM/I21.0` with `ed.annotate_codes(edata)` first."
        )
    codes = _variable_codes(edata.var)
    matchable = np.zeros(edata.n_vars, dtype=bool)
    matchable[codes.index[codes["revision"].notna()]] = True
    hint = "Map codes of other vocabularies, such as SNOMED, to ICD codes with `ed.annotate_codes(edata, backend_handle=...)` and an OMOP CDM database."
    if not matchable.any():
        raise ValueError(f"No variable has an ICD-9 or ICD-10 code. {hint}")
    if not matchable.all():
        unmatched = edata.var["vocabulary"][~matchable].astype("string").fillna("no vocabulary").value_counts()
        logger.warning(
            f"Ignoring {(~matchable).sum()} variables without an ICD-9 or ICD-10 code "
            f"({', '.join(f'{vocabulary}: {n}' for vocabulary, n in unmatched.items())}). {hint}"
        )

    categories = list(index.weights[scheme])
    membership = np.zeros((edata.n_vars, len(categories)), dtype=bool)
    for revision, prefixes_of_categories in index.codes.items():
        revision_codes = codes["code"][codes["revision"] == revision]
        for i, prefixes in enumerate(prefixes_of_categories.values()):
            membership[revision_codes.index[revision_codes.str.startswith(prefixes, na=False)], i] = True

    X = edata.X if layer is None else edata.layers[layer]
    if tem_names is not None:
        if X.ndim != 3:
            raise ValueError(f"tem_names selects timepoints of 3D data, but the data has shape {X.shape}.")
        tem_pos, _ = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")
        X = X[:, :, tem_pos]

    used = np.flatnonzero(membership.any(axis=1))
    if len(used):
        (has,) = _materialize(_has_categories(X[:, used], membership[used].astype(np.float64)))
    else:
        has = np.zeros((edata.n_obs, len(categories)), dtype=bool)

    for milder, severe in index.hierarchy:
        has[:, categories.index(milder)] &= ~has[:, categories.index(severe)]

    edata = edata.copy() if copy else edata
    key_added = method if key_added is None else key_added
    edata.obs[key_added] = has.astype(np.int64) @ np.array([index.weights[scheme][c] for c in categories])
    for i, category in enumerate(categories):
        edata.obs[f"{key_added}_{category}"] = has[:, i]
    return edata if copy else None
