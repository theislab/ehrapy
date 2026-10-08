from __future__ import annotations

import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from testing.fast_array_utils import Flags

import ehrapy as ep
from ehrapy.tools._comorbidity_tables import CHARLSON as CHARLSON_INDEX
from tests.conftest import forbid_dask_compute

VOCABULARIES_AND_CODES = [
    ("ICD10CM", "E11.9"),
    ("ICD10CM", "E1122"),
    ("ICD10", "c78.0"),
    ("ICD10GM", "C50.9"),
    ("ICD10CM", "K70.3"),
    ("ICD10CM", "K72.10"),
    ("ICD10CM", "I21.4"),
    ("ICD10CM", "I25.10"),
    ("SNOMED", "I21"),
    ("ICD10CM", "I50.9"),
    ("ICD10CM", "F32.9"),
    ("ICD10CM", "E66.01"),
    ("ICD10CM", "I10"),
    ("ICD10CM", "I13.0"),
]
CODES_OF_PATIENTS = [
    ["E11.9", "I21.4"],
    ["E11.9", "E1122"],
    ["c78.0", "C50.9"],
    ["K70.3", "K72.10"],
    ["I25.10", "I21"],
    ["I50.9", "K70.3", "F32.9"],
    ["E66.01", "F32.9"],
    ["I10", "I13.0"],
]
CHARLSON = [2, 2, 6, 3, 0, 2, 0, 1]
QUAN = [0, 1, 6, 4, 0, 4, 0, 2]
VAN_WALRAVEN = [0, 0, 12, 11, 0, 15, -7, 7]


def _var() -> pd.DataFrame:
    vocabularies, codes = zip(*VOCABULARIES_AND_CODES, strict=True)
    return pd.DataFrame(
        {"vocabulary": pd.array(vocabularies, dtype="string"), "code": pd.array(codes, dtype="string")},
        index=[f"{vocabulary}/{code}" for vocabulary, code in VOCABULARIES_AND_CODES],
    )


def _X() -> np.ndarray:
    codes = [code for _, code in VOCABULARIES_AND_CODES]
    X = np.zeros((len(CODES_OF_PATIENTS), len(codes)))
    for patient, patient_codes in enumerate(CODES_OF_PATIENTS):
        X[patient, [codes.index(code) for code in patient_codes]] = 1
    X[0, 0] = 3
    X[1, 6] = np.nan
    return X


def _edata(X) -> ed.EHRData:
    obs = pd.DataFrame(index=[f"patient_{i}" for i in range(X.shape[0])])
    tem = pd.DataFrame(index=[f"t{i}" for i in range(X.shape[2])]) if X.ndim == 3 else None
    return ed.EHRData(X, obs=obs, var=_var(), tem=tem)


@pytest.mark.parametrize(
    ("method", "weights", "expected"),
    [
        ("charlson", None, CHARLSON),
        ("charlson", "quan", QUAN),
        ("elixhauser", None, VAN_WALRAVEN),
    ],
)
def test_comorbidity_index_scores(method, weights, expected):
    edata = _edata(_X())

    ep.tl.comorbidity_index(edata, method=method, weights=weights)

    np.testing.assert_array_equal(edata.obs[method], expected)


def test_comorbidity_index_hierarchy():
    edata = _edata(_X())

    ep.tl.comorbidity_index(edata, key_added="cci")
    ep.tl.comorbidity_index(edata, method="elixhauser", key_added="eci")

    obs = edata.obs
    assert obs.loc["patient_0", "cci_diabetes_uncomplicated"]
    assert obs.loc["patient_1", ["cci_diabetes_complicated", "cci_diabetes_uncomplicated"]].tolist() == [True, False]
    assert obs.loc["patient_2", ["cci_metastatic_solid_tumor", "cci_malignancy"]].tolist() == [True, False]
    assert obs.loc["patient_3", ["cci_moderate_severe_liver_disease", "cci_mild_liver_disease"]].tolist() == [
        True,
        False,
    ]
    assert not obs.loc["patient_4", obs.columns.str.startswith("cci_")].any()
    assert obs.loc["patient_5", ["eci_liver_disease", "eci_alcohol_abuse", "eci_depression"]].all()
    assert obs.loc["patient_7", ["eci_hypertension_complicated", "eci_hypertension_uncomplicated"]].tolist() == [
        True,
        False,
    ]
    weights = pd.Series(CHARLSON_INDEX.weights["charlson"]).add_prefix("cci_")
    np.testing.assert_array_equal(obs[weights.index].astype(int) @ weights, obs["cci"])


def test_comorbidity_index_3D_tem_names():
    X = _X()
    X3 = np.zeros((*X.shape, 3))
    X3[:, :, 0] = X
    X3[2, 2] = [0, 0, 1]

    edata = _edata(X3)
    ep.tl.comorbidity_index(edata)
    np.testing.assert_array_equal(edata.obs["charlson"], CHARLSON)

    ep.tl.comorbidity_index(edata, tem_names=["t0", "t1"])
    assert edata.obs.loc["patient_2", "charlson"] == 2
    assert edata.obs.loc["patient_2", "charlson_malignancy"]

    with pytest.raises(ValueError, match="3D"):
        ep.tl.comorbidity_index(_edata(X), tem_names=["t0"])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
def test_comorbidity_index_array_types(array_type, ndim):
    X = _X() if ndim == 2 else np.repeat(_X()[:, :, None], 2, axis=2)
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("Sparse arrays are 2D.")
    edata = _edata(X)
    edata.X = array_type(X)

    with forbid_dask_compute(allowed=1):
        result = ep.tl.comorbidity_index(edata, method="elixhauser", copy=True)

    assert isinstance(result.X, array_type.cls)
    np.testing.assert_array_equal(result.obs["elixhauser"], VAN_WALRAVEN)


def test_comorbidity_index_errors():
    edata = _edata(_X())
    with pytest.raises(ValueError, match="quan"):
        ep.tl.comorbidity_index(edata, method="elixhauser", weights="quan")

    edata.var = edata.var.drop(columns="code")
    with pytest.raises(ValueError, match="annotate_codes"):
        ep.tl.comorbidity_index(edata)
