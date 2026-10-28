import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from tests.ml.conftest import longitudinal


def test_split_fractions_and_reproducibility():
    edata = longitudinal()
    ep.ml.split(edata)
    assert edata.obs["split"].value_counts().to_dict() == {"train": 210, "tuning": 45, "held_out": 45}

    other = ep.ml.split(edata, random_state=0, copy=True)
    pd.testing.assert_series_equal(other.obs["split"], edata.obs["split"])
    assert not ep.ml.split(edata, random_state=1, copy=True).obs["split"].equals(edata.obs["split"])


def test_split_patients_are_disjoint():
    edata = longitudinal()
    ep.ml.split(edata, patient_key="patient")

    assert edata.obs.groupby("patient")["split"].nunique().max() == 1


def test_split_stratify():
    edata = longitudinal()
    ep.ml.split(edata, stratify="label")

    prevalence = edata.obs.groupby("split", observed=True)["label"].mean()
    np.testing.assert_allclose(prevalence, edata.obs["label"].mean(), atol=0.02)
    with pytest.raises(ValueError, match="differs between observations of a patient"):
        ep.ml.split(edata, patient_key="patient", stratify="label")


def test_split_by_time():
    edata = longitudinal()
    ep.ml.split(edata, time_key="admission")

    admission = edata.obs.groupby("split", observed=True)["admission"]
    assert admission.max()["train"] < admission.min()["tuning"]
    assert admission.max()["tuning"] < admission.min()["held_out"]
