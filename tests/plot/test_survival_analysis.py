from collections.abc import Callable
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_kaplan_meier(mimic_2: EHRData):
    censor_idx = mimic_2.var_names.get_indexer(["censor_flg"])
    mimic_2.X[:, censor_idx] = np.where(mimic_2.X[:, censor_idx] == 0, 1, 0)

    groups = mimic_2[:, ["service_unit"]].X
    edata_ficu = mimic_2[groups == "FICU"].copy()
    edata_micu = mimic_2[groups == "MICU"].copy()
    kmf_1 = ep.tl.kaplan_meier(edata_ficu, duration_col="mort_day_censored", event_col="censor_flg", label="FICU")
    kmf_2 = ep.tl.kaplan_meier(edata_micu, duration_col="mort_day_censored", event_col="censor_flg", label="MICU")

    plot = ep.pl.kaplan_meier(
        [kmf_1, kmf_2],
        ci_show=[False, False],
        color=["k", "r"],
        xlim=(0, 750),
        ylim=(0, 1),
        xlabel="Days",
        ylabel="Proportion Survived",
    )
    assert plot is not None
    assert isinstance(plot, hv.Overlay)

    plot = ep.pl.kaplan_meier(
        [kmf_1, kmf_2],
        ci_show=[False, False],
        color=["black", "red"],
        xlim=(0, 750),
        ylim=(0, 1),
        xlabel="Days",
        ylabel="Proportion Survived",
        display_survival_statistics=True,
    )
    assert plot is not None
    assert isinstance(plot, hv.Layout)


def test_coxph_forestplot(mimic_2: EHRData):
    edata_subset = mimic_2[
        :, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]
    ].copy()
    ep.tl.cox_ph(edata_subset, duration_col="mort_day_censored", event_col="censor_flg")
    plot = ep.pl.cox_ph_forestplot(edata_subset)
    assert plot is not None
    assert isinstance(plot, hv.Overlay)


def test_ols(mimic_2: EHRData):
    edata_sample = mimic_2[:200].copy()
    co2_lm_result = ep.tl.ols(
        edata_sample, var_names=["pco2_first", "tco2_first"], formula="tco2_first ~ pco2_first", missing="drop"
    ).fit()
    plot = ep.pl.ols(
        edata_sample,
        x="pco2_first",
        y="tco2_first",
        ols_results=[co2_lm_result],
        ols_color=["red"],
        xlabel="PCO2",
        ylabel="TCO2",
    )
    assert plot is not None
    assert isinstance(plot, (hv.Overlay, hv.Scatter))
    assert plot.opts.get().kwargs["xlabel"] == "PCO2"
    assert plot.opts.get().kwargs["ylabel"] == "TCO2"


def test_ols_layer_and_sparse(rng: np.random.Generator):
    X = rng.standard_normal((20, 2))
    edata = EHRData(X=sp.csr_array(X), layers={"doubled": 2 * X}, var=pd.DataFrame(index=["a", "b"]))

    plot = ep.pl.ols(edata, x="a", y="b")
    np.testing.assert_allclose(plot.data["b"], X[:, 1])

    plot = ep.pl.ols(edata, x="a", y="b", layer="doubled")
    np.testing.assert_allclose(plot.data["b"], 2 * X[:, 1])


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
def test_ols_array_types(array_type, rng: np.random.Generator):
    X = rng.standard_normal((20, 3))
    var = pd.DataFrame(index=["a", "b", "c"])
    expected = ep.pl.ols(EHRData(X=X, var=var), x="a", y="c")
    edata = EHRData(X=array_type(X), var=var)

    with forbid_dask_compute(allowed=1):
        plot = ep.pl.ols(edata, x="a", y="c")

    pd.testing.assert_frame_equal(plot.data, expected.data)


def test_ols_3d_raises(edata_blobs_timeseries_small: EHRData):
    with pytest.raises(ValueError, match="only supports 2D data"):
        ep.pl.ols(edata_blobs_timeseries_small, x="feature_0", y="feature_1", layer=DEFAULT_TEM_LAYER_NAME)


def test_cox_ph_adjusted_curves(mimic_2_adjusted_sa):
    edata_sample = mimic_2_adjusted_sa
    duration_col, event_col = "mort_day_censored", "censor_flg"
    cph = ep.tl.cox_ph(
        edata_sample,
        duration_col=duration_col,
        event_col=event_col,
        formula="sapsi_first + afib_flg",
        layer="layer_2",
    )
    ep.tl.cox_ph_adjusted_curves(
        edata_sample,
        cph=cph,
        strata="aline_flg",
        duration_col=duration_col,
        event_col=event_col,
        method="average",
        n_bootstrap=10,
        key_added="test_adjusted",
        layer="layer_2",
    )
    plot = ep.pl.cox_ph_adjusted_curves(edata_sample, key="test_adjusted")

    assert plot is not None
    assert isinstance(plot, hv.Overlay)
