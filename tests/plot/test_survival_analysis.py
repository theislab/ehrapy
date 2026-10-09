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


def test_kaplan_meier_cumulative_incidence(mimic_2: EHRData):
    died, died_in_hospital = np.asarray(mimic_2[:, ["censor_flg", "hosp_exp_flg"]].X, dtype=float).T
    mimic_2.obs["death"] = np.select([died_in_hospital == 1, died == 1], [1, 2], 0)
    ajf = ep.tl.kaplan_meier(mimic_2, duration_col="mort_day_censored", event_col="death", event_of_interest=1)

    plot = ep.pl.kaplan_meier([ajf])
    assert isinstance(plot, hv.Overlay)
    (curve,) = (element for element in plot if type(element) is hv.Curve)
    assert curve.vdims[0].name == "Cumulative incidence"
    np.testing.assert_allclose(curve.dimension_values(1), ajf.cumulative_density_.iloc[:, 0])

    plot = ep.pl.kaplan_meier([ajf], display_survival_statistics=True, xlim=(0, 700))
    assert isinstance(plot, hv.Layout)
    table = plot.Table.I.dframe()
    expected = ajf.cumulative_density_.iloc[:, 0].asof(np.linspace(0, 700, 10)).to_numpy()
    np.testing.assert_allclose(table.iloc[0, 1:].astype(float), expected, atol=0.005)


def test_coxph_forestplot(mimic_2: EHRData):
    edata_subset = mimic_2[
        :, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]
    ].copy()
    ep.tl.cox_ph(edata_subset, duration_col="mort_day_censored", event_col="censor_flg")
    plot = ep.pl.cox_ph_forestplot(edata_subset)
    assert isinstance(plot, hv.Overlay)

    (vline,) = (element for element in plot if isinstance(element, hv.VLine))
    assert vline.data == 0
    (error_bars,) = (element for element in plot if isinstance(element, hv.ErrorBars))
    assert error_bars.horizontal
    np.testing.assert_allclose(error_bars.dimension_values("Variable"), [0, 1, 2])
    labels, header = (element for element in plot if isinstance(element, hv.Labels))
    assert header.dimension_values("y").min() - labels.dimension_values("y").max() >= 1


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


def test_ols_3D(edata_blobs_timeseries_small: EHRData):
    first = ep.pp.summarize_measurements(
        edata_blobs_timeseries_small,
        layer=DEFAULT_TEM_LAYER_NAME,
        var_names=["feature_0", "feature_1"],
        statistics=["first"],
    )

    plot = ep.pl.ols(edata_blobs_timeseries_small, x="feature_0", y="feature_1", layer=DEFAULT_TEM_LAYER_NAME)

    np.testing.assert_array_equal(plot.data["feature_0"], first.X[:, 0])


def test_ols_obs_columns_on_3D_data(edata_blobs_3d: EHRData):
    plot = ep.pl.ols(edata_blobs_3d, x="age", y="y")

    np.testing.assert_array_equal(plot.data["age"], edata_blobs_3d.obs["age"])


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
