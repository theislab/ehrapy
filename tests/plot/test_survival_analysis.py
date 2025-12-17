from collections.abc import Callable
from pathlib import Path

import holoviews as hv
import numpy as np
from ehrdata import EHRData

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_kaplan_meier(mimic_2: EHRData):
    censor_idx = mimic_2.var_names.get_indexer(["censor_flg"])
    mimic_2.X[:, censor_idx] = np.where(mimic_2.X[:, censor_idx] == 0, 1, 0)

    groups = mimic_2[:, ["service_unit"]].X
    adata_ficu = mimic_2[groups == "FICU"].copy()
    adata_micu = mimic_2[groups == "MICU"].copy()
    kmf_1 = ep.tl.kaplan_meier(adata_ficu, duration_col="mort_day_censored", event_col="censor_flg", label="FICU")
    kmf_2 = ep.tl.kaplan_meier(adata_micu, duration_col="mort_day_censored", event_col="censor_flg", label="MICU")

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
    adata_subset = mimic_2[
        :, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]
    ].copy()
    ep.tl.cox_ph(adata_subset, duration_col="mort_day_censored", event_col="censor_flg")
    plot = ep.pl.cox_ph_forestplot(adata_subset)
    assert plot is not None
    assert isinstance(plot, hv.Overlay)


def test_ols(mimic_2: EHRData):
    adata_sample = mimic_2[:200].copy()
    co2_lm_result = ep.tl.ols(
        adata_sample, var_names=["pco2_first", "tco2_first"], formula="tco2_first ~ pco2_first", missing="drop"
    ).fit()
    plot = ep.pl.ols(
        adata_sample,
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
