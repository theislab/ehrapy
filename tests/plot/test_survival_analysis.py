from pathlib import Path

import numpy as np

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_kaplan_meier(mimic_2, check_same_image):
    censor_idx = mimic_2.var_names.get_indexer(["censor_flg"])
    mimic_2.X[:, censor_idx] = np.where(mimic_2.X[:, censor_idx] == 0, 1, 0)

    groups = mimic_2[:, ["service_unit"]].X
    adata_ficu = mimic_2[groups == "FICU"].copy()
    adata_micu = mimic_2[groups == "MICU"].copy()
    kmf_1 = ep.tl.kaplan_meier(adata_ficu, duration_col="mort_day_censored", event_col="censor_flg", label="FICU")
    kmf_2 = ep.tl.kaplan_meier(adata_micu, duration_col="mort_day_censored", event_col="censor_flg", label="MICU")
    fig, ax = ep.pl.kaplan_meier(
        [kmf_1, kmf_2],
        ci_show=[False, False, False],
        color=["k", "r"],
        xlim=[0, 750],
        ylim=[0, 1],
        xlabel="Days",
        ylabel="Proportion Survived",
    )

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/kaplan_meier",
        tol=2e-1,
    )

    fig, ax = ep.pl.kaplan_meier(
        [kmf_1, kmf_2],
        ci_show=[False, False, False],
        color=["k", "r"],
        xlim=[0, 750],
        ylim=[0, 1],
        xlabel="Days",
        ylabel="Proportion Survived",
        grid=True,
        display_survival_statistics=True,
    )

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/kaplan_meier_table",
        tol=2e-1,
    )


def test_coxph_forestplot(mimic_2, check_same_image):
    adata_subset = mimic_2[
        :, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]
    ].copy()
    ep.tl.cox_ph(adata_subset, duration_col="mort_day_censored", event_col="censor_flg")
    fig, ax = ep.pl.cox_ph_forestplot(adata_subset, fig_size=(12, 3), t_adjuster=0.15, marker="o", size=2, text_size=14)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/coxph_forestplot",
        tol=2e-1,
    )


def test_ols(mimic_2, check_same_image):
    adata_sample = mimic_2[:200].copy()
    co2_lm_result = ep.tl.ols(
        adata_sample, var_names=["pco2_first", "tco2_first"], formula="tco2_first ~ pco2_first", missing="drop"
    ).fit()
    ax = ep.pl.ols(
        adata_sample,
        x="pco2_first",
        y="tco2_first",
        ols_results=[co2_lm_result],
        ols_color=["red"],
        xlabel="PCO2",
        ylabel="TCO2",
        show=False,
    )

    fig = ax.figure

    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/ols",
        tol=2e-1,
    )
