from pathlib import Path

import numpy as np

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_kaplan_meier(mimic_2, check_same_image):
    mimic_2[:, ["censor_flg"]].X = np.where(mimic_2[:, ["censor_flg"]].X == 0, 1, 0)
    groups = mimic_2[:, ["service_unit"]].X
    adata_ficu = mimic_2[groups == "FICU"]
    adata_micu = mimic_2[groups == "MICU"]
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
        display_table=True,
    )

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/kaplan_meier_table",
        tol=2e-1,
    )


def test_coxph_forestplot(mimic_2, check_same_image):
    adata_subset = mimic_2[:, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]]
    ep.tl.cox_ph(adata_subset, duration_col="mort_day_censored", event_col="censor_flg")
    fig, ax = ep.pl.cox_ph_forestplot(adata_subset, fig_size=(12, 3), t_adjuster=0.15, marker="o", size=2, text_size=14)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/coxph_forestplot",
        tol=2e-1,
    )
