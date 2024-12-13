from pathlib import Path

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_coxph_forestplot(mimic_2, check_same_image):
    adata_subset = mimic_2[:, ["mort_day_censored", "censor_flg", "gender_num", "afib_flg", "day_icu_intime_num"]]
    coxph = ep.tl.cox_ph(adata_subset, duration_col="mort_day_censored", event_col="censor_flg")
    fig, ax = ep.pl.cox_ph_forestplot(coxph, fig_size=(12, 3), t_adjuster=0.15, marker="o", size=2, text_size=14)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/coxph_forestplot",
        tol=2e-1,
    )
