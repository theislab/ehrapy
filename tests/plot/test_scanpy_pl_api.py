from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.style.use("default")

import numpy as np

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"


def test_scatter_plot(mimic_2, check_same_image):
    adata_subset = mimic_2[:, ["age", "icu_los_day"]].copy()
    ax = ep.pl.scatter(adata_subset, x="age", y="icu_los_day", color="#fe57a1", show=False)
    fig = ax.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/scatter_scanpy_plt",
        tol=2e-1,
    )


def test_heatmap_plot(adata_mini, check_same_image):
    ax_dict = ep.pl.heatmap(
        adata_mini,
        var_names=["idx", "sys_bp_entry", "dia_bp_entry", "glucose", "weight", "in_days"],
        groupby="station",
        show=False,
        figsize=(5, 6),
    )

    fig = ax_dict["heatmap_ax"].get_figure()

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/heatmap_scanpy_plt",
        tol=2e-1,
    )


def test_dotplot_plot(mimic_2, check_same_image):
    adata = mimic_2[
        :,
        [
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
            "service_unit",
        ],
    ].copy()
    adata.obs["service_unit"] = mimic_2[:, "service_unit"].X.toarray().ravel().astype(str)
    adata._inplace_subset_var([var for var in adata.var_names if var != "service_unit"])
    adata.X = adata.X.astype(float)

    ax = ep.pl.dotplot(
        adata,
        var_names=[
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
        ],
        groupby="service_unit",
        show=False,
        figsize=(5, 6),
    )

    fig = ax["mainplot_ax"].get_figure()
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/dotplot_scanpy_plt",
        tol=2e-1,
    )


def test_tracks_plot(mimic_2, check_same_image):
    adata_sample = mimic_2[
        :200, ["age", "gender_num", "weight_first", "bmi", "sapsi_first", "day_icu_intime_num", "hour_icu_intime"]
    ].copy()
    adata_sample.obs["service_unit"] = mimic_2[:200, "service_unit"].X.toarray().ravel().astype(str)
    adata_sample.obs["service_unit"] = adata_sample.obs["service_unit"].astype("category")
    adata_sample._inplace_subset_var([var for var in adata_sample.var_names if var != "service_unit"])
    adata_sample.X = adata_sample.X.astype(float)

    ax = ep.pl.tracksplot(
        adata_sample,
        var_names=["age", "gender_num", "weight_first", "bmi", "sapsi_first", "day_icu_intime_num", "hour_icu_intime"],
        groupby="service_unit",
        show=False,
    )
    fig = ax["groupby_ax"].figure
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/tracksplot_scanpy_plt",
        tol=2e-1,
    )


def test_violin_plot(mimic_2, check_same_image):
    adata_sample = mimic_2[:200, ["age"]].copy()
    adata_sample.obs["service_unit"] = mimic_2[:200, "service_unit"].X.toarray().ravel().astype(str)
    adata_sample.obs["service_unit"] = adata_sample.obs["service_unit"].astype("category")

    ax = ep.pl.violin(adata_sample, keys="age", groupby="service_unit", show=False)
    fig = ax.get_figure()

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/violin_scanpy_plt",
        tol=2e-1,
    )


def test_matrix_plot(mimic_2, check_same_image):
    adata_sample = mimic_2[
        :200,
        [
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
        ],
    ].copy()
    adata_sample.obs["service_unit"] = mimic_2[:200, "service_unit"].X.toarray().ravel().astype(str)
    adata_sample.obs["service_unit"] = adata_sample.obs["service_unit"].astype("category")
    adata_sample.X = adata_sample.X.astype(float)

    ax = ep.pl.matrixplot(
        adata_sample,
        var_names=[
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
        ],
        groupby="service_unit",
        show=False,
    )
    fig = ax["mainplot_ax"].figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/matrix_scanpy_plot",
        tol=2e-1,
    )


def test_stacked_violin_plot(mimic_2, check_same_image):
    var_names = ["icu_los_day", "hospital_los_day", "age", "gender_num", "weight_first", "bmi"]

    adata_sample = mimic_2[:200, var_names].copy()
    adata_sample.obs["service_unit"] = mimic_2[:200, "service_unit"].X.toarray().ravel().astype(str)
    adata_sample.obs["service_unit"] = adata_sample.obs["service_unit"].astype("category")
    adata_sample.X = adata_sample.X.astype(float)

    ax = ep.pl.stacked_violin(
        adata_sample,
        var_names=["icu_los_day", "hospital_los_day", "age", "gender_num", "weight_first", "bmi"],
        groupby="service_unit",
        show=False,
    )
    fig = ax["mainplot_ax"].figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/stacked_violin_scanpy_plt",
        tol=2e-1,
    )


def test_clustermap(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[:200, ["abg_count", "wbc_first", "hgb_first"]].copy()

    mask = np.isfinite(adata_sample.X).all(axis=1)
    adata_clean = adata_sample[mask].copy()

    ax = ep.pl.clustermap(adata_clean, show=False)
    fig = ax.figure
    fig.set_dpi(80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/clustermap_scanpy",
        tol=2e-1,
    )


def test_rank_features_groups(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200,
        [
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
        ],
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups(adata_sample, key="rank_features_groups", show=False)

    image = 0
    fig = ax[image].figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_scanpy_plt",
        tol=2e-1,
    )


def test_rank_features_groups_violin(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200,
        [
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
        ],
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_violin(adata_sample, key="rank_features_groups", show=False, jitter=False)

    image = 0
    fig = ax[image].figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_violin_scanpy",
        tol=2e-1,
    )


def test_rank_features_groups_stacked_violin(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200,
        [
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
            "pco2_first",
        ],
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_stacked_violin(adata_sample, key="rank_features_groups", show=False, jitter=False)

    fig = ax["mainplot_ax"].figure
    fig.set_dpi(80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_stacked_violin_scanpy",
        tol=2e-1,
    )


def test_rank_features_groups_heatmap(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200, ["wbc_first", "hgb_first", "potassium_first", "tco2_first", "bun_first", "pco2_first"]
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_heatmap(adata_sample, key="rank_features_groups", show=False)

    fig = ax["heatmap_ax"].figure
    fig.set_dpi(80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_heatmap_scanpy",
        tol=2e-1,
    )


def test_rank_features_groups_dotplot(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200, ["wbc_first", "hgb_first", "potassium_first", "tco2_first", "bun_first"]
    ].copy()
    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_dotplot(
        adata_sample, key="rank_features_groups", groupby="service_unit", show=False
    )

    fig = ax["mainplot_ax"].get_figure()
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_dotplot_scanpy",
        tol=2e-1,
    )


def test_rank_features_groups_matrixplot(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200,
        [
            "abg_count",
            "wbc_first",
            "hgb_first",
            "potassium_first",
            "tco2_first",
            "bun_first",
            "creatinine_first",
        ],
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_matrixplot(
        adata_sample, key="rank_features_groups", groupby="service_unit", show=False
    )

    fig = ax["mainplot_ax"].figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_matrixplot_scanpy",
        tol=2e-1,
    )


def test_rank_features_groups_tracksplot(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200, ["age", "gender_num", "weight_first", "bmi", "sapsi_first", "day_icu_intime_num", "hour_icu_intime"]
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_tracksplot(adata_sample, key="rank_features_groups", show=False)

    fig = ax["groupby_ax"].figure
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_tracksplot_scanpy",
        tol=2e-1,
    )
