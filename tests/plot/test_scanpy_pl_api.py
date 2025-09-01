from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.style.use("default")

import os

import numpy as np

import ehrapy as ep

CURRENT_DIR = Path(__file__).parent
_TEST_IMAGE_PATH = f"{CURRENT_DIR}/_images"

# Set NUMBA_CPU_NAME to "generic" before importing or using numba
os.environ["NUMBA_CPU_NAME"] = "generic"


def test_scatter_plot(mimic_2, check_same_image):
    adata_subset = mimic_2[:, ["age", "icu_los_day"]].copy()
    ax = ep.pl.scatter(adata_subset, x="age", y="icu_los_day", color="#fe57a1", show=False)
    fig = ax.figure

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/scatter_scanpy_plt",
        tol=2e-1,
    )
    plt.close("all")


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
    plt.close("all")


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
    plt.close("all")


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
    plt.close("all")


def test_violin_plot(mimic_2, check_same_image):
    adata_sample = mimic_2[:200, ["age"]].copy()
    adata_sample.obs["service_unit"] = mimic_2[:200, "service_unit"].X.toarray().ravel().astype(str)
    adata_sample.obs["service_unit"] = adata_sample.obs["service_unit"].astype("category")

    ax = ep.pl.violin(adata_sample, keys="age", groupby="service_unit", show=False, jitter=False)
    fig = ax.get_figure()

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/violin_scanpy_plt",
        tol=2e-1,
    )
    plt.close("all")


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
    plt.close("all")


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
    plt.close("all")


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
    plt.close("all")


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

    # To see the numerical results

    groups = adata_sample.uns["rank_features_groups"]["names"].dtype.names
    for group in groups:
        print(f"\nGroup: {group}")
        names = adata_sample.uns["rank_features_groups"]["names"][group]
        scores = adata_sample.uns["rank_features_groups"]["scores"][group]
        pvals = adata_sample.uns["rank_features_groups"]["pvals"][group]
        print("Top features:")
        for name, score, pval in zip(names, scores, pvals, strict=False):
            print(f"  {name}: score={score:.4f}, pval={pval:.4e}")

    ax = ep.pl.rank_features_groups(adata_sample, key="rank_features_groups", groups=["MICU"], show=False)

    image = 0
    fig = ax[image].figure
    fig.set_size_inches(8, 6)

    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    # fig.savefig(f"{_TEST_IMAGE_PATH}/rank_features_groups_scanpy_test_output.png", dpi=80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_scanpy_plt",
        tol=2e-1,
    )
    plt.close("all")


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

    # To see the numerical results

    # groups = adata_sample.uns["rank_features_groups"]["names"].dtype.names
    # for group in groups:
    #     print(f"\nGroup: {group}")
    #     names = adata_sample.uns["rank_features_groups"]["names"][group]
    #     scores = adata_sample.uns["rank_features_groups"]["scores"][group]
    #     pvals = adata_sample.uns["rank_features_groups"]["pvals"][group]
    #     print("Top features:")
    #     for name, score, pval in zip(names, scores, pvals, strict=False):
    #         print(f"  {name}: score={score:.4f}, pval={pval:.4e}")

    ax = ep.pl.rank_features_groups_violin(
        adata_sample, groups=["SICU"], key="rank_features_groups", jitter=False, show=False, strip=False
    )

    # for some reason, scanpy's violinplot in the test run adds text labels to the plots;
    # can't reproduce in notebook or scripts.
    # because of this, remove text labels from the test-generated plot.
    for a in ax:
        for txt in a.texts:
            txt.remove()

    ax[0].set_ylim(-20, 140)

    fig = ax[0].figure
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_violin",
        tol=2e-1,
    )
    plt.close("all")


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

    # To see the numerical results

    groups = adata_sample.uns["rank_features_groups"]["names"].dtype.names
    for group in groups:
        print(f"\nGroup: {group}")
        names = adata_sample.uns["rank_features_groups"]["names"][group]
        scores = adata_sample.uns["rank_features_groups"]["scores"][group]
        pvals = adata_sample.uns["rank_features_groups"]["pvals"][group]
        print("Top features:")
        for name, score, pval in zip(names, scores, pvals, strict=False):
            print(f"  {name}: score={score:.4f}, pval={pval:.4e}")

    ax = ep.pl.rank_features_groups_stacked_violin(adata_sample, key="rank_features_groups", show=False, jitter=False)

    fig = ax["mainplot_ax"].figure
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_stacked_violin_scanpy",
        tol=2e-1,
    )
    plt.close("all")


def test_rank_features_groups_heatmap(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200, ["wbc_first", "hgb_first", "potassium_first", "tco2_first", "bun_first", "pco2_first"]
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")

    # To see the numerical results

    groups = adata_sample.uns["rank_features_groups"]["names"].dtype.names
    for group in groups:
        print(f"\nGroup: {group}")
        names = adata_sample.uns["rank_features_groups"]["names"][group]
        scores = adata_sample.uns["rank_features_groups"]["scores"][group]
        pvals = adata_sample.uns["rank_features_groups"]["pvals"][group]
        print("Top features:")
        for name, score, pval in zip(names, scores, pvals, strict=False):
            print(f"  {name}: score={score:.4f}, pval={pval:.4e}")

    ax = ep.pl.rank_features_groups_heatmap(adata_sample, key="rank_features_groups", show=False)

    fig = ax["heatmap_ax"].figure
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_heatmap_scanpy",
        tol=2e-1,
    )
    plt.close("all")


def test_rank_features_groups_dotplot(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200, ["wbc_first", "hgb_first", "potassium_first", "tco2_first", "bun_first"]
    ].copy()
    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")

    # To see the numerical results

    groups = adata_sample.uns["rank_features_groups"]["names"].dtype.names
    for group in groups:
        print(f"\nGroup: {group}")
        names = adata_sample.uns["rank_features_groups"]["names"][group]
        scores = adata_sample.uns["rank_features_groups"]["scores"][group]
        pvals = adata_sample.uns["rank_features_groups"]["pvals"][group]
        print("Top features:")
        for name, score, pval in zip(names, scores, pvals, strict=False):
            print(f"  {name}: score={score:.4f}, pval={pval:.4e}")

    ax = ep.pl.rank_features_groups_dotplot(
        adata_sample, key="rank_features_groups", groupby="service_unit", show=False
    )

    fig = ax["mainplot_ax"].get_figure()
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_dotplot_scanpy",
        tol=2e-1,
    )
    plt.close("all")


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

    # To see the numerical results

    groups = adata_sample.uns["rank_features_groups"]["names"].dtype.names
    for group in groups:
        print(f"\nGroup: {group}")
        names = adata_sample.uns["rank_features_groups"]["names"][group]
        scores = adata_sample.uns["rank_features_groups"]["scores"][group]
        pvals = adata_sample.uns["rank_features_groups"]["pvals"][group]
        print("Top features:")
        for name, score, pval in zip(names, scores, pvals, strict=False):
            print(f"  {name}: score={score:.4f}, pval={pval:.4e}")

    ax = ep.pl.rank_features_groups_matrixplot(
        adata_sample, key="rank_features_groups", groupby="service_unit", show=False
    )

    fig = ax["mainplot_ax"].figure

    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_matrixplot_scanpy",
        tol=2e-1,
    )
    plt.close("all")


def test_rank_features_groups_tracksplot(mimic_2_encoded, check_same_image):
    adata_sample = mimic_2_encoded[
        :200, ["age", "gender_num", "weight_first", "bmi", "sapsi_first", "day_icu_intime_num", "hour_icu_intime"]
    ].copy()

    ep.tl.rank_features_groups(adata_sample, groupby="service_unit")
    ax = ep.pl.rank_features_groups_tracksplot(adata_sample, key="rank_features_groups", show=False)

    fig = ax["groupby_ax"].figure

    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/rank_features_groups_tracksplot_scanpy",
        tol=2e-1,
    )
    plt.close("all")


def test_pca(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.pca(adata)
    ep.pp.neighbors(adata)

    ax = ep.pl.pca(adata, color="service_unit", show=False)
    fig = ax.figure

    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/pca",
        tol=2e-1,
    )
    plt.close("all")


def test_pca_loadings(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.pca(adata)
    ep.pp.neighbors(adata)

    ep.pl.pca_loadings(adata, components="1,2,3", show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/pca_loadings",
        tol=2e-1,
    )

    plt.close("all")


def test_pca_variance_ration(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.pca(adata)
    ep.pp.neighbors(adata)

    ep.pl.pca_variance_ratio(adata, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/pca_variance_ratio",
        tol=2e-1,
    )

    plt.close("all")


def test_pca_overview(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.pca(adata)
    ep.pp.neighbors(adata)

    ep.pl.pca_overview(adata, components="1,2", color="service_unit", show=False)

    for id, fignum in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fignum)
        if fignum == 2:
            fig.set_size_inches(12, 6)
        else:
            fig.set_size_inches(8, 6)
        fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

        check_same_image(
            fig=fig,
            base_path=f"{_TEST_IMAGE_PATH}/pca_overview_{id}",
            tol=2e-1,
        )


def test_tsne(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata)
    ep.tl.tsne(adata)

    ep.pl.tsne(
        adata,
        color=["day_icu_intime", "service_unit"],
        wspace=0.5,
        title=["Day of ICU admission", "Service unit"],
        show=False,
    )
    fig = plt.gcf()

    fig.set_size_inches(16, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/tsne",
        tol=2e-1,
    )


def test_umap_figure(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata, random_state=0)
    ep.tl.umap(adata, random_state=0)

    ep.pl.umap(
        adata,
        color="day_icu_intime",
        frameon=False,
        vmax=["p99.0", None, None],
        vcenter=[0.015, None, None],
        show=False,
    )
    fig = plt.gcf()

    fig.set_size_inches(16, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/umap",
        tol=2e-1,
    )


def test_umap_functionality(mimic_2_sample):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata, random_state=0)
    ep.tl.umap(adata, random_state=0)

    fig1 = ep.pl.umap(
        adata,
        color="day_icu_intime",
        frameon=False,
        vmax=["p99.0", None, None],
        vcenter=[0.015, None, None],
        show=False,
    )
    assert fig1 is not None

    fig2 = ep.pl.umap(
        adata,
        color=["day_icu_intime", "service_unit"],
        frameon=True,
        show=False,
    )
    assert fig2 is not None

    fig3 = ep.pl.umap(
        adata,
        color="day_icu_intime",
        frameon=False,
        cmap="viridis",
        show=False,
    )
    assert fig3 is not None

    plt.close("all")


def test_diffmap(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata)
    ep.tl.diffmap(adata)

    ep.pl.diffmap(adata, color="service_unit", show=False)
    fig = plt.gcf()

    fig.set_size_inches(16, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/diffmap",
        tol=2e-1,
    )


def test_paga_alternative(mimic_2_encoded, check_same_image):
    adata = mimic_2_encoded.copy()
    ep.pp.knn_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata)
    ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
    ep.tl.paga(adata, groups="leiden_0_5")
    ep.pl.paga(
        adata,
        color=["leiden_0_5", "day_28_flg"],
        cmap=ep.pl.Colormaps.grey_red.value,
        title=["Leiden 0.5", "Died in less than 28 days"],
        show=False,
    )

    fig = plt.gcf()

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/paga",
        tol=2e-1,
    )
    plt.close("all")


def test_draw_graph(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata)
    ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
    ep.tl.paga(adata, groups="leiden_0_5")
    ep.pl.paga(
        adata,
        color=["leiden_0_5", "day_28_flg"],
        cmap=ep.pl.Colormaps.grey_red.value,
        title=["Leiden 0.5", "Died in less than 28 days"],
        show=False,
    )

    fig_paga = plt.gcf()
    fig_paga.set_size_inches(16, 6)
    fig_paga.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig_paga,
        base_path=f"{_TEST_IMAGE_PATH}/draw_graph1",
        tol=2e-1,
    )

    plt.close("all")

    ep.tl.draw_graph(adata, init_pos="paga")
    ep.pl.draw_graph(adata, color=["leiden_0_5", "icu_exp_flg"], legend_loc="on data", show=False)

    fig_graph = plt.gcf()
    fig_graph.set_size_inches(16, 6)
    fig_graph.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig_graph,
        base_path=f"{_TEST_IMAGE_PATH}/draw_graph2",
        tol=2e-1,
    )


def test_embedding(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata)
    ep.tl.umap(adata)

    ep.pl.embedding(adata, "X_umap", color="icu_exp_flg", show=False)

    fig = plt.gcf()

    fig.set_size_inches(16, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    # fig.savefig(f"{_TEST_IMAGE_PATH}/embedding_test_output.png", dpi=80)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/embedding",
        tol=2e-1,
    )


def test_embedding_density(mimic_2_sample, check_same_image):
    adata = mimic_2_sample.copy()
    adata = adata[~np.isnan(adata.X).any(axis=1)].copy()
    adata = adata[:200, :].copy()
    adata = ep.pp.encode(adata, autodetect=True)

    ep.pp.simple_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata)

    ep.tl.umap(adata)
    ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
    ep.tl.embedding_density(adata, groupby="leiden_0_5", key_added="icu_exp_flg")

    ep.pl.embedding_density(adata, key="icu_exp_flg", show=False)

    fig = plt.gcf()

    fig.set_size_inches(16, 6)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/embedding_density",
        tol=2e-1,
    )


def test_dpt_timeseries(mimic_2_encoded, check_same_image):
    adata = mimic_2_encoded.copy()
    ep.pp.knn_impute(adata)
    ep.pp.log_norm(adata, offset=1)
    ep.pp.neighbors(adata, method="gauss")
    ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
    ep.tl.diffmap(adata, n_comps=10)

    adata.uns["iroot"] = np.flatnonzero(adata.obs["leiden_0_5"] == "0")[0]

    ep.tl.dpt(adata, n_branchings=3)
    ep.pl.dpt_timeseries(adata, show=False)

    fig = plt.gcf()
    check_same_image(
        fig=fig,
        base_path=f"{_TEST_IMAGE_PATH}/dpt_timeseries",
        tol=2e-1,
    )
