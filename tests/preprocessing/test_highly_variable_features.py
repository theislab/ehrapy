import ehrapy as ep
from ehrapy.preprocessing._highly_variable_features import highly_variable_features


def test_highly_variable_features():
    adata = ep.dt.dermatology(encoded=True)
    ep.pp.knn_impute(adata)
    highly_variable_features(adata)

    assert "highly_variable" in adata.var.columns
    assert "highly_variable_rank" in adata.var.columns
    assert "means" in adata.var.columns
    assert "variances" in adata.var.columns
    assert "variances_norm" in adata.var.columns

    adata = ep.dt.dermatology(encoded=True)
    ep.pp.knn_impute(adata)
    highly_variable_features(adata, top_features_percentage=0.5)
    assert adata.var["highly_variable"].sum() == 17
