import ehrapy as ep


def test_obs_df():
    adata = ep.dt.mimic_2(encoded=True)
    df = ep.get.obs_df(adata, keys=["age"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(adata), 1)


def test_rank_features_groups_df():
    adata = ep.dt.mimic_2(encoded=True)
    ep.tl.rank_features_groups(adata, "service_unit")
    df = ep.get.rank_features_groups_df(adata, group="FICU")
    # since pass through of scanpy, merely testing shape
    assert df.shape == (54, 5)


def test_var_df():
    adata = ep.dt.mimic_2(encoded=True)
    df = ep.get.var_df(adata, keys=["0", "1", "2", "3"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(adata.var), 4)
