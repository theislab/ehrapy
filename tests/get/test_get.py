import ehrdata as ed

import ehrapy as ep


def test_obs_df():
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    df = ep.get.obs_df(edata, keys=["age"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(edata), 1)


def test_rank_features_groups_df():
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    ep.tl.rank_features_groups(edata, "service_unit")
    df = ep.get.rank_features_groups_df(edata, group="FICU")
    # since pass through of scanpy, merely testing shape
    assert df.shape == (54, 5)


def test_var_df():
    edata = ed.dt.mimic_2()
    edata = ep.pp.encode(edata, autodetect=True)
    df = ep.get.var_df(edata, keys=["0", "1", "2", "3"])
    # since pass through of scanpy, merely testing shape
    assert df.shape == (len(edata.var), 4)
