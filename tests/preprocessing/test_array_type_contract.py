import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from testing.fast_array_utils import Flags

import ehrapy as ep
from tests.conftest import forbid_dask_compute

ARRAY_FUNCTIONS = [
    pytest.param(ep.pp.scale_norm, {"with_mean": False}, None, 0, id="scale_norm"),
    pytest.param(ep.pp.minmax_norm, {}, None, 0, id="minmax_norm"),
    pytest.param(ep.pp.maxabs_norm, {}, None, 0, id="maxabs_norm"),
    pytest.param(ep.pp.robust_scale_norm, {"with_centering": False}, None, 0, id="robust_scale_norm"),
    pytest.param(ep.pp.quantile_norm, {"n_quantiles": 10}, None, 0, id="quantile_norm"),
    pytest.param(ep.pp.power_norm, {"standardize": False}, None, 0, id="power_norm"),
    pytest.param(ep.pp.log_norm, {}, None, 0, id="log_norm"),
    pytest.param(ep.pp.offset_negative_values, {}, None, 0, id="offset_negative_values"),
    pytest.param(ep.pp.explicit_impute, {"replacement": -1.0}, None, 0, id="explicit_impute"),
    pytest.param(ep.pp.simple_impute, {"strategy": "median"}, None, 0, id="simple_impute"),
    pytest.param(
        ep.pp.knn_impute,
        {"n_neighbors": 3, "var_names": ["0", "1", "2"], "backend": "scikit-learn"},
        None,
        0,
        id="knn_impute",
    ),
    pytest.param(ep.pp.miss_forest_impute, {"n_estimators": 10}, None, 0, id="miss_forest_impute"),
    pytest.param(ep.pp.locf_impute, {}, None, 0, id="locf_impute"),
    pytest.param(ep.pp.missing_data_mask, {}, "missing_data_mask", 0, id="missing_data_mask"),
    pytest.param(ep.pp.filter_features, {"min_obs": 15}, None, 1, id="filter_features"),
    pytest.param(ep.pp.filter_observations, {"min_vars": 3}, None, 1, id="filter_observations"),
    pytest.param(ep.pp.winsorize, {"limits": (0.1, 0.2)}, None, 0, id="winsorize"),
    pytest.param(ep.pp.clip_quantile, {"limits": (0.0, 2.0)}, None, 0, id="clip_quantile"),
    pytest.param(ep.pp.encode, {"encodings": {"one-hot": ["3"]}}, None, 1, id="encode"),
    pytest.param(ep.pp.summarize_measurements, {}, None, 0, id="summarize_measurements"),
    pytest.param(ep.pp.combat, {"batch_key": "batch"}, None, 0, id="combat"),
    pytest.param(ep.pp.regress_out, {"keys": ["covariate"]}, None, 0, id="regress_out"),
    pytest.param(ep.pp.sample, {"fraction": 0.5, "rng": 0}, None, 0, id="sample"),
]
LONGITUDINAL_ONLY = {ep.pp.locf_impute}
STATIC_COMPLETE_ONLY = {ep.pp.combat, ep.pp.regress_out}
# corrected values are dense, and MissForest refits its forests on all observations in every iteration
UNSUPPORTED = {ep.pp.combat: Flags.Sparse, ep.pp.regress_out: Flags.Sparse, ep.pp.miss_forest_impute: Flags.Dask}
NOT_ARRAY_FUNCTIONS = {
    "detect_bias",
    "highly_variable_features",
    "mcar_test",
    "neighbors",
    "pca",
    "qc_lab_measurements",
    "qc_metrics",
    "variable_correlations",
}


def test_array_functions_cover_preprocessing_api():
    expected = {getattr(ep.pp, name) for name in set(ep.pp.__all__) - NOT_ARRAY_FUNCTIONS}
    assert {param.values[0] for param in ARRAY_FUNCTIONS} == expected


@pytest.mark.array_type(skip=Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(("func", "kwargs", "written_layer", "dask_computes"), ARRAY_FUNCTIONS)
def test_array_type_contract(array_type, ndim, func, kwargs, written_layer, dask_computes, rng):
    if ndim == 3 and array_type.flags & Flags.Sparse:
        pytest.skip("sparse arrays are 2D")
    if ndim == 2 and func in LONGITUDINAL_ONLY:
        pytest.skip(f"{func.__name__} needs longitudinal data")
    if ndim == 3 and func in STATIC_COMPLETE_ONLY:
        pytest.skip(f"{func.__name__} needs 2D data")
    shape = (20, 4) if ndim == 2 else (20, 4, 3)
    X = np.where(rng.random(shape) < 0.5, 0.0, rng.gamma(2, size=shape))
    if func not in STATIC_COMPLETE_ONLY:
        X[rng.random(shape) < 0.1] = np.nan
    X[:, 3] = rng.integers(0, 3, size=X[:, 3].shape)
    X = array_type(X)
    obs = pd.DataFrame(
        {"batch": pd.Categorical(["a", "b"] * 10), "covariate": rng.normal(size=20)}, index=[str(i) for i in range(20)]
    )
    edata = ed.EHRData(X=X, obs=obs)
    edata.var[FEATURE_TYPE_KEY] = [NUMERIC_TAG] * 3 + [CATEGORICAL_TAG]

    if array_type.flags & UNSUPPORTED.get(func, Flags(0)):
        with pytest.raises(NotImplementedError):
            func(edata, **kwargs)
        return

    with forbid_dask_compute(allowed=dask_computes):
        result = func(edata, **kwargs)

    result = edata if result is None else result
    written = result.X if written_layer is None else result.layers[written_layer]
    assert type(written) is type(X)
    if array_type.flags & Flags.Dask:
        assert type(written._meta) is type(X._meta)
        assert type(written.compute()) is type(X._meta)
