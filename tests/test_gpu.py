import pytest


@pytest.mark.gpu
def test_gpu():
    assert 1 + 1 == 2


@pytest.mark.gpu
def test_rapids_imports():
    import cudf
    import cugraph
    import cuml
    import cupy
    import cuvs


@pytest.mark.gpu
def test_cupy_device_available():
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() > 0


@pytest.mark.gpu
def test_cudf_basic():
    import cudf

    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(df) == 3
