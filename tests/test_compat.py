# type: ignore
import warnings

import pytest
from anndata import AnnData
from ehrdata import EHRData

from ehrapy._compat import use_ehrdata


@pytest.fixture
def edata() -> EHRData:
    return EHRData()


@pytest.fixture
def adata() -> AnnData:
    return AnnData()


@use_ehrdata(deprecated_after="1.0.0")
def new_style_func(edata: EHRData | AnnData) -> str:
    return "processed with new style"


@use_ehrdata(deprecated_after="1.0.0")
def old_style_func(adata: EHRData | AnnData) -> str:
    return "processed with old style"


@use_ehrdata(deprecated_after="1.0.0", old_param="old_data", new_param="new_data")
def custom_param_func(new_data: EHRData | AnnData) -> str:
    return "processed with custom params"


def test_new_style_with_correct_param_and_ehrdata(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(edata=edata)
        assert len(w) == 0, "No warnings should be raised"


def test_new_style_with_old_param_and_ehrdata(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(adata=edata)
        assert len(w) == 1, "One warning should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert "Parameter 'adata' is deprecated" in str(w[0].message)


def test_new_style_with_correct_param_and_anndata(adata: AnnData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(edata=adata)
        assert len(w) == 1, "One warning should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert "Using AnnData with new_style_func is deprecated" in str(w[0].message)


def test_new_style_with_old_param_and_anndata(adata: AnnData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(adata=adata)
        assert len(w) == 2, "Two warnings should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert issubclass(w[1].category, FutureWarning)
        assert any("Parameter 'adata' is deprecated" in str(msg.message) for msg in w)
        assert any("Using AnnData with new_style_func is deprecated" in str(msg.message) for msg in w)


# Test cases for old style functions (using adata parameter)
def test_old_style_with_correct_param_and_ehrdata(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_style_func(adata=edata)
        assert len(w) == 0, "No warnings should be raised"


def test_old_style_with_new_param_and_ehrdata(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_style_func(edata=edata)
        assert len(w) == 0, "No warnings should be raised for correct type"


def test_old_style_with_correct_param_and_anndata(adata: AnnData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_style_func(adata=adata)
        assert len(w) == 1, "One warning should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert "Using AnnData with old_style_func is deprecated" in str(w[0].message)


def test_old_style_with_new_param_and_anndata(adata: AnnData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_style_func(edata=adata)
        assert len(w) == 1, "One warning should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert "Using AnnData with old_style_func is deprecated" in str(w[0].message)


# Test positional arguments
def test_positional_argument_with_ehrdata(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(edata)
        assert len(w) == 0, "No warnings should be raised"


def test_positional_argument_with_anndata(adata: AnnData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(adata)
        assert len(w) == 1, "One warning should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert "Using AnnData with new_style_func is deprecated" in str(w[0].message)


def test_custom_param_with_correct_param(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        custom_param_func(new_data=edata)
        assert len(w) == 0, "No warnings should be raised"


def test_custom_param_with_old_param(edata: EHRData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        custom_param_func(old_data=edata)
        assert len(w) == 1, "One warning should be raised"
        assert issubclass(w[0].category, FutureWarning)
        assert "Parameter 'old_data' is deprecated" in str(w[0].message)


def test_missing_parameter() -> None:
    with pytest.raises(TypeError) as excinfo:
        new_style_func()
    assert "missing required argument" in str(excinfo.value)


def test_version_in_deprecation_message(adata: AnnData) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_style_func(adata=adata)
        assert any("will be removed after version 1.0.0" in str(msg.message) for msg in w)


def test_invalid_function_signature() -> None:
    with pytest.raises(ValueError) as excinfo:

        @use_ehrdata()
        def invalid_func(wrong_param: int) -> None:
            pass

    assert "does not have parameter" in str(excinfo.value)
