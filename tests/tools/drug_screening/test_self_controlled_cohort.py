import math

import pytest

import ehrapy as ep
from ehrapy.tools.drug_screening import RateRatioResult


def test_rate_ratio_test_balanced_case():
    result = ep.tl.rate_ratio_test([5, 5], [100, 100])

    assert isinstance(result, RateRatioResult)
    assert result.rate_ratio == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.conf_int[0] < 1 < result.conf_int[1]


def test_rate_ratio_test_one_sided_interval():
    result = ep.tl.rate_ratio_test([2, 10], [50, 100], alternative="less")

    assert result.rate_ratio == pytest.approx(0.4)
    assert 0 <= result.p_value <= 1
    assert result.conf_int[0] == 0.0
    assert math.isfinite(result.conf_int[1])
