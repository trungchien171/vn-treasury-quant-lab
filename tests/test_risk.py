import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vntql.risk import backtest, christoffersen, ewma_var, hs_var, kupiec, traffic_light

rng = np.random.default_rng(1)


def test_kupiec_zero_when_exceptions_match_expectation():
    exc = np.zeros(100, bool)
    exc[:5] = True
    lr, p = kupiec(exc, 0.05)
    assert lr == pytest.approx(0, abs=1e-12) and p == pytest.approx(1)


def test_kupiec_matches_formula_and_handles_zero_exceptions():
    n, x, q = 250, 9, 0.01
    exc = np.r_[np.ones(x, bool), np.zeros(n - x, bool)]
    pi = x / n
    expected = -2 * ((n - x) * np.log(1 - q) + x * np.log(q) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    assert kupiec(exc, q)[0] == pytest.approx(expected)
    assert kupiec(np.zeros(250, bool), q)[0] == pytest.approx(-2 * 250 * np.log(1 - q))


def test_christoffersen_flags_clustering():
    spread = np.zeros(1000, bool)
    spread[::50] = True                       # 20 isolated exceptions
    clustered = np.zeros(1000, bool)
    clustered[100:110] = clustered[600:610] = True
    assert christoffersen(spread)[1] > 0.1
    assert christoffersen(clustered)[1] < 0.001


def test_var_uses_only_past_data_and_covers_iid_normal():
    pnl = pd.Series(rng.normal(size=5000))
    v = hs_var(pnl, 0.99)
    assert np.isnan(v.iloc[249]) and v.iloc[250] == pytest.approx(-pnl.iloc[:250].quantile(0.01))
    assert ewma_var(pnl, 0.99).iloc[300:].mean() == pytest.approx(stats.norm.ppf(0.99), rel=0.05)
    res, _ = backtest(pnl, ewma_var(pnl, 0.99), 0.99)
    assert res["Kupiec p"] > 0.01 and res["rate %"] == pytest.approx(1, abs=0.5)


def test_traffic_light_zones():
    assert [traffic_light(n) for n in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]
