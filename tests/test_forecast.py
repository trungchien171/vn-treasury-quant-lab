import numpy as np
import pandas as pd
import pytest

from vntql.forecast import (clark_west, evaluate, fit_mean, fit_ols, fit_ridge, fit_zero, ols_nw,
                            robust_month_end, walk_forward)

rng = np.random.default_rng(0)


def test_robust_month_end_ignores_one_day_spikes():
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    s = pd.Series(25000.0, index=idx)
    s.iloc[-1] = 250.0          # off-market print on the last day of February
    s.iloc[20] = 30000.0        # and one in late January
    m = robust_month_end(s)
    assert list(m.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert np.allclose(m, 25000.0)


def test_ridge_zero_penalty_is_ols_and_penalty_shrinks():
    X = rng.normal(size=(200, 3))
    y = 0.5 + X @ np.array([1.0, -2.0, 0.0]) + rng.normal(scale=0.1, size=200)
    A = np.column_stack([np.ones(200), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    Xn = rng.normal(size=(5, 3))
    assert fit_ols(X, y)(Xn) == pytest.approx(np.column_stack([np.ones(5), Xn]) @ b, abs=1e-10)
    big = fit_ridge(X, y, 1e6)(Xn)
    assert big == pytest.approx(np.full(5, y.mean()), abs=1e-4)


def test_walk_forward_has_no_look_ahead():
    n, h = 80, 3
    idx = pd.period_range("2015-01", periods=n, freq="M").to_timestamp("M")
    X = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.normal(size=n), index=idx)
    base = walk_forward(X, y, fit_ols, min_train=24, h=h)
    t = 50
    # targets y[t-h+1:] are not realised at origin t; X after t is not known: changing them must not move forecast t
    y2, X2 = y.copy(), X.copy()
    y2.iloc[t - h + 1:] += 100
    X2.iloc[t + 1:] += 100
    assert walk_forward(X2, y2, fit_ols, min_train=24, h=h).iloc[t] == pytest.approx(base.iloc[t])
    # ... while changing the last realised target does
    y3 = y.copy()
    y3.iloc[t - h] += 100
    assert walk_forward(X, y3, fit_ols, min_train=24, h=h).iloc[t] != pytest.approx(base.iloc[t])
    assert base.iloc[:24 + h - 1].isna().all() and base.iloc[24 + h - 1:].notna().all()


def test_drift_forecast_is_trailing_mean():
    idx = pd.period_range("2015-01", periods=40, freq="M").to_timestamp("M")
    y = pd.Series(np.arange(40.0), index=idx)
    f = walk_forward(pd.DataFrame({"x": 0.0}, index=idx), y, fit_mean, min_train=10)
    assert f.iloc[10] == pytest.approx(np.mean(np.arange(10.0)))


def test_clark_west_detects_real_signal_only():
    n = 300
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(size=n)
    zero = np.zeros(n)
    t_sig, p_sig = clark_west(y, zero, 0.6 * x)
    t_noise, p_noise = clark_west(y, zero, 0.6 * rng.normal(size=n))
    assert p_sig < 0.01 and p_noise > 0.05 and t_sig > t_noise


def test_evaluate_relative_metrics():
    idx = pd.period_range("2015-01", periods=100, freq="M").to_timestamp("M")
    y = pd.Series(rng.normal(size=100), index=idx)
    out = evaluate(y, {"Drift": pd.Series(0.0, index=idx), "Perfect": y}, bench="Drift")
    assert out.loc["Drift", "RMSE / Drift"] == 1
    assert out.loc["Perfect", "RMSE"] == 0 and out.loc["Perfect", "OOS R2 vs Drift %"] == 100
    assert out.loc["Perfect", "hit rate %"] == 100 and np.isnan(out.loc["Drift", "hit rate %"])


def test_ols_nw_recovers_coefficients():
    X = pd.DataFrame({"a": rng.normal(size=500)})
    y = pd.Series(1.0 + 2.0 * X["a"] + rng.normal(scale=0.5, size=500))
    tab, r2, n = ols_nw(X, y, lag=2)
    assert tab.loc["a", "coef"] == pytest.approx(2.0, abs=0.1) and tab.loc["a", "t (NW)"] > 20 and n == 500
    assert fit_zero(None, None)(np.ones((3, 1))) == pytest.approx(np.zeros(3))
