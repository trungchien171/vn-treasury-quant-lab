import numpy as np
import pytest

from vntql.bonds import Bond, scenario_shocks, portfolio_report
from vntql.curves import InterpCurve, NSSCurve, bootstrap_par, fit_nss
from vntql.data import parse_curve

# HNX curve on 21/10/2026 (continuous spot, par, annual spot), in %
HNX = {
    0.25: (3.7152278906, None), 0.5: (3.7518986593, None), 0.75: (3.7869254678, None),
    1: (3.8203862088, 3.8943012382293), 2: (3.9400044456, 4.01620129338198),
    3: (4.0399306001, 4.11676353571102), 5: (4.1941396715, 4.26907738528258),
    7: (4.3043695423, 4.37521861665404), 10: (4.4176024117, 4.48106919641583),
    15: (4.5327774372, 4.58380652982641), 20: (4.6073680114, 4.64562614416746),
}
T = np.array(list(HNX))
Z = np.array([v[0] for v in HNX.values()]) / 100
PAR = np.array([np.nan if v[1] is None else v[1] for v in HNX.values()]) / 100


def test_parse_curve():
    page = ("<tr><td>3 th&#225;ng</td><td>1,5</td><td></td><td>1,51</td></tr>"
            "<tr><td>10 năm</td><td>3,0</td><td>3,1</td><td>3,05</td></tr>")
    rows = parse_curve(page, "d")
    assert rows[0][1] == 0.25 and np.isnan(rows[0][3])
    assert rows[1][1:] == (10.0, 0.03, 0.031, 0.0305)


def test_nss_recovers_known_curve():
    true = NSSCurve(np.array([0.05, -0.02, 0.01, 0.005]), (1.5, 8.0))
    fit = fit_nss(T, true.zero(T))
    assert np.max(np.abs(fit.zero(T) - true.zero(T))) < 1e-6


def test_nss_fits_hnx_within_2bp():
    fit = fit_nss(T, Z)
    assert np.max(np.abs(fit.zero(T) - Z)) < 2e-4


def test_bootstrap_matches_hnx_spot():
    # bootstrapped zeros at HNX par tenors should agree with HNX's own spot curve
    boot = bootstrap_par(T, PAR, short_zeros=np.expm1(Z))
    for t in (1, 2, 3, 5):
        assert boot.zero(t) == pytest.approx(Z[T == t][0], abs=3e-4)


def test_par_bond_prices_at_par():
    curve = InterpCurve(T, Z)
    c = curve.par_yield(5)
    assert Bond(c, 5).dirty_price(curve) == pytest.approx(100, abs=1e-9)


def test_flat_curve_analytics():
    y = 0.05
    curve = InterpCurve([0.1, 30], [np.log1p(y)] * 2)   # flat 5% annual
    b = Bond(0.05, 10)
    assert b.dirty_price(curve) == pytest.approx(100, abs=1e-9)
    assert b.ytm(100) == pytest.approx(y, abs=1e-10)
    # effective (continuous-shift) duration = Macaulay; modified = Macaulay / (1+y)
    dur, _ = b.effective_duration(curve)
    assert dur == pytest.approx(b.macaulay_duration(y), rel=1e-4)
    assert b.modified_duration(y) == pytest.approx(b.macaulay_duration(y) / 1.05)
    # zero-coupon bond: Macaulay duration = maturity
    assert Bond(0.0, 7).macaulay_duration(y) == pytest.approx(7)


def test_key_rates_sum_to_parallel_dv01():
    curve = InterpCurve(T, Z)
    b = Bond(0.045, 8.4)
    assert sum(b.key_rate_dv01(curve).values()) == pytest.approx(b.dv01(curve), rel=1e-6)


def test_scenario_signs():
    curve = InterpCurve(T, Z)
    _, _, pnl = portfolio_report([("10Y", Bond(0.045, 10), 1e9)], curve, scenario_shocks())
    assert pnl.loc["Parallel +100bp", "Portfolio"] < 0 < pnl.loc["Parallel -100bp", "Portfolio"]
