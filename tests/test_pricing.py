import numpy as np
import pytest

from vntql.bonds import Bond
from vntql.curves import InterpCurve, bootstrap_par
from vntql.fx import FXForward, cip_forward, cip_forward_mm, implied_dom_rate
from vntql.options import gk_greeks, gk_price, implied_vol, strike_from_delta
from vntql.swaps import Swap


def flat(r):
    return InterpCurve([0.01, 50], [r, r])


VND = InterpCurve([0.25, 1, 5, 10], [0.037, 0.038, 0.042, 0.044])
USD = InterpCurve([0.25, 1, 5, 10], [0.042, 0.044, 0.048, 0.050])


# ---------- bootstrap with semiannual coupons ----------

def test_semiannual_bootstrap_flat():
    y = 0.05
    t = np.array([0.25, 0.5, 1, 2, 5, 10])
    c = bootstrap_par(t, np.full(6, y), short_zeros=np.full(6, y), freq=2, short_freq=2, cutoff=1 + 1e-9)
    assert np.allclose(c.zero([0.5, 1, 2, 5, 10]), 2 * np.log1p(y / 2), atol=1e-12)


# ---------- FX forwards ----------

def test_cip_flat_rates():
    F = cip_forward(25000, 1.0, r_dom=0.04, r_for=0.05)
    assert F == pytest.approx(25000 * np.exp(-0.01))
    assert implied_dom_rate(25000, F, 1.0, 0.05) == pytest.approx(0.04)


def test_cip_mm_convention():
    assert cip_forward_mm(25000, 360, 0.04, 0.05) == pytest.approx(25000 * (1 + 0.04 * 360 / 365) / 1.05)


def test_forward_at_cip_strike_has_zero_value():
    S, T = 25900, 0.75
    K = cip_forward(S, T, VND, USD)
    fwd = FXForward(1e6, K, T)
    assert fwd.value(S, VND, USD) == pytest.approx(0, abs=1e-6)
    r = fwd.risks(S, VND, USD)
    assert r["spot_delta"] == pytest.approx(1e6 * USD.df(T), rel=1e-6)
    # long USD forward pays VND K at T: lower VND rates raise that liability's PV; lower USD rates raise the asset's
    assert r["dv01_dom"] < 0 < r["dv01_for"]


# ---------- Garman-Kohlhagen ----------

def test_gk_reduces_to_black_scholes():
    # rf = 0 -> Black-Scholes; textbook value for S=K=100, r=5%, vol=20%, T=1
    assert gk_price(100, 100, 1, 0.05, 0.0, 0.2) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    S, K, T, rd, rf, v = 25900, 26200, 0.5, 0.04, 0.045, 0.03
    c, p = gk_price(S, K, T, rd, rf, v), gk_price(S, K, T, rd, rf, v, call=False)
    assert c - p == pytest.approx(S * np.exp(-rf * T) - K * np.exp(-rd * T), rel=1e-10)


def test_greeks_match_finite_differences():
    S, K, T, rd, rf, v = 25900, 26200, 0.5, 0.04, 0.045, 0.03
    g = gk_greeks(S, K, T, rd, rf, v)
    h = 1.0
    assert g["delta"] == pytest.approx((gk_price(S + h, K, T, rd, rf, v) - gk_price(S - h, K, T, rd, rf, v)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((gk_price(S + h, K, T, rd, rf, v) - 2 * g["price"] + gk_price(S - h, K, T, rd, rf, v)) / h ** 2, rel=1e-3)
    assert g["vega"] == pytest.approx((gk_price(S, K, T, rd, rf, v + 1e-4) - gk_price(S, K, T, rd, rf, v - 1e-4)) / 2e-4 / 100, rel=1e-5)
    assert g["rho_dom"] == pytest.approx((gk_price(S, K, T, rd + 1e-6, rf, v) - gk_price(S, K, T, rd - 1e-6, rf, v)) / 2e-6 * 1e-4, rel=1e-4)
    assert g["rho_for"] == pytest.approx((gk_price(S, K, T, rd, rf + 1e-6, v) - gk_price(S, K, T, rd, rf - 1e-6, v)) / 2e-6 * 1e-4, rel=1e-4)
    dt = 1 / 365
    assert g["theta"] == pytest.approx(gk_price(S, K, T - dt, rd, rf, v) - g["price"], rel=2e-2)


def test_implied_vol_and_delta_strike_roundtrip():
    S, T, rd, rf, v = 25900, 0.5, 0.04, 0.045, 0.035
    K = strike_from_delta(0.25, S, T, rd, rf, v)
    assert gk_greeks(S, K, T, rd, rf, v)["delta"] == pytest.approx(0.25, abs=1e-10)
    assert implied_vol(gk_price(S, K, T, rd, rf, v), S, K, T, rd, rf) == pytest.approx(v, abs=1e-10)


# ---------- swaps ----------

def test_par_swap_has_zero_value_and_matches_par_bond():
    s = Swap(1e9, 0.0, 5)
    rate = s.par_rate(VND)
    assert Swap(1e9, rate, 5).value(VND) == pytest.approx(0, abs=1e-3)
    # single curve, annual fixed leg: par swap rate == par yield of an annual bond
    assert rate == pytest.approx(VND.par_yield(5), abs=1e-12)
    assert Bond(rate, 5).dirty_price(VND) == pytest.approx(100, abs=1e-9)


def test_float_leg_telescopes():
    s = Swap(1.0, 0.03, 7, start=2)
    assert s.float_leg(VND) == pytest.approx(VND.df(2) - VND.df(7), abs=1e-12)


def test_swap_dv01_signs_and_key_rates():
    payer = Swap(1e9, 0.04, 5)
    receiver = Swap(1e9, 0.04, 5, pay_fixed=False)
    assert payer.dv01(VND) > 0 > receiver.dv01(VND)
    assert sum(payer.key_rate_dv01(VND).values()) == pytest.approx(payer.dv01(VND), rel=1e-6)
