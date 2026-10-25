"""Garman-Kohlhagen FX options (European), greeks and implied volatility.

S: spot (domestic per foreign), K: strike, T: years, rd/rf: continuous domestic/foreign rates,
vol: annualised. Prices are in domestic currency per 1 unit of foreign notional.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, rd, rf, vol):
    d1 = (np.log(S / K) + (rd - rf + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    return d1, d1 - vol * np.sqrt(T)


def gk_price(S, K, T, rd, rf, vol, call=True):
    d1, d2 = _d1d2(S, K, T, rd, rf, vol)
    if call:
        return S * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)
    return K * np.exp(-rd * T) * norm.cdf(-d2) - S * np.exp(-rf * T) * norm.cdf(-d1)


def gk_greeks(S, K, T, rd, rf, vol, call=True):
    """Analytic greeks. vega per 1 vol point (0.01), theta per calendar day, rho per 1bp."""
    d1, d2 = _d1d2(S, K, T, rd, rf, vol)
    dfd, dff, sq = np.exp(-rd * T), np.exp(-rf * T), np.sqrt(T)
    sign = 1 if call else -1
    delta = sign * dff * norm.cdf(sign * d1)
    gamma = dff * norm.pdf(d1) / (S * vol * sq)
    vega = S * dff * norm.pdf(d1) * sq
    theta = (-S * dff * norm.pdf(d1) * vol / (2 * sq)
             + sign * (rf * S * dff * norm.cdf(sign * d1) - rd * K * dfd * norm.cdf(sign * d2)))
    rho_d = sign * K * T * dfd * norm.cdf(sign * d2)
    rho_f = -sign * S * T * dff * norm.cdf(sign * d1)
    return {"price": gk_price(S, K, T, rd, rf, vol, call), "delta": delta, "gamma": gamma,
            "vega": vega / 100, "theta": theta / 365, "rho_dom": rho_d * 1e-4, "rho_for": rho_f * 1e-4}


def implied_vol(price, S, K, T, rd, rf, call=True, lo=1e-4, hi=3.0):
    return brentq(lambda v: gk_price(S, K, T, rd, rf, v, call) - price, lo, hi)


def strike_from_delta(delta, S, T, rd, rf, vol, call=True):
    """Strike for a given spot delta (premium-unadjusted), e.g. 25-delta call -> delta=0.25."""
    sign = 1 if call else -1
    d1 = sign * norm.ppf(sign * delta * np.exp(rf * T))
    return S * np.exp(-d1 * vol * np.sqrt(T) + (rd - rf + 0.5 * vol ** 2) * T)
