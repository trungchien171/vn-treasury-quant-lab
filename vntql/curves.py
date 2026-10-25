"""Yield curve construction: Nelson-Siegel(-Svensson) fitting and par-yield bootstrapping.

Conventions: rates are decimals, zero rates are continuously compounded, tenors in years,
coupon bonds pay annually (Vietnam Government Bonds).
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares


# ---------- curve object ----------

class Curve:
    """Base curve: subclasses implement zero(t) (continuous compounding)."""

    def zero(self, t):
        raise NotImplementedError

    def df(self, t):
        t = np.asarray(t, float)
        return np.exp(-self.zero(t) * t)

    def forward(self, t1, t2):
        """Continuously compounded forward rate between t1 and t2."""
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def par_yield(self, n):
        """Par coupon of an n-year annual-pay bond."""
        times = np.arange(1, int(n) + 1)
        d = self.df(times)
        return (1 - d[-1]) / d.sum()

    def shifted(self, shock):
        """New curve with zero(t) + shock(t); shock is a callable of t in decimals."""
        return ShiftedCurve(self, shock)


class InterpCurve(Curve):
    """Linear interpolation on continuous zero rates, flat extrapolation."""

    def __init__(self, tenors, zeros):
        order = np.argsort(tenors)
        self.tenors = np.asarray(tenors, float)[order]
        self.zeros = np.asarray(zeros, float)[order]

    def zero(self, t):
        return np.interp(np.asarray(t, float), self.tenors, self.zeros)


class ShiftedCurve(Curve):
    def __init__(self, base, shock):
        self.base, self.shock = base, shock

    def zero(self, t):
        t = np.asarray(t, float)
        return self.base.zero(t) + self.shock(t)


# ---------- Nelson-Siegel-Svensson ----------

def _f1(x):
    x = np.maximum(x, 1e-10)
    return (1 - np.exp(-x)) / x


def nss_basis(t, l1, l2=None):
    t = np.asarray(t, float)
    cols = [np.ones_like(t), _f1(t / l1), _f1(t / l1) - np.exp(-t / l1)]
    if l2 is not None:
        cols.append(_f1(t / l2) - np.exp(-t / l2))
    return np.column_stack(cols)


@dataclass
class NSSCurve(Curve):
    """beta = (b0, b1, b2[, b3]); lambdas = (l1[, l2]). Plain Nelson-Siegel when len(lambdas) == 1."""
    beta: np.ndarray
    lambdas: tuple

    def zero(self, t):
        return nss_basis(t, *self.lambdas) @ self.beta


def fit_nss(tenors, zeros, svensson=True, weights=None):
    """Fit NS / NSS to zero rates.

    Grid search over decay parameters with betas solved by OLS (the model is linear
    given lambdas), then joint nonlinear refinement with scipy least_squares.
    """
    t, y = np.asarray(tenors, float), np.asarray(zeros, float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, float)
    grid = np.geomspace(0.2, 15, 16)
    pairs = [(a, b) for a, b in product(grid, grid) if b > a * 1.5] if svensson else [(a,) for a in grid]

    def ols(lams):
        X = nss_basis(t, *lams) * w[:, None]
        beta, *_ = np.linalg.lstsq(X, y * w, rcond=None)
        return beta, np.sum((X @ beta - y * w) ** 2)

    best = min(pairs, key=lambda lams: ols(lams)[1])
    x0 = np.r_[ols(best)[0], np.log(best)]
    k = len(best)

    def resid(x):
        return (nss_basis(t, *np.exp(np.clip(x[-k:], -5, 5))) @ x[:-k] - y) * w

    sol = least_squares(resid, x0, method="lm" if len(t) >= len(x0) else "trf")
    return NSSCurve(sol.x[:-k], tuple(np.exp(np.clip(sol.x[-k:], -5, 5))))


# ---------- bootstrapping ----------

def bootstrap_par(tenors, par, short_zeros=None, freq=1, short_freq=1, cutoff=1.0):
    """Bootstrap continuous zero rates from par yields paying `freq` coupons a year.

    Tenors below `cutoff` are zero-coupon instruments: their rates come from `short_zeros`
    (compounded `short_freq` times a year). Par yields are linearly interpolated on the
    coupon grid, then DF_n = (1 - c_n/f * sum_{i<n} DF_i) / (1 + c_n/f).
    Returns an InterpCurve on the short-end tenors plus the coupon grid from `cutoff` on.
    """
    t, c = np.asarray(tenors, float), np.asarray(par, float)
    short_t, short_z = np.array([]), np.array([])
    if short_zeros is not None:
        sz = np.asarray(short_zeros, float)
        s = (t < cutoff) & ~np.isnan(sz)
        short_t, short_z = t[s], short_freq * np.log1p(sz[s] / short_freq)
    long_ = (t >= cutoff) & ~np.isnan(c)
    grid = np.arange(1, int(round(t[long_].max() * freq)) + 1) / freq
    c_n = np.interp(grid, t[long_], c[long_])
    dfs = []
    for g, cn in zip(grid, c_n):
        if g < cutoff and len(short_t):
            dfs.append(np.exp(-np.interp(g, short_t, short_z) * g))
        else:
            dfs.append((1 - cn / freq * sum(dfs)) / (1 + cn / freq))
    keep = grid >= cutoff
    zeros = -np.log(np.array(dfs)[keep]) / grid[keep]
    return InterpCurve(np.r_[short_t, grid[keep]], np.r_[short_z, zeros])
