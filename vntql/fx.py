"""FX forwards by covered interest parity (CIP). Quote convention: domestic per 1 foreign (VND per USD)."""
from dataclasses import dataclass

import numpy as np


def cip_forward(spot, T, dom_curve=None, for_curve=None, r_dom=None, r_for=None):
    """Outright forward F = S * DF_for(T) / DF_dom(T).

    Pass curves (continuous zeros) or flat continuous rates r_dom / r_for.
    """
    df_d = dom_curve.df(T) if dom_curve is not None else np.exp(-r_dom * T)
    df_f = for_curve.df(T) if for_curve is not None else np.exp(-r_for * T)
    return spot * df_f / df_d


def cip_forward_mm(spot, days, r_dom, r_for, basis_dom=365, basis_for=360):
    """Money-market convention: F = S * (1 + r_d * d/B_d) / (1 + r_f * d/B_f) (VND ACT/365, USD ACT/360)."""
    return spot * (1 + r_dom * days / basis_dom) / (1 + r_for * days / basis_for)


def implied_dom_rate(spot, fwd, T, r_for):
    """Continuous domestic rate implied by a quoted forward (used to measure CIP basis)."""
    return r_for + np.log(fwd / spot) / T


@dataclass
class FXForward:
    """Buy `notional` units of foreign currency at `strike` (domestic per foreign) at time `maturity`."""
    notional: float
    strike: float
    maturity: float
    buy_foreign: bool = True

    def value(self, spot, dom_curve, for_curve, t=0.0):
        """Mark-to-market in domestic currency at valuation time t (years from trade)."""
        tau = self.maturity - t
        if tau <= 0:
            pv = self.notional * (spot - self.strike)
        else:
            pv = self.notional * (spot * for_curve.df(tau) - self.strike * dom_curve.df(tau))
        return float(pv if self.buy_foreign else -pv)

    def risks(self, spot, dom_curve, for_curve, t=0.0, bp=1e-4):
        """Spot delta (domestic per 1 unit spot), DV01 to domestic and foreign curves (domestic ccy)."""
        def up(c):
            return c.shifted(lambda x: bp * np.ones_like(x))

        def down(c):
            return c.shifted(lambda x: -bp * np.ones_like(x))
        v = lambda s=spot, d=dom_curve, f=for_curve: self.value(s, d, f, t)  # noqa: E731
        return {
            "spot_delta": (v(s=spot * 1.0001) - v(s=spot * 0.9999)) / (spot * 0.0002),
            "dv01_dom": (v(d=down(dom_curve)) - v(d=up(dom_curve))) / 2,
            "dv01_for": (v(f=down(for_curve)) - v(f=up(for_curve))) / 2,
        }
