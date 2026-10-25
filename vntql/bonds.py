"""Fixed-rate bond pricing and interest-rate risk (annual coupons, Vietnam Government Bond style)."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

BP = 1e-4
KEY_TENORS = (0.25, 1, 2, 3, 5, 7, 10, 15, 20)


@dataclass
class Bond:
    coupon: float        # annual coupon rate, decimal
    maturity: float      # years to maturity from valuation date
    face: float = 100.0
    freq: int = 1

    def cashflows(self):
        """(times, amounts) of remaining cash flows."""
        step = 1 / self.freq
        times = np.arange(self.maturity, 0, -step)[::-1]
        amounts = np.full(times.shape, self.face * self.coupon / self.freq)
        amounts[-1] += self.face
        return times, amounts

    def accrued(self):
        times, _ = self.cashflows()
        elapsed = 1 / self.freq - times[0]
        return self.face * self.coupon * elapsed

    # ----- curve-based -----
    def dirty_price(self, curve):
        t, cf = self.cashflows()
        return float(cf @ curve.df(t))

    def clean_price(self, curve):
        return self.dirty_price(curve) - self.accrued()

    def dv01(self, curve):
        """Price change for a -1bp parallel shift of the zero curve (positive for long bonds)."""
        down = curve.shifted(lambda t: -BP * np.ones_like(t))
        up = curve.shifted(lambda t: BP * np.ones_like(t))
        return (self.dirty_price(down) - self.dirty_price(up)) / 2

    def effective_duration(self, curve, bump=10 * BP):
        p0 = self.dirty_price(curve)
        pu = self.dirty_price(curve.shifted(lambda t: bump * np.ones_like(t)))
        pd_ = self.dirty_price(curve.shifted(lambda t: -bump * np.ones_like(t)))
        return (pd_ - pu) / (2 * p0 * bump), (pd_ + pu - 2 * p0) / (p0 * bump ** 2)

    def key_rate_dv01(self, curve, keys=KEY_TENORS):
        """DV01 to a 1bp triangular bump at each key tenor; sums to the parallel DV01."""
        keys = np.asarray(keys, float)
        out = {}
        for i, k in enumerate(keys):
            bump = _triangle(keys, i)
            up = curve.shifted(lambda t, b=bump: BP * b(t))
            down = curve.shifted(lambda t, b=bump: -BP * b(t))
            out[k] = (self.dirty_price(down) - self.dirty_price(up)) / 2
        return out

    # ----- yield-based -----
    def price_from_yield(self, y):
        t, cf = self.cashflows()
        return float(cf @ (1 + y / self.freq) ** (-t * self.freq))

    def ytm(self, dirty_price):
        return brentq(lambda y: self.price_from_yield(y) - dirty_price, -0.5, 1.0)

    def macaulay_duration(self, y):
        t, cf = self.cashflows()
        pv = cf * (1 + y / self.freq) ** (-t * self.freq)
        return float(t @ pv / pv.sum())

    def modified_duration(self, y):
        return self.macaulay_duration(y) / (1 + y / self.freq)

    def convexity(self, y):
        t, cf = self.cashflows()
        f = self.freq
        pv = cf * (1 + y / f) ** (-t * f)
        return float(((t * f) * (t * f + 1) * pv).sum() / (pv.sum() * f ** 2 * (1 + y / f) ** 2))


def _triangle(keys, i):
    """Hat function equal to 1 at keys[i], 0 at neighbours, flat beyond the ends."""
    vals = np.zeros(len(keys))
    vals[i] = 1.0
    return lambda t: np.interp(np.asarray(t, float), keys, vals)


# ---------- portfolio & scenarios ----------

def illustrative_portfolio(curve, total=100e9):
    """VND 100bn face across 2Y/5Y/10Y/15Y bonds with coupons at the curve's par yields."""
    return [(f"{m}Y", Bond(round(curve.par_yield(m), 4), m), w * total)
            for m, w in ((2, 0.20), (5, 0.30), (10, 0.35), (15, 0.15))]


def scenario_shocks():
    """Standard rate shocks (decimal shifts as functions of tenor)."""
    return {
        "Parallel +100bp": lambda t: 100 * BP * np.ones_like(t),
        "Parallel -100bp": lambda t: -100 * BP * np.ones_like(t),
        "Bear steepener": lambda t: BP * np.interp(t, [0, 2, 20], [0, 25, 100]),
        "Bull flattener": lambda t: -BP * np.interp(t, [0, 2, 20], [0, 25, 100]),
        "Short-end +100bp": lambda t: BP * np.interp(t, [0, 2, 5], [100, 100, 0]),
        "Twist (short +50, long -50)": lambda t: BP * np.interp(t, [0, 5, 20], [50, 0, -50]),
    }


def portfolio_report(positions, curve, shocks=None):
    """positions: list of (name, Bond, face_amount). Returns per-bond risk and scenario P&L."""
    import pandas as pd

    rows = []
    for name, b, amount in positions:
        units = amount / b.face
        p = b.dirty_price(curve)
        y = b.ytm(p)
        dur, cvx = b.effective_duration(curve)
        rows.append({
            "bond": name, "coupon": b.coupon, "maturity": b.maturity, "face": amount,
            "dirty_price": p, "ytm": y, "market_value": p * units,
            "mod_duration": b.modified_duration(y), "eff_duration": dur, "convexity": cvx,
            "dv01": b.dv01(curve) * units,
        })
    risk = pd.DataFrame(rows).set_index("bond")

    krd = pd.DataFrame({name: {k: v * amount / b.face for k, v in b.key_rate_dv01(curve).items()}
                        for name, b, amount in positions}).T

    pnl = {}
    for sname, shock in (shocks or scenario_shocks()).items():
        shocked = curve.shifted(shock)
        pnl[sname] = {name: (b.dirty_price(shocked) - b.dirty_price(curve)) * amount / b.face
                      for name, b, amount in positions}
    pnl = pd.DataFrame(pnl).T
    pnl["Portfolio"] = pnl.sum(axis=1)
    return risk, krd, pnl
