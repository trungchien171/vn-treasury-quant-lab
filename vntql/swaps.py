"""Fixed-for-floating interest rate swaps, single-curve valuation (projection = discounting curve)."""
from dataclasses import dataclass

import numpy as np

from .bonds import BP, KEY_TENORS, _triangle


@dataclass
class Swap:
    notional: float
    fixed_rate: float
    maturity: float          # years from today to final payment
    start: float = 0.0       # forward start (years); 0 = spot-starting
    fixed_freq: int = 1
    float_freq: int = 4
    pay_fixed: bool = True

    def _times(self, freq):
        n = int(round((self.maturity - self.start) * freq))
        return self.start + np.arange(1, n + 1) / freq

    def annuity(self, curve):
        t = self._times(self.fixed_freq)
        return float((curve.df(t) / self.fixed_freq).sum())

    def fixed_leg(self, curve):
        return self.notional * self.fixed_rate * self.annuity(curve)

    def float_leg(self, curve):
        """Sum of projected forwards x accrual x DF; telescopes to DF(start) - DF(T) on one curve."""
        t = self._times(self.float_freq)
        prev = np.r_[self.start, t[:-1]]
        tau = t - prev
        fwd = (curve.df(prev) / curve.df(t) - 1) / tau      # simple-compounded forward rates
        return self.notional * float((fwd * tau * curve.df(t)).sum())

    def par_rate(self, curve):
        return self.float_leg(curve) / (self.notional * self.annuity(curve))

    def value(self, curve):
        pv = self.float_leg(curve) - self.fixed_leg(curve)
        return pv if self.pay_fixed else -pv

    def dv01(self, curve):
        """Value change for +1bp parallel shift (payer swaps gain when rates rise)."""
        up = curve.shifted(lambda t: BP * np.ones_like(t))
        down = curve.shifted(lambda t: -BP * np.ones_like(t))
        return (self.value(up) - self.value(down)) / 2

    def key_rate_dv01(self, curve, keys=KEY_TENORS):
        keys = np.asarray(keys, float)
        out = {}
        for i, k in enumerate(keys):
            b = _triangle(keys, i)
            up = curve.shifted(lambda t, b=b: BP * b(t))
            down = curve.shifted(lambda t, b=b: -BP * b(t))
            out[k] = (self.value(up) - self.value(down)) / 2
        return out
