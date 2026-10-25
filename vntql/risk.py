"""VaR models and backtests: rolling historical simulation, EWMA, Kupiec / Christoffersen tests, Basel traffic light."""
import numpy as np
import pandas as pd
from scipy import stats


def hs_var(pnl, level=0.99, window=250):
    """Rolling historical-simulation VaR for day t from the previous `window` P&Ls (positive number = loss)."""
    return -pnl.shift(1).rolling(window).quantile(1 - level)


def ewma_var(pnl, level=0.99, lam=0.94, warmup=250):
    """RiskMetrics: normal VaR with EWMA volatility, using P&L up to t-1 only."""
    var = pnl.pow(2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    out = stats.norm.ppf(level) * np.sqrt(var)
    out.iloc[:warmup] = np.nan
    return out


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(exceptions, p):
    """Unconditional coverage (proportion of failures) test. Returns (LR, p-value)."""
    n, x = len(exceptions), int(np.sum(exceptions))
    pi = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(exceptions):
    """Independence test: does an exception today make one tomorrow more likely? Returns (LR, p-value)."""
    e = np.asarray(exceptions, int)
    prev, cur = e[:-1], e[1:]
    n00, n01 = np.sum((prev == 0) & (cur == 0)), np.sum((prev == 0) & (cur == 1))
    n10, n11 = np.sum((prev == 1) & (cur == 0)), np.sum((prev == 1) & (cur == 1))
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / len(cur)
    restricted = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    unrestricted = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr = -2 * (restricted - unrestricted)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def traffic_light(n_exceptions):
    """Basel zone for 99% VaR over the last 250 days."""
    return "green" if n_exceptions <= 4 else "yellow" if n_exceptions <= 9 else "red"


def backtest(pnl, var, level):
    """Exception count, rate, Kupiec, Christoffersen and the joint conditional-coverage test."""
    df = pd.DataFrame({"pnl": pnl, "var": var}).dropna()
    exc = (df["pnl"] < -df["var"]).to_numpy()
    lr_uc, p_uc = kupiec(exc, 1 - level)
    lr_ind, p_ind = christoffersen(exc)
    lr_cc = lr_uc + lr_ind
    return pd.Series({"days": len(df), "exceptions": int(exc.sum()), "expected": len(df) * (1 - level),
                      "rate %": exc.mean() * 100, "Kupiec p": p_uc, "Christoffersen p": p_ind,
                      "joint p": 1 - stats.chi2.cdf(lr_cc, 2)}), pd.Series(exc, index=df.index)
