"""Out-of-sample forecasting toolkit: expanding-window walk-forward, linear / ridge models, forecast evaluation.

Alignment convention: row t of X holds information available at the end of period t; y[t] is the
change from t to t+h. At forecast origin t only targets y[i] with i <= t - h are realised, so models are
fit on those rows and applied to X[t].
"""
import numpy as np
import pandas as pd
from scipy import stats


def robust_month_end(daily, window=5):
    """Month-end level as the median of the last `window` daily closes of each month.

    Uses only that month's data (no look-ahead) and removes the one-day off-market prints that
    Yahoo's USD/VND history contains, especially before 2021.
    """
    s = daily.dropna()
    return s.groupby(s.index.to_period("M")).apply(lambda x: x.iloc[-window:].median()).to_timestamp("M")


# ---------- models: each takes (X_train, y_train) and returns predict(X) ----------

def fit_mean(X, y):
    m = float(np.mean(y))
    return lambda Xn: np.full(len(Xn), m)


def fit_zero(X, y):
    return lambda Xn: np.zeros(len(Xn))


def fit_ridge(X, y, alpha=0.0):
    """Ridge on standardised features with an unpenalised intercept; alpha = 0 is OLS."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + alpha * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    b0 = y.mean()
    return lambda Xn: b0 + ((np.asarray(Xn, float) - mu) / sd) @ beta


def fit_ols(X, y):
    return fit_ridge(X, y, 0.0)


def fit_ridge_cv(X, y, alphas=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0), h=1, val=24):
    """Pick alpha by rolling-origin validation on the last `val` rows of the training set, then refit."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    if n < 2 * val:
        return fit_ridge(X, y, alphas[-1])
    err = {}
    for a in alphas:
        e = [fit_ridge(X[:t - h + 1], y[:t - h + 1], a)(X[t:t + 1])[0] - y[t] for t in range(n - val, n)]
        err[a] = np.mean(np.square(e))
    return fit_ridge(X, y, min(err, key=err.get))


def ols_nw(X, y, lag=0):
    """In-sample OLS with intercept and Newey-West t-stats. Returns (table, R^2, n)."""
    df = pd.concat([X, y.rename("_y")], axis=1).dropna()
    A = np.column_stack([np.ones(len(df)), df[X.columns].to_numpy(float)])
    yy = df["_y"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    e = yy - A @ beta
    Ainv = np.linalg.inv(A.T @ A)
    S = (A * e[:, None]).T @ (A * e[:, None])
    for k in range(1, lag + 1):
        G = (A[k:] * e[k:, None]).T @ (A[:-k] * e[:-k, None])
        S += (1 - k / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(Ainv @ S @ Ainv))
    r2 = 1 - e @ e / ((yy - yy.mean()) @ (yy - yy.mean()))
    tab = pd.DataFrame({"coef": beta, "t (NW)": beta / se}, index=["const", *X.columns])
    return tab, r2, len(df)


# ---------- walk-forward ----------

def walk_forward(X, y, fit, min_train=48, h=1):
    """Expanding-window one-step forecasts. X: DataFrame, y: Series on the same index (NaN where unrealised)."""
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    out = pd.Series(np.nan, index=X.index)
    for t in range(min_train + h - 1, len(X)):
        last = t - h                                   # last realised target at origin t
        model = fit(Xv[:last + 1], yv[:last + 1])
        out.iloc[t] = model(Xv[t:t + 1])[0]
    return out


# ---------- evaluation ----------

def newey_west_tstat(x, lag):
    """t-statistic of the mean of x with Newey-West (Bartlett) standard errors."""
    x = np.asarray(x, float)
    n = len(x)
    u = x - x.mean()
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def clark_west(y, f_bench, f_model, h=1):
    """Clark-West (2007) test that a larger model beats a nested benchmark. Returns (t-stat, one-sided p)."""
    y, fb, fm = map(lambda a: np.asarray(a, float), (y, f_bench, f_model))
    adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    if np.allclose(adj, adj[0]):
        return np.nan, np.nan
    t = newey_west_tstat(adj, max(h - 1, 0))
    return t, 1 - stats.norm.cdf(t)


def evaluate(y, forecasts, bench="Drift", h=1):
    """RMSE, ratio and OOS R^2 vs the benchmark, direction hit rate, Clark-West vs the benchmark."""
    df = pd.DataFrame(forecasts).assign(y=y).dropna()
    yb = df[bench]
    rows = {}
    for name in forecasts:
        e = df["y"] - df[name]
        rmse = np.sqrt((e ** 2).mean())
        r = {"RMSE": rmse,
             f"RMSE / {bench}": rmse / np.sqrt(((df["y"] - yb) ** 2).mean()),
             f"OOS R2 vs {bench} %": (1 - (e ** 2).sum() / ((df["y"] - yb) ** 2).sum()) * 100,
             "hit rate %": (np.sign(df[name]) == np.sign(df["y"])).mean() * 100 if df[name].abs().sum() else np.nan}
        if name != bench:
            r["CW t"], r["CW p"] = clark_west(df["y"], yb, df[name], h)
        rows[name] = r
    out = pd.DataFrame(rows).T
    out.attrs["n"] = len(df)
    out.attrs["period"] = (df.index.min(), df.index.max())
    return out
