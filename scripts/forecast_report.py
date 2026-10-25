"""Forecasting USD/VND and VND short rates with market and macro drivers: walk-forward out-of-sample tests.

-> reports/forecast.md + charts. Run scripts/fetch_hnx.py, fetch_market.py and fetch_macro.py first.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vntql.data import load, wide  # noqa: E402
from vntql.forecast import (evaluate, fit_mean, fit_ols, fit_ridge_cv, fit_zero, ols_nw,  # noqa: E402
                            robust_month_end, walk_forward)
from vntql.macro import load_drivers  # noqa: E402
from vntql.market import load_ust  # noqa: E402

OUT = ROOT / "reports"
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
MACRO_LAG = 2      # months between a CPI / trade print's reference month and when it is safely public
START = "2012-01"  # after the 2011 devaluations; Yahoo USD/VND before that is too sparse and noisy


def monthly_panel():
    daily, macro = load_drivers()
    m = pd.DataFrame({k: robust_month_end(daily[k]) for k in daily})
    ust = load_ust()
    m["ust2y"] = ust[2.0].resample("ME").last() * 1e4                 # bp
    vgb = wide(load()) * 1e4                                          # HNX continuous spot, bp
    m["vgb3m"], m["vgb1y"] = robust_month_end(vgb[0.25]), robust_month_end(vgb[1.0])
    m = m.join(macro, how="left")

    f = pd.DataFrame(index=m.index)
    lx = 100 * np.log(m["usdvnd"])
    f["mom1"], f["mom3"] = lx.diff(), lx.diff(3)
    for k in ("dxy", "usdcny", "brent"):
        f[f"{k}1"] = 100 * np.log(m[k]).diff()
    f["d_ust2y"] = m["ust2y"].diff()
    infl = lambda c: 100 * np.log(c).diff(12)  # noqa: E731
    f["infl_diff"] = (infl(m["cpi_vn"]) - infl(m["cpi_us"])).shift(MACRO_LAG)
    tb = (m["exports_vn"] - m["imports_vn"]).rolling(12).sum() / (m["exports_vn"] + m["imports_vn"]).rolling(12).sum()
    f["trade_bal"] = 100 * tb.shift(MACRO_LAG)
    f["vn_infl"] = infl(m["cpi_vn"]).shift(MACRO_LAG)
    f["d_vgb1y"], f["d_vgb3m"] = m["vgb1y"].diff(), m["vgb3m"].diff()
    f["fx3"] = f["mom3"]
    y = {h: lx.shift(-h) - lx for h in (1, 3)}
    y["r1"] = m["vgb1y"].shift(-1) - m["vgb1y"]
    return m, f, y


def run_models(f, y, specs, h, min_train, bench):
    preds = {name: walk_forward(f[cols] if cols else f[["mom1"]], y, fit, min_train, h)
             for name, (cols, fit) in specs.items()}
    return evaluate(y, preds, bench=bench, h=h), preds


def fx_forecasts(f, y):
    f = f.loc[START:]
    ridge = lambda h: (lambda X, yy: fit_ridge_cv(X, yy, h=h))  # noqa: E731
    market = ["mom1", "dxy1", "usdcny1", "d_ust2y"]
    macro = ["infl_diff", "trade_bal"]
    out = {}
    for h in (1, 3):
        mom = "mom1" if h == 1 else "mom3"
        specs = {"RW": (None, fit_zero), "Drift": (None, fit_mean), "AR (momentum)": ([mom], fit_ols),
                 "Market": ([*market, mom] if h == 3 else market, fit_ols), "Macro": (macro, fit_ols),
                 "Ridge (all)": ([*market, "brent1", *macro], ridge(h))}
        cols = sorted({c for s in specs.values() if s[0] for c in s[0]})
        ff = f[cols].dropna()
        out[h] = run_models(ff, y[h].reindex(ff.index), specs, h, 48, "Drift")
    return out


def rate_forecasts(f, y):
    cols = ["d_vgb1y", "mom1", "fx3", "d_ust2y", "vn_infl"]
    ff = f[cols].dropna()
    specs = {"RW": (None, fit_zero), "Drift": (None, fit_mean), "AR": (["d_vgb1y"], fit_ols),
             "FX pressure": (["d_vgb1y", "fx3"], fit_ols), "FX + US + CPI": (cols, fit_ols)}
    return run_models(ff, y["r1"].reindex(ff.index), specs, 1, 36, "RW")


def cum_sse_chart(y, preds, bench, path, title):
    df = pd.DataFrame(preds).assign(y=y).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in preds:
        if name in (bench, "RW"):
            continue
        ax.plot(df.index, ((df["y"] - df[bench]) ** 2 - (df["y"] - df[name]) ** 2).cumsum(), label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title=title, ylabel=f"cumulative SSE({bench}) - SSE(model)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def rolling_beta_chart(f, path, window=36):
    df = f[["mom1", "dxy1", "usdcny1"]].loc[START:].dropna()
    betas = {}
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window:i]
        A = np.column_stack([np.ones(window), w[["dxy1", "usdcny1"]]])
        betas[w.index[-1]] = np.linalg.lstsq(A, w["mom1"], rcond=None)[0][1:]
    b = pd.DataFrame(betas, index=["DXY", "USD/CNY"]).T
    fig, ax = plt.subplots(figsize=(8, 3.5))
    b.plot(ax=ax)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title=f"USD/VND monthly beta to DXY and USD/CNY (rolling {window}M, same-month moves)", ylabel="beta")
    fig.tight_layout()
    fig.savefig(path)
    return b


def fmt(df):
    return df.round(3).to_markdown()


def main():
    m, f, y = monthly_panel()
    fx = fx_forecasts(f, y)
    rates, rate_preds = rate_forecasts(f, y)

    contemp, r2, n = ols_nw(f.loc[START:, ["dxy1", "usdcny1", "d_ust2y"]], f.loc[START:, "mom1"], lag=3)
    betas = rolling_beta_chart(f, OUT / "fx_beta.png")
    cum_sse_chart(y[1], fx[1][1], "Drift", OUT / "fx_oos.png", "USD/VND 1M forecasts vs drift (above 0 = model ahead)")
    cum_sse_chart(y["r1"], rate_preds, "RW", OUT / "rate_oos.png", "VGB 1Y 1M-ahead change vs random walk")

    lx = 100 * np.log(m["usdvnd"].loc[START:])
    drift_pa = lx.diff().mean() * 12
    fx_lead, _, n_lead = ols_nw(f[["d_vgb1y", "fx3"]].loc["2020":], y["r1"].loc["2020":], lag=1)
    t1, t3 = fx[1][0], fx[3][0]
    md = [
        "# Forecasting USD/VND and VND Short Rates",
        f"Monthly data {START} to {m.index[-1]:%b %Y}. USD/VND month-end = median of the month's last 5 daily closes "
        "(Yahoo's history has one-day off-market prints). Macro data (IMF: CPI, goods trade) enter with a "
        f"{MACRO_LAG}-month publication lag. All forecasts are **out of sample**: expanding window, refit every month, "
        "using only data available at the forecast date.", "",
        "## 1. What moves USD/VND (same-month, in-sample)",
        f"Regression of the monthly USD/VND log change (%) on same-month DXY and USD/CNY changes (%) and the "
        f"change in the US 2Y yield (bp); Newey-West t-stats, n = {n}, R² = {r2:.2f}.", "",
        fmt(contemp), "",
        "![beta](fx_beta.png)", "",
        f"Latest 36M betas: DXY {betas['DXY'].iloc[-1]:.2f}, USD/CNY {betas['USD/CNY'].iloc[-1]:.2f}. "
        f"This is a description, not a forecast: it maps a USD or CNY scenario into a USD/VND scenario.", "",
        f"## 2. Forecasting USD/VND (1M and 3M ahead)",
        f"Benchmark: **drift** (the trailing mean change; USD/VND has risen {drift_pa:.1f}% a year on average since {START[:4]} "
        "under the SBV's managed crawl). The pure random walk (no change) is also shown. "
        "CW = Clark-West test of the model against the nested drift benchmark (one-sided p).", "",
        f"**1M ahead** ({t1.attrs['n']} out-of-sample months, {t1.attrs['period'][0]:%b %Y} to {t1.attrs['period'][1]:%b %Y}):", "",
        fmt(t1), "", "![oos](fx_oos.png)", "",
        f"**3M ahead** (overlapping, {t3.attrs['n']} months; CW uses Newey-West with 2 lags):", "", fmt(t3), "",
        "Models: AR = own past change; Market = past 1M changes in DXY, USD/CNY, US 2Y yield and USD/VND; "
        "Macro = VN-US inflation differential and 12M goods trade balance (% of trade); Ridge = all of these plus Brent, "
        "penalty chosen by rolling validation inside each training window.", "",
        "## 3. Forecasting VND short rates (VGB 1Y, 1M ahead)",
        "HNX government 1Y spot rate as a proxy for VND short-term funding (interbank data is not publicly "
        "downloadable). Benchmark: random walk. FX pressure = past 3M USD/VND change, testing the SBV channel "
        "of defending the currency by tightening VND liquidity.", "",
        f"Out of sample {rates.attrs['n']} months, {rates.attrs['period'][0]:%b %Y} to {rates.attrs['period'][1]:%b %Y}:", "",
        fmt(rates), "", "![rates](rate_oos.png)", "",
        f"In-sample check, 2020 onward (n = {n_lead}): next-month change in VGB 1Y (bp) on this month's change (bp) and the past 3M USD/VND change (%):", "",
        fmt(fx_lead), "",
        "## 4. What I take from this",
        f"None of the models beat the drift forecast for USD/VND. The best 1M model is {t1['RMSE / Drift'].drop('Drift').min():.3f} "
        f"times the drift's error, and the lowest Clark-West p-value across 1M and 3M is "
        f"{min(t1['CW p'].min(), t3['CW p'].min()):.2f}. The market model only did well around the late-2022 sell-off "
        "and lost that edge afterwards. With public data, monthly moves in a managed currency look unpredictable.", "",
        f"USD/VND moves with USD/CNY much more than with DXY. Over the last three years the beta to USD/CNY is about "
        f"{betas['USD/CNY'].iloc[-1]:.1f}, and DXY adds little once CNY is in the regression. For stress scenarios, "
        "a CNY shock is the more useful input.", "",
        f"Short rates are a different story. Monthly changes in the VGB 1Y rate have an autocorrelation of "
        f"{f['d_vgb1y'].loc['2020':].autocorr():.2f}, so a plain AR model cuts the forecast error by about "
        f"{(1 - rates.loc['AR', 'RMSE / RW']) * 100:.0f}% compared with assuming no change. Most of the gain comes from "
        "the 2023 easing cycle, and HNX's curve fitting may smooth rates on its own, so I would use this for risk "
        "forecasts rather than as a trading signal.", "",
        "Past USD/VND weakness does predict higher short rates when used alone, but the effect goes away once the "
        "rate's own trend is included, and it does not improve the out-of-sample forecasts.", "",
    ]
    (OUT / "forecast.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
