"""Build the curve & risk report from cached HNX data -> reports/summary.md + charts."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vntql.bonds import illustrative_portfolio, portfolio_report, scenario_shocks  # noqa: E402
from vntql.curves import InterpCurve, bootstrap_par, fit_nss  # noqa: E402
from vntql.data import BREAKS, changes, load, wide  # noqa: E402
from vntql.pca import curve_pca, pca_shock  # noqa: E402
from vntql.risk import backtest, ewma_var, hs_var, traffic_light  # noqa: E402

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def latest_curve(df):
    d = df["date"].max()
    day = df[df["date"] == d].sort_values("tenor")
    t, z, par = day["tenor"].to_numpy(), day["spot_cont"].to_numpy(), day["par"].to_numpy()
    ns, nss = fit_nss(t, z, svensson=False), fit_nss(t, z)
    boot = bootstrap_par(t, par, short_zeros=day["spot_annual"].to_numpy())

    grid = np.linspace(0.1, 20, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, z * 100, "ko", label="HNX spot (continuous)")
    ax.plot(grid, ns.zero(grid) * 100, label="Nelson-Siegel")
    ax.plot(grid, nss.zero(grid) * 100, label="Nelson-Siegel-Svensson")
    ax.plot(boot.tenors, boot.zeros * 100, "s", ms=3, alpha=0.6, label="Bootstrapped from par")
    ax.plot(grid[1:], nss.forward(grid[:-1], grid[1:]) * 100, "--", alpha=0.7, label="NSS instantaneous forward")
    ax.set(xlabel="Tenor (years)", ylabel="%", title=f"Vietnam Government Bond curve, {d:%d %b %Y}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "curve_latest.png")

    fit = pd.DataFrame({
        "HNX spot %": z * 100,
        "NS %": ns.zero(t) * 100, "NSS %": nss.zero(t) * 100,
        "Bootstrap %": boot.zero(t) * 100,
    }, index=pd.Index(t, name="tenor"))
    fit["NSS err bp"] = (fit["NSS %"] - fit["HNX spot %"]) * 100
    fit["Boot err bp"] = (fit["Bootstrap %"] - fit["HNX spot %"]) * 100
    return d, InterpCurve(t, z), nss, fit


def fit_quality(df):
    rows = []
    for d, day in df.groupby("date"):
        t, z = day["tenor"].to_numpy(), day["spot_cont"].to_numpy()
        if len(t) < 8:
            continue
        r = {"date": d}
        for name, sv in (("NS", False), ("NSS", True)):
            c = fit_nss(t, z, svensson=sv)
            r[name] = np.sqrt(np.mean((c.zero(t) - z) ** 2)) * 1e4
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


def history_chart(Zw):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ten in (1, 2, 5, 10, 20):
        if ten in Zw:
            a1.plot(Zw.index, Zw[ten] * 100, lw=1, label=f"{ten}Y")
    a1.set(ylabel="% (cont.)", title="HNX zero rates")
    a1.legend(fontsize=8, ncol=5)
    a2.plot(Zw.index, (Zw[10] - Zw[2]) * 1e4, color="C3", lw=1)
    a2.axhline(0, color="k", lw=0.5)
    a2.set(ylabel="bp", title="10Y - 2Y slope")
    fig.tight_layout()
    fig.savefig(OUT / "history.png")


def pca_chart(loadings, explained):
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in loadings:
        ax.plot(loadings.index, loadings[c], "o-", label=f"{c} ({explained[c]:.0%})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Tenor (years)", ylabel="Loading", title="PCA of 20-day zero-rate changes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pca.png")


def scenario_pnl(positions, curve, dZ):
    """Full-revaluation P&L of today's portfolio under each historical curve change (rows of dZ)."""
    tenors = dZ.columns.to_numpy(float)
    base = sum(b.dirty_price(curve) * amt / b.face for _, b, amt in positions)
    pnl = []
    for dz in dZ.to_numpy():
        shocked = curve.shifted(lambda t, dz=dz: np.interp(t, tenors, dz))
        pnl.append(sum(b.dirty_price(shocked) * amt / b.face for _, b, amt in positions) - base)
    return pd.Series(pnl, index=dZ.index)


def historical_var(positions, curve, Zw, horizon=10, levels=(0.95, 0.99)):
    """Full-revaluation historical-simulation VaR on overlapping `horizon`-day curve changes."""
    pnl = scenario_pnl(positions, curve, changes(Zw, horizon)).to_numpy()
    var = {f"VaR {lv:.0%}": -np.quantile(pnl, 1 - lv) for lv in levels}
    tail = pnl[pnl <= np.quantile(pnl, 1 - levels[-1])]
    var[f"ES {levels[-1]:.0%}"] = -tail.mean()
    return pd.Series(var), pnl


def main():
    df = load()
    Zw = wide(df).dropna()
    d, curve, nss, fit = latest_curve(df)
    history_chart(Zw)
    dZ = Zw.diff()
    noise = pd.Series({t: dZ[t].autocorr() for t in Zw.columns})
    b = Zw.index.searchsorted(pd.Timestamp(BREAKS[0]))
    brk = Zw[10.0].iloc[[b - 1, b]] * 100

    fq = fit_quality(df)
    explained, loadings, scores = curve_pca(Zw, horizon=20)
    explained_1d, _, _ = curve_pca(Zw, horizon=1)
    pca_chart(loadings, explained)

    # Illustrative portfolio: VND 100bn face, coupons set at today's par yields
    B = 1e9
    positions = illustrative_portfolio(curve, 100 * B)
    shocks = scenario_shocks()
    move20 = scores.abs().quantile(0.99)                             # 99th pct 20-day factor move (score units)
    for pc in loadings:
        size = move20[pc] * np.abs(loadings[pc]).max()               # bp at the most-loaded tenor
        shocks[f"PCA {pc} 99% 20d ({size:.0f}bp)"] = pca_shock(loadings, pc, size)
    risk, krd, pnl = portfolio_report(positions, curve, shocks)
    var, var_pnl = historical_var(positions, curve, Zw)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(var_pnl / B, bins=60, color="C0", alpha=0.8)
    ax.axvline(-var["VaR 99%"] / B, color="C3", label=f"99% VaR = {var['VaR 99%'] / B:.2f}bn")
    ax.set(xlabel="10-day P&L (VND bn)", title="Historical-simulation P&L, illustrative portfolio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "var.png")

    # VaR backtest: 1-day P&L of the same (constant-maturity) portfolio against VaR estimated the day before
    daily = scenario_pnl(positions, curve, changes(Zw, 1))
    models = {"HS 250d, 99%": (hs_var(daily, 0.99), 0.99), "HS 250d, 95%": (hs_var(daily, 0.95), 0.95),
              "EWMA, 99%": (ewma_var(daily, 0.99), 0.99), "EWMA, 95%": (ewma_var(daily, 0.95), 0.95)}
    bt, exc = {}, {}
    for name, (v, lv) in models.items():
        bt[name], exc[name] = backtest(daily, v, lv)
    bt = pd.DataFrame(bt).T
    last250 = {name: int(exc[name].iloc[-250:].sum()) for name in models if "99%" in name}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(daily.index, daily / B, width=1.5, color="0.6", label="daily P&L")
    for name, c in (("HS 250d, 99%", "C3"), ("EWMA, 99%", "C0")):
        ax.plot(daily.index, -models[name][0] / B, color=c, lw=1, label=f"-VaR {name}")
        e = exc[name][exc[name]].index
        ax.plot(e, daily[e] / B, "v", color=c, ms=4)
    ax.set(ylabel="VND bn", title="1-day VaR backtest, illustrative portfolio (markers = exceptions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "var_backtest.png")

    bn = lambda x: f"{x / B:,.2f}"  # noqa: E731
    md = [
        "# Vietnam Government Bond Curve & Rate Risk Report",
        f"Data: HNX daily yield curve, {Zw.index.min():%d %b %Y} to {d:%d %b %Y} ({len(Zw)} days).",
        "", "## 1. Curve construction (latest day)", "![curve](curve_latest.png)", "",
        f"NSS parameters: beta = {np.round(nss.beta * 100, 3).tolist()} (%), lambda = {np.round(nss.lambdas, 2).tolist()}",
        "", fit.round(3).to_markdown(), "",
        "## 2. Fit quality across history (RMSE, bp)", fq.describe().round(2).to_markdown(), "",
        "## 2b. Data quality",
        "Lag-1 autocorrelation of daily zero-rate changes (negative = day-to-day refit noise that mean-reverts):", "",
        noise.round(2).to_frame("lag-1 autocorr").T.to_markdown(), "",
        f"Known anomaly excluded from PCA and VaR windows: snap-back on {', '.join(BREAKS)} "
        f"(10Y {brk.iloc[0]:.2f}% -> {brk.iloc[1]:.2f}% in one day, after drifting up from ~2.6% since late Sep 2024 "
        "with the 2Y flat; likely a fit artefact from thin long-end trading).", "",
        "## 3. Curve history", "![history](history.png)", "",
        "## 4. PCA of 20-day changes", "![pca](pca.png)", "",
        pd.DataFrame({"20-day changes": explained, "1-day changes": explained_1d}).map("{:.1%}".format).to_markdown(), "",
        "Daily changes understate the level factor because of the refit noise in section 2b; "
        "at a 20-day horizon the classic level/slope/curvature structure appears.", "",
        "## 5. Illustrative portfolio (VND 100bn face, coupons at today's par yields)",
        risk.assign(market_value=risk["market_value"].map(bn), face=risk["face"].map(bn),
                    dv01=risk["dv01"].map(lambda x: f"{x / 1e6:,.1f}m"))
            .round(4).to_markdown(), "",
        f"Portfolio DV01: VND {risk['dv01'].sum() / 1e6:,.1f}m per bp", "",
        "### Key-rate DV01 (VND m per bp)", (krd / 1e6).round(2).to_markdown(), "",
        "### Scenario P&L (VND bn, full revaluation)", (pnl / B).round(3).to_markdown(), "",
        "### Historical-simulation VaR (10-day, overlapping windows)", "![var](var.png)", "",
        (var / B).round(3).to_frame("VND bn").to_markdown(), "",
        "### VaR backtest (1-day)",
        "Each day's P&L is compared with the VaR estimated the evening before. HS uses the previous 250 days; "
        "EWMA is the RiskMetrics normal model with lambda 0.94. Kupiec tests whether the number of exceptions "
        "is right, Christoffersen whether they bunch together.", "",
        "![backtest](var_backtest.png)", "",
        bt.round(3).to_markdown(), "",
        "Last 250 days, 99% VaR: " + ", ".join(f"{k} {n} exceptions ({traffic_light(n)} zone)" for k, n in last250.items())
        + ". Daily HNX changes include refit noise (section 2b), so 1-day P&L here is noisier than a real book's.", "",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
