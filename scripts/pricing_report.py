"""Treasury pricing report: USD/VND forwards (CIP), FX options (Garman-Kohlhagen), VND swaps and hedging.

-> reports/pricing.md + charts. Run scripts/fetch_hnx.py and scripts/fetch_market.py first.
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
from vntql.bonds import KEY_TENORS, illustrative_portfolio  # noqa: E402
from vntql.curves import InterpCurve  # noqa: E402
from vntql.data import load, wide  # noqa: E402
from vntql.fx import FXForward, cip_forward  # noqa: E402
from vntql.market import load_usdvnd, load_ust, ust_curve, usd_zero_1y  # noqa: E402
from vntql.options import gk_greeks, strike_from_delta  # noqa: E402
from vntql.swaps import Swap  # noqa: E402

OUT = ROOT / "reports"
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
B = 1e9


def snapshot(Zw, ust, fx, d):
    """VND curve, USD curve and spot as of date d (latest available on or before d)."""
    vnd = InterpCurve(Zw.columns.to_numpy(float), Zw.loc[:d].iloc[-1].to_numpy())
    usd = ust_curve(ust.loc[:d].iloc[-1])
    return vnd, usd, float(fx.loc[:d].iloc[-1])


def forward_table(S, vnd, usd):
    rows = []
    for name, T in (("1M", 1 / 12), ("3M", 0.25), ("6M", 0.5), ("1Y", 1), ("2Y", 2), ("5Y", 5)):
        F = cip_forward(S, T, vnd, usd)
        rows.append({"tenor": name, "VND zero %": vnd.zero(T) * 100, "USD zero %": usd.zero(T) * 100,
                     "forward": F, "points": F - S, "premium % p.a.": np.log(F / S) / T * 100})
    return pd.DataFrame(rows).set_index("tenor")


def cip_vs_realised(Zw, ust, fx):
    """1Y CIP forward premium vs the realised 1Y change in USD/VND."""
    fxb = fx[fx.index.dayofweek < 5]
    df = pd.DataFrame({"r_vnd": Zw[1.0], "r_usd": usd_zero_1y(ust)}).dropna()
    df = df.join(fxb.rename("spot"), how="inner")
    df["premium"] = df["r_vnd"] - df["r_usd"]                       # log(F/S) for T = 1
    fut = fxb.reindex(df.index + pd.DateOffset(years=1), method="ffill", tolerance=pd.Timedelta("7D"))
    df["realised"] = np.log(fut.to_numpy() / df["spot"].to_numpy())
    df = df.dropna()
    # long USD via 1Y forward: P&L per USD = S_T - F = S (exp(realised) - exp(premium))
    df["long_usd_fwd_ret"] = np.exp(df["realised"]) - np.exp(df["premium"])
    slope, intercept = np.polyfit(df["premium"], df["realised"], 1)
    stats = pd.Series({
        "observations (overlapping daily)": len(df),
        "mean 1Y forward premium %": df["premium"].mean() * 100,
        "mean realised 1Y change %": df["realised"].mean() * 100,
        "correlation": df["premium"].corr(df["realised"]),
        "regression slope (realised on premium)": slope,
        "regression intercept %": intercept * 100,
        "long USD fwd: mean return %": df["long_usd_fwd_ret"].mean() * 100,
        "long USD fwd: hit rate %": (df["long_usd_fwd_ret"] > 0).mean() * 100,
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["premium"] * 100, label="1Y CIP forward premium (VND - USD rate)")
    ax.plot(df.index, df["realised"] * 100, label="Realised USD/VND change over next 1Y")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(ylabel="% (log)", title="USD/VND: what the forward priced vs what happened")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cip_vs_realised.png")
    return stats


def realised_vol(fx, d, window=252):
    r = np.log(fx[fx.index.dayofweek < 5].loc[:d]).diff().dropna().iloc[-window:]
    return float(r.std() * np.sqrt(252))


def option_table(S, vnd, usd, vol, T=0.5, notional=1e6):
    rd, rf = float(vnd.zero(T)), float(usd.zero(T))
    F = cip_forward(S, T, r_dom=rd, r_for=rf)
    rows = {}
    for name, K, call in (("ATMF call", F, True), ("ATMF put", F, False),
                          ("25d USD call", strike_from_delta(0.25, S, T, rd, rf, vol), True),
                          ("25d USD put", strike_from_delta(-0.25, S, T, rd, rf, vol, call=False), False)):
        g = gk_greeks(S, K, T, rd, rf, vol, call)
        rows[name] = {"strike": K, "premium (VND/USD)": g["price"], "premium % of spot": g["price"] / S * 100,
                      "delta": g["delta"], "gamma (per 1M USD)": g["gamma"] * notional,
                      "vega (VND m, 1 vol pt)": g["vega"] * notional / 1e6,
                      "theta (VND m/day)": g["theta"] * notional / 1e6,
                      "rho VND (VND m/bp)": g["rho_dom"] * notional / 1e6,
                      "rho USD (VND m/bp)": g["rho_for"] * notional / 1e6}
    return pd.DataFrame(rows).T, rd, rf


def hedge(vnd):
    """DV01-hedge the Project 1 bond portfolio with payer swaps; compare key-rate residuals."""
    pos = illustrative_portfolio(vnd, 100 * B)
    port = pd.Series(0.0, index=list(KEY_TENORS))
    for _, b, amt in pos:
        port += pd.Series(b.key_rate_dv01(vnd)) * amt / b.face
    port = -port                                               # value change for +1bp (long bonds lose)

    def payer(m, n=1.0):
        s = Swap(n, 0, m)
        return Swap(n, s.par_rate(vnd), m)

    kr = {m: pd.Series(payer(m).key_rate_dv01(vnd)) for m in (2, 5, 10, 15)}
    n10 = -port.sum() / kr[10].sum()
    one = port + n10 * kr[10]

    def kr_hedge(tenors):
        """Least-squares payer notionals matching all key-rate buckets."""
        A = np.column_stack([kr[m] for m in tenors])
        n, *_ = np.linalg.lstsq(A, -port.to_numpy(), rcond=None)
        return n, port + A @ n

    n2, two = kr_hedge((5, 10))              # only the liquid 5Y and 10Y points
    n4, four = kr_hedge((2, 5, 10, 15))      # matched maturities: sanity check, a par bond = receiver swap + FRN

    krd = pd.DataFrame({"Bonds": port, "+ 10Y payer": one, "+ 5Y/10Y payers": two,
                        "+ 2/5/10/15Y payers": four}) / 1e6
    notionals = pd.Series({"10Y (DV01 hedge)": n10,
                           **{f"{m}Y (5Y/10Y key-rate hedge)": x for m, x in zip((5, 10), n2)},
                           **{f"{m}Y (matched-maturity hedge)": x for m, x in zip((2, 5, 10, 15), n4)}}) / B
    rates = pd.Series({f"{m}Y par swap %": payer(m).fixed_rate * 100 for m in (2, 5, 10, 15)})
    return krd, notionals, rates


def main():
    Zw, ust, fx = wide(load()), load_ust(), load_usdvnd()
    d = min(Zw.index.max(), ust.index.max(), fx.index.max())
    vnd, usd, S = snapshot(Zw, ust, fx, d)

    fwd = forward_table(S, vnd, usd)
    stats = cip_vs_realised(Zw, ust, fx)

    # FX forward MTM: bought USD 10m 1Y forward ~6 months ago at the then-CIP rate
    d0 = Zw.index[Zw.index.searchsorted(d - pd.DateOffset(months=6))]
    vnd0, usd0, S0 = snapshot(Zw, ust, fx, d0)
    K = cip_forward(S0, 1.0, vnd0, usd0)
    trade = FXForward(10e6, K, 1.0)
    t = (d - d0).days / 365
    mtm = trade.value(S, vnd, usd, t)
    risks = trade.risks(S, vnd, usd, t)

    vol = realised_vol(fx, d)
    opts, rd, rf = option_table(S, vnd, usd, vol)
    krd, notionals, swap_rates = hedge(vnd)

    f2 = lambda x: f"{x:,.2f}"  # noqa: E731
    md = [
        "# Treasury Pricing: USD/VND Forwards, FX Options, VND Swaps",
        f"Valuation date {d:%d %b %Y}. USD/VND spot {S:,.0f} (Yahoo). USD curve: US Treasury CMT, bootstrapped. "
        "VND curve: HNX government bond curve, used as a **proxy** for VND funding rates (the interbank / "
        "deposit curve and quoted forward points are not publicly downloadable), so forwards here are "
        "CIP-theoretical, without cross-currency basis.", "",
        "## 1. CIP forwards", fwd.round(2).to_markdown(), "",
        "Forward points are negative when VND rates are below USD rates: USD trades at a forward discount.", "",
        "## 2. Did the forward predict USD/VND? (1Y, 2020 onward)", "![cip](cip_vs_realised.png)", "",
        stats.round(3).to_frame("value").to_markdown(), "",
        "Under uncovered interest parity the slope would be 1 and the intercept 0. For a managed currency like VND the "
        "realised path is driven by SBV policy and the USD cycle rather than the rate differential. "
        "Caveats: overlapping 1Y windows (about 5 independent observations), a sample dominated by the 2022-2025 "
        "VND depreciation, and government-curve proxies for funding rates.", "",
        "## 3. FX forward mark-to-market",
        f"Bought USD 10m 1Y forward on {d0:%d %b %Y} at {K:,.1f} (spot then {S0:,.0f}). Today, {t:.2f}y later:", "",
        pd.Series({"MTM (VND m)": mtm / 1e6, "spot delta (USD)": risks["spot_delta"],
                   "DV01 to VND curve (VND m/bp)": risks["dv01_dom"] / 1e6,
                   "DV01 to USD curve (VND m/bp)": risks["dv01_for"] / 1e6}).map(f2).to_frame("value").to_markdown(), "",
        "## 4. FX options (Garman-Kohlhagen, 6M, USD 1m notional)",
        f"Vol = 1Y realised USD/VND vol {vol:.2%} (no public implied-vol quotes). rd = {rd:.3%}, rf = {rf:.3%} (continuous).", "",
        opts.round(3).to_markdown(), "",
        "## 5. VND interest rate swaps and hedging the bond portfolio",
        "Single-curve valuation on the VND proxy curve (annual fixed vs quarterly floating). "
        f"Par rates: {', '.join(f'{k} {v:.3f}' for k, v in swap_rates.items())}.", "",
        "Project 1's illustrative VND 100bn bond portfolio hedged with payer swaps. "
        "Key-rate DV01 = value change for a +1bp bump, VND m:", "",
        krd.round(2).to_markdown(), "",
        "Payer swap notionals (VND bn):", "", notionals.round(1).to_frame("notional").to_markdown(), "",
        f"A single 10Y swap removes the parallel DV01 (gross key-rate exposure {krd['Bonds'].abs().sum():.1f}m -> "
        f"{krd['+ 10Y payer'].abs().sum():.1f}m) but leaves offsetting curve risk: net short duration at 10Y, net long at 2Y/5Y/15Y. "
        f"Using only the liquid 5Y and 10Y swaps to match key rates leaves {krd['+ 5Y/10Y payers'].abs().sum():.1f}m. "
        "Matched-maturity swaps hedge exactly (notionals equal the bond faces) because, on a single curve, a par bond "
        "is a receiver swap plus a floating-rate note: a consistency check on the pricers.", "",
    ]
    (OUT / "pricing.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "pricing.md")


if __name__ == "__main__":
    main()
