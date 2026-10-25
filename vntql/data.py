"""Vietnam Government Bond yield curve data from HNX (Hanoi Stock Exchange).

HNX publishes a daily fitted curve with continuous spot rates, par yields and
annually-compounded spot rates for 11 tenors (3M to 20Y).
"""
import html
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import truststore

# hnx.vn serves an incomplete certificate chain; the OS trust store resolves it, certifi does not
truststore.inject_into_ssl()

URL = "https://hnx.vn/ModuleReportBonds/Bond_YieldCurve/SearchAndNextPageYieldCurveData"
CACHE = Path(__file__).resolve().parent.parent / "data" / "hnx_curve.csv"
COLUMNS = ["date", "tenor", "spot_cont", "par", "spot_annual"]
# Data anomaly: HNX's long end drifted up (10Y 2.6% -> 3.9%, 23 Sep - 21 Oct 2024, 2Y flat) and
# snapped back to 2.61% on this date; most likely a fit artefact from thin long-end trading.
# Windows spanning the snap-back are excluded from change-based statistics.
BREAKS = ("2024-10-22",)


def _tenor_years(label):
    n, unit = label.split()
    return int(n) / 12 if unit.startswith("th") else float(n)


def _num(s):
    s = s.strip().replace(",", ".")
    return float(s) / 100 if s else np.nan


def parse_curve(page, d):
    """Parse the HNX HTML fragment into rows (rates as decimals)."""
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", page, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) != 4:
            continue
        cont, par, annual = map(_num, cells[1:])
        if np.isnan(annual):
            annual = np.expm1(cont)
        rows.append((d, _tenor_years(cells[0]), cont, par, annual))
    return rows


def fetch_day(d, session=None):
    s = session or requests
    r = s.post(URL, data={"pDate": d.strftime("%d/%m/%Y")},
               headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}, timeout=20)
    r.raise_for_status()
    return parse_curve(r.text, d)


def update_cache(start=date(2020, 1, 1), end=None, pause=0.3):
    """Download missing business days into the CSV cache and return the full history."""
    end = end or date.today()
    df = load(raise_if_missing=False)
    have = set(df["date"].dt.date) if len(df) else set()
    last = max(have) if have else start - timedelta(days=1)
    new, s = [], requests.Session()
    d = max(start, last + timedelta(days=1))
    while d <= end:
        if d.weekday() < 5 and d not in have:
            try:
                new += fetch_day(d, s)
            except requests.RequestException as e:
                print(f"{d}: {e}")
            time.sleep(pause)
        d += timedelta(days=1)
    if new:
        new = pd.DataFrame(new, columns=COLUMNS).assign(date=lambda x: pd.to_datetime(x["date"]))
        df = pd.concat([df, new]) if len(df) else new
        CACHE.parent.mkdir(exist_ok=True)
        df.sort_values(["date", "tenor"]).to_csv(CACHE, index=False)
    return load()


def load(raise_if_missing=True):
    if not CACHE.exists():
        if raise_if_missing:
            raise FileNotFoundError(f"{CACHE} missing; run scripts/fetch_hnx.py")
        return pd.DataFrame(columns=COLUMNS).astype({"date": "datetime64[ns]"})
    return pd.read_csv(CACHE, parse_dates=["date"])


def wide(df, field="spot_cont"):
    """Date x tenor matrix of one field."""
    return df.pivot(index="date", columns="tenor", values=field).sort_index()


def changes(rates, horizon=1, breaks=BREAKS):
    """`horizon`-day changes, dropping windows that span a known data break."""
    dx = rates.diff(horizon).dropna()
    idx = rates.index
    for b in pd.to_datetime(list(breaks)):
        pos = idx.searchsorted(b)
        bad = idx[pos:pos + horizon]
        dx = dx.drop(bad, errors="ignore")
    return dx
