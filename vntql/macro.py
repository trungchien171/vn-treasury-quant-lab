"""Market and macro drivers for USD/VND: Yahoo Finance prices and IMF monthly statistics, cached under data/."""
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from . import data  # noqa: F401  (installs the OS trust store for HTTPS)

DATA = data.CACHE.parent
UA = {"User-Agent": "Mozilla/5.0"}
YAHOO = {"usdvnd": "VND=X", "dxy": "DX-Y.NYB", "usdcny": "CNY=X", "brent": "BZ=F"}
IMF = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,{flow}/{key}?startPeriod={start}"
IMF_SERIES = {
    "cpi_vn": ("CPI", "VNM.CPI._T.IX.M"),          # consumer price index
    "cpi_us": ("CPI", "USA.CPI._T.IX.M"),
    "exports_vn": ("ITG", "VNM.XG.FOB_USD.M"),     # goods exports, USD
    "imports_vn": ("ITG", "VNM.MG.CIF_USD.M"),     # goods imports, USD
}


def yahoo_daily(symbol, start="2010-01-01"):
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2=9999999999&interval=1d"
    res = requests.get(url, headers=UA, timeout=30).json()["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(res["timestamp"], unit="s").normalize()).dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index.dayofweek < 5]


def imf_monthly(flow, key, start="2008"):
    """One IMF SDMX series as a month-end indexed Series."""
    r = requests.get(IMF.format(flow=flow, key=key, start=start), headers=UA, timeout=60)
    r.raise_for_status()
    obs = re.findall(r'TIME_PERIOD="(\d{4})-M(\d{2})" OBS_VALUE="([^"]+)"', r.text)
    if not obs:
        raise ValueError(f"no observations for {flow}/{key}")
    idx = pd.to_datetime([f"{y}-{m}-01" for y, m, _ in obs]) + pd.offsets.MonthEnd(0)
    return pd.Series([float(v) for *_, v in obs], index=idx).sort_index()


def fetch_drivers(start="2010-01-01"):
    """Daily market prices -> data/market_daily.csv; monthly IMF macro -> data/macro_monthly.csv."""
    daily = pd.DataFrame({k: yahoo_daily(sym, start) for k, sym in YAHOO.items()})
    daily.index.name = "date"
    daily.to_csv(DATA / "market_daily.csv")
    macro = pd.DataFrame({k: imf_monthly(f, key, start[:4]) for k, (f, key) in IMF_SERIES.items()})
    macro.index.name = "date"
    macro.to_csv(DATA / "macro_monthly.csv")
    return daily, macro


def load_drivers():
    daily = pd.read_csv(DATA / "market_daily.csv", index_col="date", parse_dates=True)
    macro = pd.read_csv(DATA / "macro_monthly.csv", index_col="date", parse_dates=True)
    return daily, macro
