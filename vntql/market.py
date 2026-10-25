"""USD rates (US Treasury par curve) and USD/VND spot history, cached under data/."""
import io
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import data  # noqa: F401  (installs the OS trust store for HTTPS)
from .curves import bootstrap_par

DATA = data.CACHE.parent
UST_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           "daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
UST_TENORS = {"1 Mo": 1 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25, "4 Mo": 4 / 12, "6 Mo": 0.5, "1 Yr": 1,
              "2 Yr": 2, "3 Yr": 3, "5 Yr": 5, "7 Yr": 7, "10 Yr": 10, "20 Yr": 20, "30 Yr": 30}
UA = {"User-Agent": "Mozilla/5.0"}


def fetch_ust(start_year=2020, end_year=None):
    """US Treasury par yield curve (CMT, bond-equivalent %, decimals out). Date x tenor."""
    end_year = end_year or date.today().year
    frames = []
    for y in range(start_year, end_year + 1):
        r = requests.get(UST_URL.format(y=y), headers=UA, timeout=30)
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]))
    df = pd.concat(frames).set_index("Date").sort_index()
    df = df[[c for c in UST_TENORS if c in df]].rename(columns=UST_TENORS) / 100
    df.index.name = "date"
    df.to_csv(DATA / "ust_par.csv")
    return df


def fetch_usdvnd(start="2019-01-01"):
    """Daily USD/VND close (VND per USD) from Yahoo Finance."""
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/VND=X?period1={p1}&period2=9999999999&interval=1d"
    res = requests.get(url, headers=UA, timeout=30).json()["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(res["timestamp"], unit="s").normalize(), name="usdvnd")
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    s.to_csv(DATA / "usdvnd.csv")
    return s


def load_ust():
    return pd.read_csv(DATA / "ust_par.csv", index_col="date", parse_dates=True).rename(columns=float)


def load_usdvnd():
    return pd.read_csv(DATA / "usdvnd.csv", index_col="date", parse_dates=True)["usdvnd"]


def ust_curve(row):
    """Zero curve from one day of CMT par yields: bills (<= 1Y) as semiannual zeros, notes/bonds bootstrapped."""
    row = row.dropna()
    t, y = row.index.to_numpy(float), row.to_numpy(float)
    return bootstrap_par(t, y, short_zeros=y, freq=2, short_freq=2, cutoff=1.0 + 1e-9)


def usd_zero_1y(ust):
    """Continuous 1Y USD rate series from the 1Y CMT (bond-equivalent)."""
    return 2 * np.log1p(ust[1.0] / 2)
