"""Download US Treasury par curves and USD/VND spot history into data/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vntql.market import fetch_ust, fetch_usdvnd  # noqa: E402

if __name__ == "__main__":
    ust = fetch_ust(2020)
    fx = fetch_usdvnd("2019-01-01")
    print(f"UST {ust.index.min():%Y-%m-%d}..{ust.index.max():%Y-%m-%d}; USD/VND {fx.index.min():%Y-%m-%d}..{fx.index.max():%Y-%m-%d}")
