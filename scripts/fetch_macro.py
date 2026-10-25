"""Download USD/VND drivers (Yahoo daily prices, IMF monthly CPI and trade) into data/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vntql.macro import fetch_drivers  # noqa: E402

if __name__ == "__main__":
    daily, macro = fetch_drivers("2010-01-01")
    print("daily", daily.index.min().date(), daily.index.max().date(), daily.notna().sum().to_dict())
    print("macro", macro.index.min().date(), macro.index.max().date(), macro.notna().sum().to_dict())
    print(macro.tail(4))
