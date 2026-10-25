"""Download / update the HNX government bond yield curve history into data/hnx_curve.csv."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vntql.data import update_cache  # noqa: E402

if __name__ == "__main__":
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2020, 1, 1)
    df = update_cache(start=start)
    print(f"{df['date'].nunique()} days, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
