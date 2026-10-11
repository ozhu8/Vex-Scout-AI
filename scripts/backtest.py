"""Usage: python scripts/backtest.py <SKU>   (e.g. DEMO-SYNTHETIC)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vexscout import db, predictor  # noqa: E402

if __name__ == "__main__":
    conn = db.connect()
    ev = db.one(conn, "SELECT * FROM events WHERE sku=?", (sys.argv[1],))
    if not ev:
        sys.exit(f"Event {sys.argv[1]} not in database - ingest it first")
    bt = predictor.backtest(conn, ev["id"])
    bt.pop("matches", None)
    print(json.dumps(bt, indent=2))
