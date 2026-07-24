import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "olist.db"
ANALYSIS_DIR = ROOT / "analysis"
EXPORT_DIR = ROOT / "exports"


def sql_files():
    files = sorted(ANALYSIS_DIR.glob("*.sql"))
    files += sorted((ANALYSIS_DIR / "quality").glob("*.sql"))
    return files


def main():
    if not DB_PATH.exists():
        print("Database not found. Run load_to_sqlite.py first.")
        return 1
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    for path in sql_files():
        query = path.read_text()
        df = pd.read_sql_query(query, conn)
        out = EXPORT_DIR / f"{path.stem}.csv"
        df.to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"{path.stem:36s} {len(df):>4d} rows -> exports/{out.name}")
    conn.close()
    print(f"done -> {EXPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
