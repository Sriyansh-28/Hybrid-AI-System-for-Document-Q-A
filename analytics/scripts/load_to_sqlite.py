import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "olist.db"
SCHEMA_PATH = ROOT / "schema.sql"

TABLES = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "rename": {"customer_zip_code_prefix": "customer_zip_prefix"},
        "numeric": [],
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "rename": {"seller_zip_code_prefix": "seller_zip_prefix"},
        "numeric": [],
    },
    "category_translation": {
        "file": "product_category_name_translation.csv",
        "rename": {
            "product_category_name": "category_name",
            "product_category_name_english": "category_name_english",
        },
        "numeric": [],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "rename": {
            "product_category_name": "category_name",
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_desc_length",
        },
        "numeric": [
            "product_name_length",
            "product_desc_length",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "rename": {},
        "numeric": [],
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "rename": {},
        "numeric": ["order_item_id", "price", "freight_value"],
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "rename": {},
        "numeric": ["payment_sequential", "payment_installments", "payment_value"],
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "rename": {},
        "numeric": ["review_score"],
    },
}

LOAD_ORDER = [
    "customers",
    "sellers",
    "category_translation",
    "products",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
]


def load_table(conn, name, spec):
    df = pd.read_csv(RAW_DIR / spec["file"], dtype=str, keep_default_na=False)
    df = df.replace("", None)
    if spec["rename"]:
        df = df.rename(columns=spec["rename"])
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({name})").fetchall()]
    df = df[[c for c in cols if c in df.columns]]
    for col in spec["numeric"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_sql(name, conn, if_exists="append", index=False)
    return len(df)


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("No raw data found. Run download_data.py first.")
        return 1
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    for name in LOAD_ORDER:
        n = load_table(conn, name, TABLES[name])
        print(f"loaded {name:22s} {n:>7d} rows")
    conn.commit()
    conn.close()
    print(f"done -> {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
