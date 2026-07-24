import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE_URL = "https://raw.githubusercontent.com/Ganesh7699/Brazilian-E-Commerce-OList/main"

FILES = [
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "product_category_name_translation.csv",
]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"skip  {name} (exists)")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"fetch {name}")
        urllib.request.urlretrieve(url, dest)
    print(f"done -> {RAW_DIR}")


if __name__ == "__main__":
    sys.exit(main())
