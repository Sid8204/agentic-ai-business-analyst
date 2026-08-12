"""
Loads the Olist Brazilian E-Commerce CSVs (from data/raw/) into a DuckDB
file (olist.duckdb) and creates a handful of analyst-friendly views on top
of the raw tables so agents can start from clean aggregates instead of
re-deriving the same joins in every query.
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "olist.duckdb"

# Kaggle's zip sometimes nests files one level deeper (olist_*.csv) or
# strips the "olist_" prefix depending on how it was extracted, so map by
# a substring match rather than requiring an exact filename.
TABLES = {
    "orders": "orders_dataset",
    "order_items": "order_items_dataset",
    "order_payments": "order_payments_dataset",
    "order_reviews": "order_reviews_dataset",
    "customers": "customers_dataset",
    "products": "products_dataset",
    "sellers": "sellers_dataset",
    "geolocation": "geolocation_dataset",
    "category_translation": "product_category_name_translation",
}


def find_csv(stem: str) -> Path | None:
    matches = list(RAW_DIR.glob(f"*{stem}*.csv"))
    return matches[0] if matches else None


def main() -> None:
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.csv")):
        print(f"No CSVs found in {RAW_DIR}. Download the Olist dataset from "
              f"kaggle.com/datasets/olistbr/brazilian-ecommerce and extract "
              f"it there first.", file=sys.stderr)
        sys.exit(1)

    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))

    loaded = {}
    for table_name, stem in TABLES.items():
        csv_path = find_csv(stem)
        if csv_path is None:
            print(f"  ! skipping {table_name}: no file matching *{stem}*.csv")
            continue
        con.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto(?, "
            f"sample_size=-1)",
            [str(csv_path)],
        )
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        loaded[table_name] = count
        print(f"  loaded {table_name}: {count:,} rows (from {csv_path.name})")

    missing = set(TABLES) - set(loaded)
    if missing:
        print(f"WARNING: missing tables {missing} — some tools may fail.",
              file=sys.stderr)

    build_views(con, loaded)
    con.close()
    print(f"\nDatabase built at {DB_PATH} ({sum(loaded.values()):,} total rows "
          f"across {len(loaded)} tables)")


def build_views(con: duckdb.DuckDBPyConnection, loaded: dict) -> None:
    if "orders" in loaded and "order_items" in loaded:
        con.execute("""
            CREATE VIEW monthly_sales AS
            SELECT
                date_trunc('month', o.order_purchase_timestamp) AS month,
                SUM(oi.price) AS gross_revenue,
                SUM(oi.freight_value) AS freight_revenue,
                COUNT(DISTINCT o.order_id) AS order_count
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY 1
            ORDER BY 1
        """)
        print("  created view: monthly_sales")

    if "orders" in loaded and "order_items" in loaded and "sellers" in loaded:
        con.execute("""
            CREATE VIEW seller_performance AS
            SELECT
                s.seller_id,
                s.seller_state,
                date_trunc('month', o.order_purchase_timestamp) AS month,
                SUM(oi.price) AS revenue,
                COUNT(DISTINCT o.order_id) AS order_count,
                COUNT(DISTINCT oi.product_id) AS distinct_products
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.order_id
            JOIN sellers s ON s.seller_id = oi.seller_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY 1, 2, 3
            ORDER BY 3, revenue DESC
        """)
        print("  created view: seller_performance")

    if all(t in loaded for t in ("orders", "order_items", "products", "category_translation")):
        con.execute("""
            CREATE VIEW category_performance AS
            SELECT
                COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
                date_trunc('month', o.order_purchase_timestamp) AS month,
                SUM(oi.price) AS revenue,
                COUNT(DISTINCT o.order_id) AS order_count
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.order_id
            JOIN products p ON p.product_id = oi.product_id
            LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY 1, 2
            ORDER BY 2, revenue DESC
        """)
        print("  created view: category_performance")

    if all(t in loaded for t in ("orders", "order_items", "customers")):
        con.execute("""
            CREATE VIEW customer_orders AS
            SELECT
                c.customer_unique_id,
                o.order_id,
                o.order_purchase_timestamp,
                SUM(oi.price) AS order_value
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.order_id
            JOIN customers c ON c.customer_id = o.customer_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY 1, 2, 3
        """)
        print("  created view: customer_orders")


if __name__ == "__main__":
    main()
