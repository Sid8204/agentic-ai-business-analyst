"""SQL / data-analysis tool. This is the agent's primary way of grounding
answers in real numbers instead of guessing.
"""

import re

from agents.db import get_connection

MAX_ROWS = 200

_TABLE_DESCRIPTIONS = {
    "orders": "One row per order: status, customer_id, purchase/delivery timestamps.",
    "order_items": "One row per item within an order: product_id, seller_id, price, freight_value.",
    "order_payments": "Payment records per order: payment_type, installments, value.",
    "order_reviews": "Customer review per order: review_score (1-5), comment timestamps.",
    "customers": "customer_id <-> customer_unique_id mapping, city/state.",
    "products": "Product catalog: category name (Portuguese), dimensions, weight.",
    "sellers": "seller_id, seller city/state.",
    "geolocation": "Zip-code prefix to lat/lng lookup.",
    "category_translation": "Maps Portuguese product_category_name to English.",
    "monthly_sales": "VIEW — gross_revenue, freight_revenue, order_count per calendar month. Start here for trend questions. NOTE: 2016-09/10/12 are pre-launch seed orders (near-zero volume, Nov 2016 is missing entirely) and 2018-09 is truncated by the dataset's collection cutoff (only 1 order) — exclude these edge months from trend/decline analysis and treat 2017-01 through 2018-08 as the real operational window.",
    "seller_performance": "VIEW — revenue, order_count, distinct_products per seller per month. Start here for seller comparisons.",
    "category_performance": "VIEW — revenue, order_count per product category (English name) per month.",
    "customer_orders": "VIEW — order_value per customer (customer_unique_id) per order, for RFM/segmentation.",
}


def get_schema() -> dict:
    """Returns every table/view with its columns and a short description.
    Call this before writing SQL if you're unsure what's available."""
    con = get_connection()
    rows = con.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
    """).fetchall()

    tables: dict[str, list[str]] = {}
    for table_name, column_name, data_type in rows:
        tables.setdefault(table_name, []).append(f"{column_name} ({data_type})")

    return {
        "tables": [
            {
                "name": name,
                "description": _TABLE_DESCRIPTIONS.get(name, ""),
                "columns": cols,
            }
            for name, cols in tables.items()
        ]
    }


_DISALLOWED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|copy|pragma)\b",
    re.IGNORECASE,
)


def run_sql(query: str) -> dict:
    """Executes a read-only SQL SELECT query against the Olist DuckDB
    database and returns the results. Only SELECT/WITH statements are
    allowed. Results are capped at 200 rows — aggregate in SQL rather
    than pulling raw rows when possible.
    """
    query = query.strip().rstrip(";")

    if ";" in query:
        return {"error": "Only a single SQL statement is allowed per call."}

    stripped = query.lstrip().lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        return {"error": "Only SELECT/WITH (read-only) statements are allowed."}

    if _DISALLOWED.search(query):
        return {"error": "Query contains a disallowed keyword (only read-only SELECT is permitted)."}

    try:
        con = get_connection()
        cursor = con.execute(query)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS + 1)
    except Exception as exc:  # noqa: BLE001 — surfaced back to the LLM to self-correct
        return {"error": f"SQL error: {exc}"}

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]

    return {
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }


TOOLS = [
    {
        "name": "get_schema",
        "description": get_schema.__doc__.strip(),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql",
        "description": run_sql.__doc__.strip(),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single read-only SQL SELECT/WITH statement."},
            },
            "required": ["query"],
        },
    },
]

DISPATCH = {"get_schema": lambda **kw: get_schema(), "run_sql": lambda **kw: run_sql(**kw)}
