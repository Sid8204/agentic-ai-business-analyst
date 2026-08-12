"""SQL / data-analysis tool. This is the agent's primary way of grounding
answers in real numbers instead of guessing.
"""

import re

from agents.db import get_connection

MAX_ROWS = 200

_2016_CAVEAT = (
    " CAVEAT: this table includes Sep/Oct/Dec 2016 orders that are pre-launch "
    "seed/test data (near-zero volume; Nov 2016 has none at all) — NOT a real "
    "trading period. Any query touching 2016 (e.g. a '2016 vs 2017' comparison) "
    "will produce a technically-correct but misleading result (e.g. a "
    "false '10,000%+ growth' figure) unless you explicitly caveat that 2016 "
    "isn't a comparable full year. Prefer monthly_sales/seller_performance/"
    "category_performance (already scoped to the real 2017-01–2018-08 window) "
    "for any trend or year-over-year question."
)

_TABLE_DESCRIPTIONS = {
    "orders": "One row per order: status, customer_id, purchase/delivery timestamps." + _2016_CAVEAT,
    "order_items": "One row per item within an order: product_id, seller_id, price, freight_value.",
    "order_payments": "Payment records per order: payment_type, installments, value.",
    "order_reviews": "Customer review per order: review_score (1-5), comment timestamps.",
    "customers": "customer_id <-> customer_unique_id mapping, city/state.",
    "products": "Product catalog: category name (Portuguese), dimensions, weight.",
    "sellers": "seller_id, seller city/state.",
    "geolocation": "Zip-code prefix to lat/lng lookup.",
    "category_translation": "Maps Portuguese product_category_name to English.",
    "monthly_sales": "VIEW — gross_revenue, freight_revenue, order_count per calendar month, already scoped to the real operational window (2017-01 through 2018-08; a handful of near-zero pre-launch 2016 orders and a truncated final month are excluded). Start here for trend questions.",
    "seller_performance": "VIEW — revenue, order_count, distinct_products per seller per month (same operational window as monthly_sales). Start here for seller comparisons.",
    "category_performance": "VIEW — revenue, order_count per product category (English name) per month (same operational window as monthly_sales).",
    "customer_orders": "VIEW — order_value per customer (customer_unique_id) per order, for RFM/segmentation. Deliberately NOT scoped to the operational window (recency calculations need full history)." + _2016_CAVEAT,
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


def find_largest_decline(metric: str = "gross_revenue") -> dict:
    """Finds the single month-over-month decline with the largest drop in
    a monthly_sales metric, within the real operational window (2017-01 to
    2018-08 — see monthly_sales caveats). Use this instead of hand-writing
    a LAG()/window-function query to find 'which month declined the most'.
    Returns that month, the prior month, the change, and the full
    month-over-month series for context.
    """
    if metric not in ("gross_revenue", "freight_revenue", "order_count"):
        return {"error": f"Unknown metric '{metric}'. Use gross_revenue, freight_revenue, or order_count."}

    con = get_connection()
    rows = con.execute(f"""
        SELECT month, {metric},
               LAG({metric}) OVER (ORDER BY month) AS prev_value,
               LAG(month) OVER (ORDER BY month) AS prev_month
        FROM monthly_sales
        WHERE month BETWEEN '2017-01-01' AND '2018-08-01'
        ORDER BY month
    """).fetchall()

    series = []
    largest = None
    for month, value, prev_value, prev_month in rows:
        if prev_value is None:
            continue
        change = value - prev_value
        pct_change = round(change / prev_value * 100, 1) if prev_value else None
        entry = {
            "month": month.strftime("%Y-%m"),
            "prev_month": prev_month.strftime("%Y-%m"),
            "value": round(value, 2),
            "prev_value": round(prev_value, 2),
            "change": round(change, 2),
            "pct_change": pct_change,
        }
        series.append(entry)
        if largest is None or entry["change"] < largest["change"]:
            largest = entry

    if largest is None:
        return {"error": "Not enough months of data to compute month-over-month changes."}

    return {"metric": metric, "largest_decline": largest, "series": series}


_DIMENSION_VIEWS = {
    "seller": ("seller_performance", "seller_id"),
    "category": ("category_performance", "category"),
}


def compare_periods(dimension: str, period_a: str, period_b: str, top_n: int = 10) -> dict:
    """Compares revenue for either 'seller' or 'category' across two
    specific months (format 'YYYY-MM') and returns entities sorted by
    change ascending — the biggest decliners first. Use this to attribute
    a change in the overall trend to specific sellers/categories, e.g.
    'which sellers dropped the most between 2017-10 and 2017-11'. This is
    more reliable than hand-writing a FULL OUTER JOIN yourself.
    """
    if dimension not in _DIMENSION_VIEWS:
        return {"error": "dimension must be 'seller' or 'category'"}
    view, key_col = _DIMENSION_VIEWS[dimension]

    try:
        con = get_connection()
        rows = con.execute(f"""
            WITH a AS (
                SELECT {key_col} AS entity, revenue FROM {view}
                WHERE date_trunc('month', month) = date_trunc('month', CAST(? AS DATE))
            ),
            b AS (
                SELECT {key_col} AS entity, revenue FROM {view}
                WHERE date_trunc('month', month) = date_trunc('month', CAST(? AS DATE))
            )
            SELECT COALESCE(a.entity, b.entity) AS entity,
                   COALESCE(a.revenue, 0) AS value_a,
                   COALESCE(b.revenue, 0) AS value_b,
                   COALESCE(b.revenue, 0) - COALESCE(a.revenue, 0) AS change
            FROM a FULL OUTER JOIN b ON a.entity = b.entity
            ORDER BY change ASC
            LIMIT ?
        """, [f"{period_a}-01", f"{period_b}-01", top_n]).fetchall()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"compare_periods failed: {exc}"}

    return {
        "dimension": dimension,
        "period_a": period_a,
        "period_b": period_b,
        "top_decliners": [
            {"entity": r[0], "value_a": round(r[1], 2), "value_b": round(r[2], 2), "change": round(r[3], 2)}
            for r in rows
        ],
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
    {
        "name": "find_largest_decline",
        "description": find_largest_decline.__doc__.strip(),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["gross_revenue", "freight_revenue", "order_count"], "default": "gross_revenue"},
            },
        },
    },
    {
        "name": "compare_periods",
        "description": compare_periods.__doc__.strip(),
        "input_schema": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": ["seller", "category"]},
                "period_a": {"type": "string", "description": "Baseline month, format YYYY-MM."},
                "period_b": {"type": "string", "description": "Comparison month, format YYYY-MM."},
                "top_n": {"type": "integer", "default": 10},
            },
            "required": ["dimension", "period_a", "period_b"],
        },
    },
]

DISPATCH = {
    "get_schema": lambda **kw: get_schema(),
    "run_sql": lambda **kw: run_sql(**kw),
    "find_largest_decline": lambda **kw: find_largest_decline(**kw),
    "compare_periods": lambda **kw: compare_periods(**kw),
}
