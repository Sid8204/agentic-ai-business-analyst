"""ML tools: forecasting, anomaly detection, and customer segmentation.
These are the agent's ML capabilities beyond plain SQL aggregation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from agents.db import get_connection


# The raw Olist dataset includes a handful of pre-launch "seed" orders in
# 2016 (Sep/Oct/Dec, near-zero volume, Nov entirely missing) and a final
# month (2018-09) truncated by the dataset's collection cutoff to a single
# order. Both are data artifacts, not real business signal, so ML analysis
# is scoped to the operational window where the marketplace was actually
# running at scale.
OPERATIONAL_START = "2017-01-01"
OPERATIONAL_END = "2018-08-01"


def _monthly_sales_df() -> pd.DataFrame:
    con = get_connection()
    df = con.execute(
        "SELECT * FROM monthly_sales WHERE month BETWEEN ? AND ? ORDER BY month",
        [OPERATIONAL_START, OPERATIONAL_END],
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def forecast_sales(periods: int = 3, metric: str = "gross_revenue") -> dict:
    """Forecasts a monthly_sales metric (gross_revenue, freight_revenue, or
    order_count) N months into the future using exponential smoothing
    (Holt's linear trend method). Also returns the historical series and
    month-over-month growth rates so trend/decline questions are grounded
    in actual computed numbers, not guesses.
    """
    if metric not in ("gross_revenue", "freight_revenue", "order_count"):
        return {"error": f"Unknown metric '{metric}'. Use gross_revenue, freight_revenue, or order_count."}

    df = _monthly_sales_df()
    if len(df) < 4:
        return {"error": f"Only {len(df)} months of data available; need at least 4 to forecast."}

    series = df.set_index("month")[metric]
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
    series = series.asfreq("MS")
    if series.isna().any():
        series = series.interpolate()

    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        forecast = fit.forecast(periods)
        forecast.index = pd.date_range(
            start=series.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS"
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Forecasting failed: {exc}"}

    history = [
        {"month": d.strftime("%Y-%m"), "value": round(float(v), 2)}
        for d, v in series.items()
    ]
    growth = series.pct_change().dropna()
    forecast_points = [
        {"month": d.strftime("%Y-%m"), "value": round(float(v), 2)}
        for d, v in forecast.items()
    ]

    return {
        "metric": metric,
        "history": history,
        "month_over_month_growth_pct": [
            {"month": d.strftime("%Y-%m"), "growth_pct": round(float(v) * 100, 1)}
            for d, v in growth.items()
        ],
        "forecast": forecast_points,
        "chart": {"type": "line_with_forecast", "history": history, "forecast": forecast_points, "metric": metric},
    }


def detect_anomalies(metric: str = "gross_revenue", z_threshold: float = 1.5) -> dict:
    """Detects anomalous months in a monthly_sales metric by fitting a
    linear trend and flagging months whose residual z-score exceeds
    z_threshold (default 1.5). Use this to find genuine dips/spikes rather
    than relying on eyeballing a chart.
    """
    if metric not in ("gross_revenue", "freight_revenue", "order_count"):
        return {"error": f"Unknown metric '{metric}'. Use gross_revenue, freight_revenue, or order_count."}

    df = _monthly_sales_df()
    if len(df) < 4:
        return {"error": f"Only {len(df)} months of data available; need at least 4 to detect anomalies."}

    y = df[metric].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    residuals = y - trend
    std = residuals.std(ddof=1) or 1e-9
    z_scores = residuals / std

    anomalies = [
        {
            "month": df["month"].iloc[i].strftime("%Y-%m"),
            "value": round(float(y[i]), 2),
            "expected": round(float(trend[i]), 2),
            "z_score": round(float(z_scores[i]), 2),
            "direction": "spike" if z_scores[i] > 0 else "dip",
        }
        for i in range(len(y))
        if abs(z_scores[i]) >= z_threshold
    ]

    return {
        "metric": metric,
        "z_threshold": z_threshold,
        "anomalies": anomalies,
        "trend_slope_per_month": round(float(slope), 2),
    }


def segment_customers(n_clusters: int = 4) -> dict:
    """Segments customers into n_clusters groups using KMeans on RFM
    features (Recency in days, Frequency = order count, Monetary = total
    spend). Returns per-cluster summary stats so segments can be described
    in business terms (e.g. 'high-value, recent, frequent').
    """
    con = get_connection()
    df = con.execute("""
        SELECT customer_unique_id,
               MAX(order_purchase_timestamp) AS last_order,
               COUNT(*) AS frequency,
               SUM(order_value) AS monetary
        FROM customer_orders
        GROUP BY customer_unique_id
        HAVING COUNT(*) >= 1
    """).df()

    if len(df) < n_clusters * 5:
        return {"error": f"Only {len(df)} customers available; too few to reliably form {n_clusters} clusters."}

    now = df["last_order"].max()
    df["recency_days"] = (now - pd.to_datetime(df["last_order"])).dt.days

    features = df[["recency_days", "frequency", "monetary"]]
    scaled = StandardScaler().fit_transform(features)

    try:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        df["cluster"] = km.fit_predict(scaled)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Clustering failed: {exc}"}

    summary = (
        df.groupby("cluster")
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        )
        .round(1)
        .reset_index()
        .to_dict(orient="records")
    )

    return {"n_clusters": n_clusters, "total_customers": len(df), "segments": summary}


TOOLS = [
    {
        "name": "forecast_sales",
        "description": forecast_sales.__doc__.strip(),
        "input_schema": {
            "type": "object",
            "properties": {
                "periods": {"type": "integer", "description": "Number of future months to forecast.", "default": 3},
                "metric": {"type": "string", "enum": ["gross_revenue", "freight_revenue", "order_count"], "default": "gross_revenue"},
            },
        },
    },
    {
        "name": "detect_anomalies",
        "description": detect_anomalies.__doc__.strip(),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["gross_revenue", "freight_revenue", "order_count"], "default": "gross_revenue"},
                "z_threshold": {"type": "number", "default": 1.5},
            },
        },
    },
    {
        "name": "segment_customers",
        "description": segment_customers.__doc__.strip(),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_clusters": {"type": "integer", "default": 4},
            },
        },
    },
]

DISPATCH = {
    "forecast_sales": lambda **kw: forecast_sales(**kw),
    "detect_anomalies": lambda **kw: detect_anomalies(**kw),
    "segment_customers": lambda **kw: segment_customers(**kw),
}
