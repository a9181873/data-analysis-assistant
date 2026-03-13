"""
風險建模指標 — KS 統計量、Lift Chart、Gain Chart
"""

import numpy as np
import pandas as pd


def ks_statistic(y_true, y_prob, n_bins: int = 10):
    """
    計算 KS 統計量（Kolmogorov-Smirnov）。

    Args:
        y_true: 實際標籤 (0/1)
        y_prob: 預測為正類的機率
        n_bins: 分組數

    Returns:
        (ks_value, ks_table DataFrame)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    df["decile"] = pd.qcut(df.index, n_bins, labels=False, duplicates="drop") + 1

    total_events = y_true.sum()
    total_non_events = len(y_true) - total_events

    table = df.groupby("decile").agg(
        count=("y_true", "size"),
        events=("y_true", "sum"),
    ).reset_index()

    table["non_events"] = table["count"] - table["events"]
    table["event_rate"] = table["events"] / table["count"]
    table["cum_events"] = table["events"].cumsum()
    table["cum_non_events"] = table["non_events"].cumsum()
    table["cum_event_pct"] = table["cum_events"] / total_events
    table["cum_non_event_pct"] = table["cum_non_events"] / total_non_events
    table["ks"] = (table["cum_event_pct"] - table["cum_non_event_pct"]).abs()

    ks_value = table["ks"].max()
    return ks_value, table


def lift_chart_data(y_true, y_prob, n_bins: int = 10):
    """
    計算 Lift Chart 數據。

    Returns:
        DataFrame with columns: decile, response_rate, cumulative_lift
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False, duplicates="drop") + 1

    overall_rate = y_true.mean()

    table = df.groupby("decile").agg(
        count=("y_true", "size"),
        events=("y_true", "sum"),
    ).reset_index()

    table["response_rate"] = table["events"] / table["count"]
    table["cum_events"] = table["events"].cumsum()
    table["cum_count"] = table["count"].cumsum()
    table["cum_response_rate"] = table["cum_events"] / table["cum_count"]
    table["cumulative_lift"] = table["cum_response_rate"] / overall_rate

    return table[["decile", "response_rate", "cumulative_lift", "cum_response_rate"]]


def gain_chart_data(y_true, y_prob, n_bins: int = 10):
    """
    計算 Gain (Cumulative Capture Rate) Chart 數據。

    Returns:
        DataFrame with columns: percentile, cumulative_gain
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False, duplicates="drop") + 1

    total_events = y_true.sum()

    table = df.groupby("decile").agg(
        events=("y_true", "sum"),
    ).reset_index()

    table["cum_events"] = table["events"].cumsum()
    table["cumulative_gain"] = table["cum_events"] / total_events
    actual_bins = table["decile"].max()
    table["percentile"] = table["decile"] * (100 / actual_bins)

    # 加入起點
    origin = pd.DataFrame([{"decile": 0, "events": 0, "cum_events": 0,
                            "cumulative_gain": 0.0, "percentile": 0.0}])
    table = pd.concat([origin, table], ignore_index=True)

    return table[["percentile", "cumulative_gain"]]
