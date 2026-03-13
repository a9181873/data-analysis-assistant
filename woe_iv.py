"""
WOE/IV 分析模組 — Weight of Evidence & Information Value
用於信用風險建模的變數篩選與分箱。
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def auto_bin(series: pd.Series, y: pd.Series, n_bins: int = 10,
             method: str = "quantile"):
    """
    自動分箱。

    Args:
        series: 連續變數
        y: 二元目標變數 (0/1)
        n_bins: 分組數
        method: 'quantile' (等頻) / 'equal_width' (等寬) / 'tree' (決策樹最佳切點)

    Returns:
        binned Series (categorical)
    """
    clean_mask = series.notna() & y.notna()
    s = series[clean_mask]
    target = y[clean_mask]

    if method == "tree":
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_bins,
            random_state=42,
        )
        tree.fit(s.values.reshape(-1, 1), target)
        thresholds = sorted(set(tree.tree_.threshold[tree.tree_.threshold != -2]))
        bins = [-np.inf] + thresholds + [np.inf]
        binned = pd.cut(series, bins=bins, duplicates="drop")
    elif method == "equal_width":
        binned = pd.cut(series, bins=n_bins, duplicates="drop")
    else:  # quantile
        binned = pd.qcut(series, q=n_bins, duplicates="drop")

    return binned


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str,
                     n_bins: int = 10, method: str = "quantile"):
    """
    計算單一變數的 WOE 和 IV。

    Returns:
        (woe_table DataFrame, total_iv float)
    """
    s = df[feature].copy()
    y = df[target].copy()

    # 判斷是否為連續變數
    if s.dtype in ("object", "category") or s.nunique() <= n_bins:
        binned = s.astype(str)
    else:
        binned = auto_bin(s, y, n_bins=n_bins, method=method)

    temp = pd.DataFrame({"bin": binned, "target": y}).dropna()

    total_events = (temp["target"] == 1).sum()
    total_non_events = (temp["target"] == 0).sum()

    if total_events == 0 or total_non_events == 0:
        raise ValueError("目標變數只有一個類別，無法計算 WOE/IV。")

    grouped = temp.groupby("bin", observed=False)["target"].agg(
        count="size", events="sum"
    ).reset_index()

    grouped["non_events"] = grouped["count"] - grouped["events"]
    grouped["event_rate"] = grouped["events"] / grouped["count"]

    # Laplace smoothing 避免 ln(0)
    grouped["events_smooth"] = grouped["events"] + 0.5
    grouped["non_events_smooth"] = grouped["non_events"] + 0.5

    grouped["dist_events"] = grouped["events_smooth"] / (total_events + 0.5 * len(grouped))
    grouped["dist_non_events"] = grouped["non_events_smooth"] / (total_non_events + 0.5 * len(grouped))

    grouped["woe"] = np.log(grouped["dist_events"] / grouped["dist_non_events"])
    grouped["iv_component"] = (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]

    total_iv = grouped["iv_component"].sum()

    result = grouped[["bin", "count", "events", "non_events", "event_rate", "woe", "iv_component"]]
    return result, total_iv


def calculate_iv_table(df: pd.DataFrame, features: list, target: str,
                       n_bins: int = 10, method: str = "quantile"):
    """
    計算多個變數的 IV 並排名。

    Returns:
        DataFrame with columns: feature, IV, predictive_power
    """
    rows = []
    for feat in features:
        try:
            _, iv = calculate_woe_iv(df, feat, target, n_bins, method)
            rows.append({"feature": feat, "IV": round(iv, 4)})
        except Exception as e:
            rows.append({"feature": feat, "IV": 0.0, "error": str(e)})

    result = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)

    def _power(iv):
        if iv < 0.02:
            return "無預測力"
        elif iv < 0.1:
            return "弱"
        elif iv < 0.3:
            return "中等"
        elif iv < 0.5:
            return "強"
        else:
            return "極強 (需檢查)"

    result["predictive_power"] = result["IV"].apply(_power)
    return result
