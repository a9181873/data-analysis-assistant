"""
PSI (Population Stability Index) 模組 — 監控模型/變數穩定度。
"""

import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series,
                  bins: int = 10, method: str = "quantile") -> tuple:
    """
    計算單一變數的 PSI。

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Args:
        expected: 基準期資料 (如訓練集)
        actual: 監控期資料 (如新資料)
        bins: 分組數
        method: 'quantile' 或 'equal_width'

    Returns:
        (psi_value, psi_table DataFrame)
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # 用基準期的分位數做分箱
    if method == "quantile":
        _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates="drop")
    else:
        _, bin_edges = pd.cut(expected, bins=bins, retbins=True, duplicates="drop")

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_binned = pd.cut(expected, bins=bin_edges)
    actual_binned = pd.cut(actual, bins=bin_edges)

    expected_counts = expected_binned.value_counts(sort=False)
    actual_counts = actual_binned.value_counts(sort=False)

    # Laplace smoothing
    expected_pct = (expected_counts + 0.5) / (len(expected) + 0.5 * len(expected_counts))
    actual_pct = (actual_counts + 0.5) / (len(actual) + 0.5 * len(actual_counts))

    table = pd.DataFrame({
        "bin": expected_counts.index.astype(str),
        "expected_count": expected_counts.values,
        "actual_count": actual_counts.values,
        "expected_pct": expected_pct.values,
        "actual_pct": actual_pct.values,
    })

    table["psi_component"] = (table["actual_pct"] - table["expected_pct"]) * \
                              np.log(table["actual_pct"] / table["expected_pct"])

    psi_value = table["psi_component"].sum()
    return psi_value, table


def calculate_psi_report(df_expected: pd.DataFrame, df_actual: pd.DataFrame,
                         columns: list = None, bins: int = 10) -> pd.DataFrame:
    """
    計算多個變數的 PSI 報告。

    Returns:
        DataFrame with columns: variable, psi, stability
    """
    if columns is None:
        # 只取共同的數值欄位
        common_numeric = set(
            df_expected.select_dtypes(include=["number"]).columns
        ) & set(
            df_actual.select_dtypes(include=["number"]).columns
        )
        columns = sorted(common_numeric)

    rows = []
    for col in columns:
        try:
            psi_val, _ = calculate_psi(df_expected[col], df_actual[col], bins=bins)
            rows.append({"variable": col, "psi": round(psi_val, 4)})
        except Exception as e:
            rows.append({"variable": col, "psi": None, "error": str(e)})

    result = pd.DataFrame(rows)

    def _stability(psi):
        if psi is None:
            return "錯誤"
        if psi < 0.1:
            return "穩定"
        elif psi < 0.25:
            return "輕微偏移"
        else:
            return "顯著偏移"

    result["stability"] = result["psi"].apply(_stability)
    return result.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
