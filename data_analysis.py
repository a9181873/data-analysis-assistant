"""
統計分析模組
包含敘述統計、假設檢定、迴歸分析、卡方檢定、ANOVA、相關分析。
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def descriptive_statistics(df):
    """計算數據框的敘述統計，返回 DataFrame 以供視覺化顯示。"""
    return df.describe(include='all').T.reset_index().rename(columns={"index": "欄位"})


def perform_ttest(df, column1, column2=None, alpha=0.05):
    """
    執行獨立樣本 t 檢定或單樣本 t 檢定。
    如果提供 column2，則執行獨立樣本 t 檢定。
    如果只提供 column1，則執行單樣本 t 檢定 (與0比較)。
    """
    if column1 not in df.columns:
        return f"錯誤：列 '{column1}' 不存在。"

    data1 = df[column1].dropna()

    if column2:
        if column2 not in df.columns:
            return f"錯誤：列 '{column2}' 不存在。"
        data2 = df[column2].dropna()
        if len(data1) < 2 or len(data2) < 2:
            return "錯誤：進行 t 檢定需要至少兩個有效數據點。"
        statistic, p_value = stats.ttest_ind(data1, data2)
        result = f"獨立樣本 t 檢定:\n  組1 ({column1}) 平均值: {data1.mean():.2f}\n  組2 ({column2}) 平均值: {data2.mean():.2f}\n  t 統計量: {statistic:.3f}\n  P 值: {p_value:.3f}"
        if p_value < alpha:
            result += f"\n  在顯著水準 {alpha} 下，拒絕虛無假設，兩組平均值存在顯著差異。"
        else:
            result += f"\n  在顯著水準 {alpha} 下，不拒絕虛無假設，兩組平均值無顯著差異。"
    else:
        if len(data1) < 2:
            return "錯誤：進行 t 檢定需要至少兩個有效數據點。"
        statistic, p_value = stats.ttest_1samp(data1, 0)
        result = f"單樣本 t 檢定 (與0比較):\n  樣本 ({column1}) 平均值: {data1.mean():.2f}\n  t 統計量: {statistic:.3f}\n  P 值: {p_value:.3f}"
        if p_value < alpha:
            result += f"\n  在顯著水準 {alpha} 下，拒絕虛無假設，樣本平均值與0存在顯著差異。"
        else:
            result += f"\n  在顯著水準 {alpha} 下，不拒絕虛無假設，樣本平均值與0無顯著差異。"
    return result


def perform_linear_regression(df, target_column, feature_columns):
    """執行多元線性迴歸。"""
    if target_column not in df.columns:
        return f"錯誤：目標列 '{target_column}' 不存在。"
    for col in feature_columns:
        if col not in df.columns:
            return f"錯誤：特徵列 '{col}' 不存在。"

    model_df = df[[target_column] + feature_columns].dropna()
    if model_df.empty:
        return "錯誤：選擇的列中包含缺失值或數據不足，無法進行迴歸分析。"

    X = model_df[feature_columns]
    y = model_df[target_column]

    if len(X) < 2 or len(y) < 2:
        return "錯誤：進行迴歸分析需要至少兩個有效數據點。"

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary().as_text()


# ============================================================
# 新增：分類型統計檢定
# ============================================================

def perform_chi_square_test(df, col1, col2, alpha=0.05):
    """
    執行卡方獨立性檢定 (Chi-Square Test of Independence)。
    檢驗兩個類別變數之間是否獨立。
    """
    if col1 not in df.columns:
        return f"錯誤：列 '{col1}' 不存在。"
    if col2 not in df.columns:
        return f"錯誤：列 '{col2}' 不存在。"

    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    result = f"卡方獨立性檢定:\n"
    result += f"  變數1: {col1}\n  變數2: {col2}\n"
    result += f"\n  列聯表:\n{contingency_table.to_string()}\n"
    result += f"\n  卡方統計量 (χ²): {chi2:.3f}\n"
    result += f"  自由度 (df): {dof}\n"
    result += f"  P 值: {p_value:.4f}\n"

    if p_value < alpha:
        result += f"\n  ✅ 在顯著水準 {alpha} 下，拒絕虛無假設。\n"
        result += f"  結論：{col1} 與 {col2} 之間存在顯著關聯。"
    else:
        result += f"\n  ❌ 在顯著水準 {alpha} 下，不拒絕虛無假設。\n"
        result += f"  結論：{col1} 與 {col2} 之間無顯著關聯。"

    # Cramér's V 效果量
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim > 0:
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        result += f"\n\n  Cramér's V (效果量): {cramers_v:.3f}"
        if cramers_v < 0.1:
            result += " (微弱關聯)"
        elif cramers_v < 0.3:
            result += " (弱關聯)"
        elif cramers_v < 0.5:
            result += " (中等關聯)"
        else:
            result += " (強關聯)"

    return result


def perform_anova(df, group_col, value_col, alpha=0.05):
    """
    執行單因子變異數分析 (One-Way ANOVA)。
    比較多個群組的平均值是否存在差異。
    """
    if group_col not in df.columns:
        return f"錯誤：分組列 '{group_col}' 不存在。"
    if value_col not in df.columns:
        return f"錯誤：數值列 '{value_col}' 不存在。"

    groups = df.groupby(group_col)[value_col].apply(lambda x: x.dropna().tolist())
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) < 2:
        return "錯誤：至少需要兩個有效群組（每組至少 2 個數據點）。"

    f_stat, p_value = stats.f_oneway(*groups)

    result = f"單因子變異數分析 (One-Way ANOVA):\n"
    result += f"  分組變數: {group_col}\n  數值變數: {value_col}\n"

    # 各組統計
    group_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std'])
    result += f"\n  各組統計:\n{group_stats.to_string()}\n"
    result += f"\n  F 統計量: {f_stat:.3f}\n"
    result += f"  P 值: {p_value:.4f}\n"

    if p_value < alpha:
        result += f"\n  ✅ 在顯著水準 {alpha} 下，拒絕虛無假設。\n"
        result += f"  結論：不同 {group_col} 群組的 {value_col} 平均值存在顯著差異。"

        # Tukey HSD 事後比較
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            clean_df = df[[group_col, value_col]].dropna()
            tukey = pairwise_tukeyhsd(clean_df[value_col], clean_df[group_col], alpha=alpha)
            result += f"\n\n  Tukey HSD 事後檢定:\n{tukey.summary()}"
        except Exception:
            pass
    else:
        result += f"\n  ❌ 在顯著水準 {alpha} 下，不拒絕虛無假設。\n"
        result += f"  結論：不同 {group_col} 群組的 {value_col} 平均值無顯著差異。"

    # 效果量 η²
    grand_mean = df[value_col].dropna().mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((val - grand_mean) ** 2 for g in groups for val in g)
    if ss_total > 0:
        eta_squared = ss_between / ss_total
        result += f"\n\n  η² (效果量): {eta_squared:.3f}"
        if eta_squared < 0.01:
            result += " (微弱效果)"
        elif eta_squared < 0.06:
            result += " (小效果)"
        elif eta_squared < 0.14:
            result += " (中效果)"
        else:
            result += " (大效果)"

    return result


def perform_correlation_analysis(df, columns=None, method='pearson'):
    """
    執行相關分析，計算相關矩陣並進行顯著性檢定。
    method: 'pearson', 'spearman', 'kendall'
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
    else:
        for col in columns:
            if col not in df.columns:
                return f"錯誤：列 '{col}' 不存在。"
        numeric_df = df[columns].select_dtypes(include=['number'])

    if numeric_df.shape[1] < 2:
        return "錯誤：至少需要兩個數值型欄位來計算相關性。"

    corr_matrix = numeric_df.corr(method=method)

    result = f"相關分析 ({method.capitalize()} 相關):\n"
    result += f"\n相關矩陣:\n{corr_matrix.round(3).to_string()}\n"

    # 顯著性檢定（成對）
    result += f"\n顯著性檢定 (P 值):\n"
    cols = numeric_df.columns.tolist()
    p_values = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            data_i = numeric_df[cols[i]].dropna()
            data_j = numeric_df[cols[j]].dropna()
            common_idx = data_i.index.intersection(data_j.index)
            if len(common_idx) >= 3:
                if method == 'pearson':
                    _, p = stats.pearsonr(data_i[common_idx], data_j[common_idx])
                elif method == 'spearman':
                    _, p = stats.spearmanr(data_i[common_idx], data_j[common_idx])
                else:
                    _, p = stats.kendalltau(data_i[common_idx], data_j[common_idx])
                p_values.loc[cols[i], cols[j]] = p
                p_values.loc[cols[j], cols[i]] = p

    result += p_values.round(4).to_string()

    # 解讀相關強度
    result += "\n\n相關強度參考:\n"
    result += "  |r| < 0.1: 無相關\n"
    result += "  0.1 ≤ |r| < 0.3: 弱相關\n"
    result += "  0.3 ≤ |r| < 0.5: 中等相關\n"
    result += "  0.5 ≤ |r| < 0.7: 強相關\n"
    result += "  |r| ≥ 0.7: 非常強相關"

    return result


if __name__ == '__main__':
    # 測試代碼
    data = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [10, 12, 11, 15, 14, 18, 16, 20, 19, 22],
        'C': [20, 25, 22, 28, 26, 30, 28, 35, 32, 38],
        'Group': ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z'],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    }
    df = pd.DataFrame(data)

    print("=== 敘述統計 ===")
    print(descriptive_statistics(df))

    print("\n=== t 檢定 ===")
    print(perform_ttest(df, 'B', 'C'))

    print("\n=== 線性迴歸 ===")
    print(perform_linear_regression(df, 'B', ['A', 'C']))

    print("\n=== 卡方檢定 ===")
    print(perform_chi_square_test(df, 'Group', 'Gender'))

    print("\n=== ANOVA ===")
    print(perform_anova(df, 'Group', 'B'))

    print("\n=== 相關分析 ===")
    print(perform_correlation_analysis(df, ['A', 'B', 'C']))
