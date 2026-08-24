"""
統計分析模組
包含敘述統計、假設檢定、迴歸分析、卡方檢定、ANOVA、相關分析。
含自動前置檢查：常態性 (Shapiro-Wilk)、變異數同質性 (Levene)，
不合假設時自動改用無母數方法並說明原因。
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


# ============================================================
# 前提假設檢查與無母數檢定（P1 統計嚴謹性補強）
# ============================================================

def check_normality(data, alpha=0.05):
    """
    Shapiro-Wilk 常態性檢定。樣本 >5000 時自動抽樣（Shapiro 統計量在
    大樣本下過度敏感，微小偏離即拒絕，參考 scipy 官方建議 N<5000）。

    Returns:
        (is_normal: bool, p_value: float, n_used: int)
    """
    clean = pd.Series(data).dropna()
    if len(clean) < 3:
        return True, np.nan, len(clean)
    if len(clean) > 5000:
        clean = clean.sample(5000, random_state=42)
    try:
        stat, p = stats.shapiro(clean)
        return bool(p >= alpha), float(p), len(clean)
    except Exception:
        return True, np.nan, len(clean)


def check_variance_homogeneity(groups, alpha=0.05):
    """
    Levene 變異數同質性檢定（Brown-Forsythe 中心化，對偏態穩健）。

    Returns:
        (equal_var: bool, p_value: float)
    """
    groups = [pd.Series(g).dropna() for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return True, np.nan
    try:
        stat, p = stats.levene(*groups, center='median')
        return bool(p >= alpha), float(p)
    except Exception:
        return True, np.nan


def perform_mannwhitney(df, column1, column2=None, group_col=None, alpha=0.05):
    """
    Mann-Whitney U 檢定（無母數）：比較兩組中位數差異。
    適用：資料非常態或為序位尺度。
    可用兩種呼叫方式：
    - perform_mannwhitney(df, col1, col2)          兩獨立樣本
    - perform_mannwhitney(df, value_col, group_col=g)  依分組欄拆兩組
    """
    if group_col is not None:
        vals = df[[column1, group_col]].dropna()
        levels = vals[group_col].unique()
        if len(levels) != 2:
            return f"錯誤：Mann-Whitney U 需要恰好 2 組，實際 {len(levels)} 組。"
        data1 = vals.loc[vals[group_col] == levels[0], column1]
        data2 = vals.loc[vals[group_col] == levels[1], column2 or column1] if False else \
                vals.loc[vals[group_col] == levels[1], column1]
    else:
        data1 = df[column1].dropna()
        data2 = df[column2].dropna() if column2 else None

    if data2 is None:
        return "錯誤：需要第二組資料。"

    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    result = (
        f"Mann-Whitney U 檢定（無母數，比較中位數）:\n"
        f"  組1 ({column1}) 中位數: {data1.median():.4f} (n={len(data1)})\n"
        f"  組2 中位數: {data2.median():.4f} (n={len(data2)})\n"
        f"  U 統計量: {u_stat:.3f}\n  P 值: {p_value:.4g}\n"
    )
    result += ("  ✅ 兩組分布存在顯著差異。" if p_value < alpha
               else "  ❌ 未達顯著水準，無法拒絕「兩組分布相同」的虛無假設。")
    # 效果量 rank-biserial correlation
    n1, n2 = len(data1), len(data2)
    rbc = 1 - (2 * u_stat) / (n1 * n2)
    result += f"\n  效果量 (rank-biserial r): {rbc:.3f}"
    return result


def perform_kruskal(df, group_col, value_col, alpha=0.05):
    """
    Kruskal-Wallis H 檢定（無母數 ANOVA）：比較多組中位數差異。
    適用：非常態、變異數不同質的多組比較。
    """
    if group_col not in df.columns or value_col not in df.columns:
        return "錯誤：欄位不存在。"

    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return "錯誤：至少需要兩個有效群組。"

    h_stat, p_value = stats.kruskal(*groups)
    k = len(groups)
    n_total = sum(len(g) for g in groups)

    result = (
        f"Kruskal-Wallis H 檢定（無母數多組比較）:\n"
        f"  分組變數: {group_col}（{k} 組）｜數值變數: {value_col}\n"
        f"  H 統計量: {h_stat:.3f}\n  P 值: {p_value:.4g}\n"
    )
    if p_value < alpha:
        result += f"  ✅ 各組分布存在顯著差異。\n\n  各組中位數:\n"
        medians = df.groupby(group_col)[value_col].median()
        result += medians.to_string()
        # Dunn 型事後成對比較（Bonferroni 校正的 Mann-Whitney）
        from itertools import combinations
        result += "\n\n  成對事後檢定（Mann-Whitney U + Bonferroni 校正）:\n"
        pairs = list(combinations(range(k), 2))
        corrected_alpha = alpha / max(len(pairs), 1)
        level_names = [name for name, _ in df.groupby(group_col)]
        for i, j in pairs:
            _, p_pair = stats.mannwhitneyu(groups[i], groups[j])
            sig = "**" if p_pair < corrected_alpha else ""
            result += f"    {level_names[i]} vs {level_names[j]}: P={p_pair:.4g} {sig}\n"
        result += f"    （校正後 α = {corrected_alpha:.4f}）"
    else:
        result += "  ❌ 各組間未達顯著差異。"
    return result


def compute_vif(df, columns=None):
    """
    計算變數間的 VIF（變異數膨脹因子），診斷多元共線性。
    判讀：VIF>10 嚴重共線、VIF>5 需留意、VIF<5 可接受。

    Returns:
        DataFrame(feature, VIF)，依 VIF 降序。
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[columns].dropna()
    if X.shape[0] < len(columns) + 1 or X.shape[1] < 2:
        return pd.DataFrame({"feature": columns,
                             "VIF": [np.nan] * len(columns)})

    # 去除零變異欄避免奇異矩陣
    keep = [c for c in X.columns if X[c].std() > 0]
    X = X[keep]

    vif_rows = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.nan
        vif_rows.append({"feature": col, "VIF": round(float(v), 3)})
    result = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    return result


def descriptive_statistics(df):
    """計算數據框的敘述統計，返回 DataFrame 以供視覺化顯示。"""
    return df.describe(include='all').T.reset_index().rename(columns={"index": "欄位"})


def perform_ttest(df, column1, column2=None, alpha=0.05, auto_check=True):
    """
    執行獨立樣本 t 檢定或單樣本 t 檢定，含自動前提檢查：
    1. Shapiro-Wilk 常態性 → 不常態時改用 Mann-Whitney U（無母數）
    2. Levene 變異數同質性 → 不同質時採用 Welch's t（校正自由度）
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

        # ── 自動前提檢查 ──
        prelude = ""
        use_method = "student"
        if auto_check:
            normal1, p_norm1, n1 = check_normality(data1, alpha)
            normal2, p_norm2, n2 = check_normality(data2, alpha)
            equal_var, p_levene = check_variance_homogeneity([data1, data2], alpha)

            if not (normal1 and normal2):
                prelude = (
                    f"⚠️ **前提檢查**：常態性檢定未通過"
                    f"（Shapiro P={p_norm1:.4g} / {p_norm2:.4g}）→ "
                    f"自動改用 **Mann-Whitney U 無母數檢定**。\n\n"
                )
                mw = perform_mannwhitney(df, column1, column2, alpha=alpha)
                return prelude + mw
            if not equal_var:
                use_method = "welch"
                prelude = (
                    f"ℹ️ **前提檢查**：Levene 變異數同質性未通過（P={p_levene:.4g}）→ "
                    f"採用 **Welch's t 檢定**（不假設變異數相等）。\n"
                )

        statistic, p_value = stats.ttest_ind(
            data1, data2, equal_var=(use_method == "student"))
        method_label = ("Welch's t 檢定" if use_method == "welch"
                        else "獨立樣本 t 檢定")
        result = prelude + (
            f"{method_label}:\n"
            f"  組1 ({column1}) 平均值: {data1.mean():.2f}\n"
            f"  組2 ({column2}) 平均值: {data2.mean():.2f}\n"
            f"  t 統計量: {statistic:.3f}\n  P 值: {p_value:.3f}"
        )
        # Cohen's d 效果量
        pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2)
                             / (len(data1)+len(data2)-2))
        if pooled_std > 0:
            cohens_d = abs(data1.mean() - data2.mean()) / pooled_std
            d_label = ("微小" if cohens_d < 0.2 else "小" if cohens_d < 0.5
                       else "中等" if cohens_d < 0.8 else "大")
            result += f"\n  效果量 Cohen's d: {cohens_d:.3f}（{d_label}效果）"

        if p_value < alpha:
            result += f"\n  在顯著水準 {alpha} 下，拒絕虛無假設，兩組平均值存在顯著差異。"
        else:
            result += f"\n  在顯著水準 {alpha} 下，不拒絕虛無假設，兩組平均值無顯著差異。"
        return result
    else:
        if len(data1) < 2:
            return "錯誤：進行 t 檢定需要至少兩個有效數據點。"

        prelude = ""
        if auto_check:
            normal1, p_norm1, _ = check_normality(data1, alpha)
            if not normal1:
                prelude = (f"⚠️ **前提檢查**：常態性檢定未通過（Shapiro P={p_norm1:.4g}）→ "
                           f"t 檢定結果僅供參考，建議搭配 Wilcoxon 符號等級檢定。\n")

        statistic, p_value = stats.ttest_1samp(data1, 0)
        result = prelude + (
            f"單樣本 t 檢定 (與0比較):\n"
            f"  樣本 ({column1}) 平均值: {data1.mean():.2f}\n"
            f"  t 統計量: {statistic:.3f}\n  P 值: {p_value:.3f}"
        )
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


def perform_anova(df, group_col, value_col, alpha=0.05, auto_check=True):
    """
    執行單因子變異數分析 (One-Way ANOVA)，含自動前提檢查：
    1. 各組 Shapiro-Wilk 常態性 → 多數組不常態時改用 Kruskal-Wallis（無母數）
    2. Levene 變異數同質性 → 不同質時提示（ANOVA 對大樣本穩健，僅警示）
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

    # ── 自動前提檢查 ──
    prelude = ""
    if auto_check and len(groups) <= 20:  # 組數過多時跳過逐一常態檢定
        normal_flags = [check_normality(g, alpha)[0] for g in groups]
        n_not_normal = sum(1 for f in normal_flags if not f)
        if n_not_normal > len(groups) / 2:
            prelude = (
                f"⚠️ **前提檢查**：{n_not_normal}/{len(groups)} 組常態性未通過"
                f"（Shapiro-Wilk）→ 自動改用 **Kruskal-Wallis 無母數檢定**。\n\n"
            )
            return prelude + perform_kruskal(df, group_col, value_col, alpha)
        equal_var, p_levene = check_variance_homogeneity(groups, alpha)
        if not equal_var:
            prelude = (
                f"ℹ️ **前提檢查**：Levene 變異數同質性未通過（P={p_levene:.4g}）。"
                f"ANOVA 在各組樣本數相近時仍屬穩健；若組數懸殊建議改用 "
                f"Welch's ANOVA 或 Kruskal-Wallis。\n\n"
            )

    f_stat, p_value = stats.f_oneway(*groups)

    f_stat, p_value = stats.f_oneway(*groups)

    result = prelude + f"單因子變異數分析 (One-Way ANOVA):\n"
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
