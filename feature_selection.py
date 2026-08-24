"""
特徵篩選模組 (Feature Selection Suite)
────────────────────────────────────────
複刻實戰驗證的「變數海選」流程，提供四種互補的篩選方法：

1. univariate_screen()      單變量統計篩選（f_classif / 卡方 / f_regression）
2. correlated_pairs_prune() 高相關配對擇優（|r| > 門檻時依樹模型重要性保留較優者）
3. rfe_select()             RFE 遞歸特徵消除
4. aic_stepwise()           AIC 逐步選擇（貪婪法，避免暴力窮舉的組合爆炸）

以及一鍵跑完整套件的 run_full_suite()。
"""

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.feature_selection import f_classif, f_regression, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── IV 判讀門檻（信用評分實務標準）──
IV_KEEP_MIN = 0.02      # 低於此值 → 無預測力，建議剔除
IV_LEAK_ALERT = 0.5     # 高於此值 → 疑似目標洩漏，需人工確認


# ═══════════════════════════════════════════════
# 1. 單變量統計篩選
# ═══════════════════════════════════════════════
def univariate_screen(df: pd.DataFrame, target: str,
                      features=None, task_type="classification",
                      alpha=0.05, max_cat_cardinality=20):
    """
    對所有特徵做單變量顯著性檢定：
    - 數值特徵（分類任務）→ f_classif（ANOVA F 檢定）
    - 數值特徵（迴歸任務）→ f_regression
    - 類別特徵（分類任務）→ 卡方檢定（需先編碼；高基數欄自動跳過）

    Returns:
        DataFrame: feature, method, statistic, p_value, significant
    """
    if features is None:
        features = [c for c in df.columns if c != target]
    y = df[target]

    num_feats = [f for f in features
                 if pd.api.types.is_numeric_dtype(df[f])]
    cat_feats = [f for f in features
                 if not pd.api.types.is_numeric_dtype(df[f])
                 and df[f].nunique() <= max_cat_cardinality]

    rows = []

    if num_feats:
        X_num = df[num_feats].fillna(df[num_feats].median())
        if task_type == "classification":
            stat_arr, p_arr = f_classif(X_num, y)
        else:
            stat_arr, p_arr = f_regression(X_num, y)
        for f, s, p in zip(num_feats, stat_arr, p_arr):
            rows.append({"feature": f, "method": "ANOVA-F" if task_type == "classification" else "F-regression",
                         "statistic": round(float(s), 4),
                         "p_value": float(p),
                         "significant": bool(p < alpha)})

    for f in cat_feats:
        try:
            ct = pd.crosstab(df[f].fillna("(missing)"), y)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            # 期望次數 <5 的格數過多時提示（卡方前提）
            chi2_stat = stats.chi2_contingency(ct)
            expected = chi2_stat[3]
            low_exp_ratio = float((expected < 5).mean())
            rows.append({
                "feature": f, "method": "Chi-square",
                "statistic": round(float(chi2_stat[0]), 4),
                "p_value": float(chi2_stat[1]),
                "significant": bool(chi2_stat[1] < alpha),
                "_low_expected_ratio": round(low_exp_ratio, 3),
            })
        except Exception:
            continue

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("p_value").reset_index(drop=True)
    return result


# ═══════════════════════════════════════════════
# 2. WOE/IV 預測力排名（複用 woe_iv.py）
# ═══════════════════════════════════════════════
def iv_ranking(df: pd.DataFrame, target: str, features=None,
               n_bins=10, method="quantile"):
    """
    計算所有變數的 IV 並排名（僅適用二元分類目標）。
    回傳 (排名表, 建議剔除清單, 洩漏警示清單)。
    """
    from woe_iv import calculate_iv_table

    if features is None:
        features = [c for c in df.columns if c != target]
    iv_table = calculate_iv_table(df, features, target,
                                  n_bins=n_bins, method=method)
    drop_list = iv_table.loc[iv_table["IV"] < IV_KEEP_MIN, "feature"].tolist()
    leak_list = iv_table.loc[iv_table["IV"] >= IV_LEAK_ALERT, "feature"].tolist()
    return iv_table, drop_list, leak_list


# ═══════════════════════════════════════════════
# 3. 高相關配對擇優
# ═══════════════════════════════════════════════
def correlated_pairs_prune(df: pd.DataFrame, target: str, features=None,
                           corr_threshold=0.9, importance_model=None):
    """
    找出 |r| > corr_threshold 的特徵配對，
    依樹模型重要性保留每對中較優者。

    對應人工筆記 Feature_select_model 的核心邏輯：
    相關矩陣 → 配對去重 → XGBoost/RF 重要性 → 重要性差異比較。

    Returns:
        dict: {
            "pairs": DataFrame(val1, val2, corr, importance_1, importance_2),
            "drop_suggestions": [被建議剔除的變數],
            "importance": Series(全特徵重要性),
        }
    """
    if features is None:
        features = [c for c in df.columns if c != target]

    model_df = df[features + [target]].copy()
    # 類別欄位快速編碼以便計算相關與重要性
    for c in features:
        if not pd.api.types.is_numeric_dtype(model_df[c]):
            model_df[c] = model_df[c].astype("category").cat.codes
    model_df = model_df.dropna()

    X, y = model_df[features], model_df[target]

    # 相關矩陣 → 取上三角配對
    corr = X.corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    seen = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if r >= corr_threshold:
                a, b = cols[i], cols[j]
                key = frozenset((a, b))
                if key not in seen:
                    seen.add(key)
                    pairs.append((a, b, round(float(r), 4)))

    if not pairs:
        return {"pairs": pd.DataFrame(),
                "drop_suggestions": [],
                "importance": pd.Series(dtype=float)}

    # 樹模型重要性（預設 RF，速度快且穩健）
    if importance_model is None:
        is_clf = y.nunique() <= 20 and not pd.api.types.is_float_dtype(y)
        if is_clf:
            importance_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor
            importance_model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1)
    try:
        importance_model.fit(X, y)
        importance = pd.Series(importance_model.feature_importances_,
                               index=features)
    except Exception:
        importance = pd.Series(0.0, index=features)

    pair_rows = []
    drop_set = set()
    for a, b, r in pairs:
        imp_a = float(importance.get(a, 0))
        imp_b = float(importance.get(b, 0))
        loser = b if imp_a >= imp_b else a
        drop_set.add(loser)
        pair_rows.append({
            "val1": a, "importance_1": round(imp_a, 4),
            "val2": b, "importance_2": round(imp_b, 4),
            "corr": r,
            "建議剔除": loser,
        })

    pairs_df = pd.DataFrame(pair_rows).sort_values("corr", ascending=False)
    return {"pairs": pairs_df,
            "drop_suggestions": sorted(drop_set),
            "importance": importance.sort_values(ascending=False)}


# ═══════════════════════════════════════════════
# 4. RFE 遞歸特徵消除
# ═══════════════════════════════════════════════
def rfe_select(df: pd.DataFrame, target: str, features=None,
               n_features_to_select=10, task_type="classification"):
    """
    以 LogisticRegression（balanced）為基學習器執行 RFE。
    對應人工筆記的 Recursive Feature Elimination 區塊。
    """
    if features is None:
        features = [c for c in df.columns if c != target]

    model_df = df[features + [target]].copy()
    for c in features:
        if not pd.api.types.is_numeric_dtype(model_df[c]):
            model_df[c] = model_df[c].astype("category").cat.codes
    model_df = model_df.dropna()

    X, y = model_df[features], model_df[target]
    n_select = min(n_features_to_select, len(features))

    base = (LogisticRegression(max_iter=1000, class_weight="balanced")
            if task_type == "classification"
            else _linear_baseline())

    selector = RFE(base, n_features_to_select=n_select, step=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X, y)

    ranking = pd.Series(selector.ranking_, index=features).sort_values()
    selected = [f for f in features if selector.support_[list(features).index(f)]]
    return selected, ranking


def _linear_baseline():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


# ═══════════════════════════════════════════════
# 5. AIC 逐步選擇（貪婪法）
# ═══════════════════════════════════════════════
def _aic_of(X: pd.DataFrame, y: pd.Series) -> float:
    """以 OLS 計算 AIC（與人工筆記一致：sm.OLS(y, add_constant(X)).fit().aic）。"""
    import statsmodels.api as sm
    X_const = sm.add_constant(X)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y.astype(float), X_const.astype(float)).fit()
        return float(res.aic)
    except Exception:
        return np.inf


def aic_stepwise(df: pd.DataFrame, target: str, features=None,
                 direction="forward", max_steps=30):
    """
    AIC 逐步選擇。使用貪婪法（每次加入/剔除使 AIC 最小的變數），
    避免 itertools 全組合窮舉的組合爆炸（人工筆記中 15 變數跑了 40 分鐘；
    貪婪法同規模 < 5 秒）。

    direction: "forward"（逐步加入）或 "both"（加入後允許回剔）
    """
    if features is None:
        features = [c for c in df.columns if c != target]

    model_df = df[features + [target]].copy()
    for c in features:
        if not pd.api.types.is_numeric_dtype(model_df[c]):
            model_df[c] = model_df[c].astype("category").cat.codes
    model_df = model_df.dropna()

    X_all, y = model_df[features], model_df[target]

    included = []
    current_aic = _aic_of(X_all[included] if included else X_all[[]], y)
    trace = [(tuple(), round(current_aic, 2))]

    for _ in range(min(max_steps, len(features))):
        best_candidate, best_aic = None, current_aic

        # Forward：嘗試加入每個未入選變數
        candidates = [f for f in features if f not in included]
        for cand in candidates:
            trial_aic = _aic_of(X_all[included + [cand]], y)
            if trial_aic < best_aic - 1e-6:
                best_candidate, best_aic = cand, trial_aic

        # Backward（direction="both"）：嘗試剔除已入選變數
        removed_candidate = None
        if direction == "both" and len(included) > 1:
            for excl in list(included):
                trial_set = [f for f in included if f != excl]
                trial_aic = _aic_of(X_all[trial_set], y)
                if trial_aic < best_aic - 1e-6:
                    best_aic = trial_aic
                    removed_candidate = excl
                    best_candidate = None

        if removed_candidate:
            included.remove(removed_candidate)
            current_aic = best_aic
            trace.append((tuple(included), round(current_aic, 2)))
        elif best_candidate:
            included.append(best_candidate)
            current_aic = best_aic
            trace.append((tuple(included), round(current_aic, 2)))
        else:
            break  # 無法再改善 → 收斂

    return included, trace


# ═══════════════════════════════════════════════
# 一鍵完整海選
# ═══════════════════════════════════════════════
def run_full_suite(df: pd.DataFrame, target: str,
                   numeric_features=None, categorical_features=None,
                   task_type="classification",
                   corr_threshold=0.9, rfe_n=8):
    """
    執行完整海選流程並彙整報告：
    Step 1  單變量篩選（p>alpha 剔除）
    Step 2  高相關配對擇優
    Step 3  RFE 收斂至精簡集
    Step 4  （分類+二元目標）AIC 逐步交叉驗證

    Returns:
        dict 含各步驟結果與最終建議特徵集
    """
    t0 = time.time()
    all_feats = []
    if numeric_features:
        all_feats += list(numeric_features)
    if categorical_features:
        all_feats += list(categorical_features)
    if not all_feats:
        all_feats = [c for c in df.columns if c != target]
    all_feats = [f for f in dict.fromkeys(all_feats) if f != target]

    # ── Step 1: 單變量 ──
    uni = univariate_screen(df, target, all_feats, task_type)
    sig_feats = uni.loc[uni["significant"], "feature"].tolist() if not uni.empty else all_feats
    dropped_uni = [f for f in all_feats if f not in sig_feats]
    # 保底：全部不顯著時不至於空集合
    if not sig_feats:
        sig_feats = all_feats[:min(5, len(all_feats))]

    # ── Step 2: 高相關配對擇優（僅數值欄可算相關）──
    numeric_only = [f for f in sig_feats if pd.api.types.is_numeric_dtype(df[f])]
    prune = {"pairs": pd.DataFrame(), "drop_suggestions": [], "importance": pd.Series(dtype=float)}
    if len(numeric_only) >= 2:
        try:
            prune = correlated_pairs_prune(df, target, numeric_only, corr_threshold)
        except Exception:
            pass
    after_corr = [f for f in sig_feats if f not in prune["drop_suggestions"]]

    # ── Step 3: RFE ──
    rfe_selected, rfe_ranking = None, None
    if len(after_corr) > rfe_n:
        try:
            rfe_selected, rfe_ranking = rfe_select(
                df, target, after_corr, n_features_to_select=rfe_n, task_type=task_type)
        except Exception:
            rfe_selected = after_corr[:rfe_n]
    else:
        rfe_selected = after_corr

    # ── Step 4: AIC 逐步（僅分類任務且樣本可控時）──
    aic_selected, aic_trace = None, None
    if task_type == "classification" and len(rfe_selected) <= 15 and len(rfe_selected) >= 3:
        try:
            sample_df = df
            if len(df) > 20000:  # 大表抽樣加速 OLS
                sample_df = df.sample(20000, random_state=42)
            aic_selected, aic_trace = aic_stepwise(
                sample_df, target, rfe_selected, direction="both")
        except Exception:
            pass

    elapsed = round(time.time() - t0, 1)

    report = {
        "elapsed_sec": elapsed,
        "univariate_table": uni,
        "dropped_by_univariate": dropped_uni,
        "correlation_pairs": prune["pairs"],
        "dropped_by_correlation": prune["drop_suggestions"],
        "rfe_selected": rfe_selected,
        "rfe_ranking": rfe_ranking,
        "aic_selected": aic_selected,
        "aic_trace": aic_trace,
        "final_features": aic_selected if aic_selected else rfe_selected,
        "n_original": len(all_feats),
        "n_final": len(aic_selected if aic_selected else rfe_selected),
    }
    return report


def format_suite_report(report: dict) -> str:
    """將 run_full_suite 結果轉為 Markdown 報告。"""
    lines = ["## 🔍 特徵海選結果", ""]
    lines.append(f"> 流程耗時 **{report['elapsed_sec']}** 秒｜"
                 f"原始 {report['n_original']} 個變數 → 建議保留 **{report['n_final']}** 個")

    uni = report["univariate_table"]
    if not uni.empty:
        lines.append("### Step 1｜單變量統計篩選")
        top = uni.head(12)[["feature", "method", "statistic", "p_value", "significant"]]
        lines.append(top.to_markdown(index=False))
        dropped = report["dropped_by_univariate"]
        if dropped:
            lines.append(f"\n❌ 不顯著（p>0.05）建議剔除：{', '.join(f'`{d}`' for d in dropped)}")
        lines.append("")

    pairs = report["correlation_pairs"]
    if isinstance(pairs, pd.DataFrame) and not pairs.empty:
        lines.append("### Step 2｜高相關配對擇優")
        lines.append(pairs.to_markdown(index=False))
        lines.append("")

    if report.get("rfe_ranking") is not None:
        lines.append("### Step 3｜RFE 排名")
        rk = report["rfe_ranking"].head(15).reset_index()
        rk.columns = ["feature", "rank"]
        lines.append(rk.to_markdown(index=False))
        lines.append("")

    if report.get("aic_trace"):
        lines.append("### Step 4｜AIC 逐步選擇軌跡")
        for feats, aic in report["aic_trace"]:
            label = ", ".join(feats) if feats else "（空模型）"
            lines.append(f"- AIC {aic} ← [{label}]")
        lines.append("")

    final = report["final_features"]
    lines.append("---")
    lines.append(f"### ✅ 最終建議特徵集（{len(final)} 個）\n"
                 + ", ".join(f"`{f}`" for f in final))
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    test_df = pd.DataFrame({
        "x1": x1,                       # 有訊號
        "x1_copy": x1 * 1.01 + rng.normal(0, 0.001, n),   # 與 x1 幾乎共線
        "noise1": rng.normal(0, 1, n),  # 噪聲
        "noise2": rng.normal(0, 1, n),
        "cat": rng.choice(["A", "B", "C"], n),
        "target": (x1 + rng.normal(0, 0.5, n) > 0).astype(int),
    })
    rep = run_full_suite(test_df, "target",
                         numeric_features=["x1", "x1_copy", "noise1", "noise2"],
                         categorical_features=["cat"])
    print(format_suite_report(rep))
