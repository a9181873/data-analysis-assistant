"""
AI 分析建議引擎 (Analysis Advisor Engine)
─────────────────────────────────────────
基於 profiler.py 的剖析結果，以「規則引擎」產生具體、可執行的分析建議。

設計原則（對應實戰驗證的人工流程）：
1. 建議正確性由程式端規則保證 —— 引用真實統計數字當證據。
2. 每條建議附帶可直接點擊執行的 action（按鈕參數）。
3. LLM 只負責把規則產出的報告潤飾成顧問語氣，不負責決策。

三大類建議：
① 遺漏值處理（逐欄策略，複刻人工逐欄判斷）
② 演算法建模策略（目標偵測 → 不平衡 → 模型路徑）
③ 變數篩選與分箱（IV 表 → 相關配對 → 分箱法選擇）
"""

import pandas as pd
import numpy as np

from profiler import (
    build_data_profile,
    HIGH_MISSING_PCT,
    SKEW_THRESHOLD,
    IMBALANCE_THRESHOLD,
)

# ═══════════════════════════════════════════════
# 資料結構
# ═══════════════════════════════════════════════
# Recommendation = {
#   "category": "missing" | "modeling" | "feature",
#   "title": str,
#   "reason": str,          # 引用真實數字的說明
#   "actions": [ {label, module, params}, ... ]
# }


def _mk_action(label, module, **params):
    return {"label": label, "module": module, "params": params}


# ═══════════════════════════════════════════════
# ① 遺漏值處理建議
# ═══════════════════════════════════════════════
def recommend_missing_handling(df: pd.DataFrame, profile: dict) -> list:
    """
    逐欄檢視缺失情況，產生欄位級策略建議。
    策略邏輯（複刻實戰流程）：
    - 數值 + 高偏態或離群多 → 中位數補值（均值會被極端值拉偏）
    - 數值 + 分布近常態     → 平均數補值
    - 類別                  → 眾數補值 或 新增 "Missing" 類別
    - 缺失率 > 50%          → 建議整欄捨棄（或加缺失旗標）
    - 全域：完全重複列      → 建議刪除
    """
    recs = []
    col_strategies = {}
    fill_lines = []      # 逐欄補值策略的說明行
    drop_lines = []      # 高缺失捨棄的說明行（與補值分開，避免報告重複）
    drop_cols = []

    for name, p in profile["columns"].items():
        if p.get("error"):
            continue
        miss_pct = p.get("missing_pct", 0)
        if miss_pct <= 0:
            continue

        role = p.get("role")
        if miss_pct > HIGH_MISSING_PCT:
            drop_cols.append(name)
            drop_lines.append(
                f"- `{name}`：缺失率 **{miss_pct:.0%}** 過高，建議整欄捨棄（保留資訊量太低）"
            )
            continue

        if role in ("numeric", "numeric_discrete"):
            skew = abs(p.get("skewness", 0) or 0)
            outlier_pct = p.get("outlier_pct", 0) or 0
            if skew > SKEW_THRESHOLD or outlier_pct > 0.05:
                strategy = "median"
                reason = f"偏態 {p['skewness']}、離群 {outlier_pct:.1%} → 用中位數避免極端值干擾"
            else:
                strategy = "mean"
                reason = f"偏態僅 {p['skewness']}、分布近常態 → 平均數即可"
            col_strategies[name] = {"strategy": strategy}
            fill_lines.append(f"- `{name}`（缺失 {miss_pct:.0%}）：{strategy} 補值。{reason}")
        else:
            tops = p.get("top_values", [])
            # 眾數佔比過半才用眾數，否則視缺失為獨立狀態
            if tops and tops[0].get("pct", 0) > 0.5:
                col_strategies[name] = {"strategy": "mode"}
                fill_lines.append(
                    f"- `{name}`（缺失 {miss_pct:.0%}）：眾數補值（`{tops[0]['value']}` 已佔 "
                    f"{tops[0]['pct']:.0%}）"
                )
            else:
                col_strategies[name] = {"strategy": "fill_value", "fill_value": "Missing"}
                fill_lines.append(
                    f"- `{name}`（缺失 {miss_pct:.0%}）：類別分布分散，建議填入獨立類別 "
                    f"`Missing`（缺失本身可能帶有資訊）"
                )

    if drop_cols:
        recs.append({
            "category": "missing",
            "title": f"高缺失欄位（>{int(HIGH_MISSING_PCT*100)}%）",
            "reason": "\n".join(drop_lines),
            "actions": [_mk_action(
                f"🗑️ 捨棄 {len(drop_cols)} 個高缺失欄位",
                "drop_columns",
                columns=drop_cols,
            )],
        })

    if col_strategies:
        recs.append({
            "category": "missing",
            "title": "遺漏值逐欄補值策略",
            "reason": "\n".join(fill_lines),
            "actions": [_mk_action(
                f"🛠️ 一鍵套用 {len(col_strategies)} 欄補值策略",
                "apply_missing_strategy",
                column_strategies=col_strategies,
            )],
        })

    if profile.get("duplicate_rows", 0) > 0:
        recs.append({
            "category": "missing",
            "title": "重複列清理",
            "reason": f"發現 **{profile['duplicate_rows']:,}** 列完全重複，建模前應移除以免資料洩漏。",
            "actions": [_mk_action("🧹 移除重複列", "drop_duplicates")],
        })

    return recs


# ═══════════════════════════════════════════════
# ② 演算法 / 建模策略建議
# ═══════════════════════════════════════════════
DEFAULT_CANDIDATES_CLF = ["Logistic Regression (邏輯迴歸)", "Random Forest (隨機森林)",
                          "XGBoost"]
DEFAULT_CANDIDATES_REG = ["Linear Regression (線性迴歸)", "Random Forest Regressor (隨機森林迴歸)",
                          "XGBoost Regressor"]


def recommend_modeling_strategy(df: pd.DataFrame, profile: dict) -> list:
    """
    依目標型別 + 樣本數 + 不平衡程度推導模型路徑。
    對應實戰案例：0.49% 事件率的理賠資料 → class_weight balanced / 抽樣平衡。
    """
    recs = []
    targets = profile.get("candidate_targets", [])
    n_rows = profile["n_rows"]

    if not targets:
        recs.append({
            "category": "modeling",
            "title": "尚未偵測到明確的預測目標",
            "reason": ("未找到二元欄位。若要做迴歸預測請指定連續目標欄；"
                       "若是分群任務可直接使用機器學習面板的分群功能。"),
            "actions": [],
        })
        return recs

    target = targets[0]
    tgt_name = target["column"]
    event_rate = target["event_rate"]
    minority = target["minority_class"]

    # ── 不平衡處理建議 ──
    balance_lines = []
    balance_actions = []
    if event_rate < IMBALANCE_THRESHOLD:
        ratio = int(round(1 / max(event_rate, 1e-9)))
        # 欠採樣至 1:10：以少數類筆數 ×10 為目標，上限為多數類實際筆數
        minority_n = min(target["counts"].values())
        majority_n = max(target["counts"].values())
        sample_n = min(majority_n, int(round(minority_n * 10)))
        balance_lines.append(
            f"目標 `{tgt_name}` 少數類 `{minority}` 僅佔 **{event_rate:.2%}**（約 1:{ratio}），屬高度不平衡。\n"
            f"- 若用演算法內建：`class_weight='balanced'`\n"
            f"- 若要抽樣平衡：建議欠採樣至 1:10（非事件抽樣 {sample_n:,} 筆）\n"
            f"- ⚠️ 評估指標應以 **AUC / F1 / Recall** 為主，Accuracy 會失真"
        )
        balance_actions.append(_mk_action(
            "⚖️ 以 class_weight='balanced' 建模", "ml_compare",
            task_type="classification", target_col=tgt_name,
            balancing="class_weight",
        ))
        balance_actions.append(_mk_action(
            "📉 欠採樣至 1:10 後建模", "ml_compare",
            task_type="classification", target_col=tgt_name,
            balancing="undersample_10x",
        ))

    # ── 模型路徑建議 ──
    path_lines = []
    candidates = DEFAULT_CANDIDATES_CLF
    sample_note = ""
    if n_rows < 1000:
        sample_note = "樣本數 < 1,000，務必搭配交叉驗證以防過擬合。"
    elif n_rows > 100000:
        sample_note = "大樣本資料，樹系模型表現通常最佳，但訓練時間較長。"

    path_lines.append(
        f"**建議建模路徑**（{n_rows:,} 筆 × {profile['n_cols']} 欄）：\n"
        f"1. 先跑 **Logistic Regression 作為 baseline**（可解釋、訓練快）\n"
        f"2. 再跑 Random Forest / XGBoost 比較非線性增益\n"
        f"3. 以 **K-fold CV + AUC** 選出最穩定模型\n"
        f"{sample_note}"
    )
    compare_action = _mk_action(
        "🏆 一鍵比較候選模型（含 CV）", "ml_compare",
        task_type="classification", target_col=tgt_name,
        models=candidates, balancing="none",
    )

    body = "\n\n".join(balance_lines + path_lines)
    all_actions = ([a for a in balance_actions] +
                   [compare_action] +
                   [_mk_action(f"🎯 直接用 `{tgt_name}` 開啟建模面板", "ml",
                               task_type="classification", target_col=tgt_name)])

    recs.append({
        "category": "modeling",
        "title": f"建模策略 — 目標 `{tgt_name}`（事件率 {event_rate:.2%}）",
        "reason": body,
        "actions": all_actions[:4],
    })
    return recs


# ═══════════════════════════════════════════════
# ③ 變數篩選與分箱建議
# ═══════════════════════════════════════════════
def recommend_binning_and_selection(df: pd.DataFrame, profile: dict) -> list:
    """
    對應「如何切變數」的兩個層次：
    - 篩選：IV 表排序 → 高相關配對擇優 → f_classif/RFE/AIC
    - 分箱：偏態/離群/二元目標 → 分箱法選擇
    """
    recs = []
    targets = profile.get("candidate_targets", [])
    numeric_feats = [c for c in profile["numeric_cols"]
                     if not any(t["column"] == c for t in targets)]

    has_binary_target = bool(targets)

    # ── 篩選路徑總覽 ──
    screen_steps = []
    screen_params = {}
    if has_binary_target:
        tgt = targets[0]["column"]
        screen_params = {
            "target_col": tgt,
            "task_type": "classification",
            "numeric_features": numeric_feats,
            "categorical_features": profile["categorical_cols"],
        }
        screen_steps = [
            f"1. **WOE/IV 表**排序所有變數的預測力（IV<0.02 剔除、IV>0.5 警示洩漏）",
            "2. **f_classif 單變量篩選**剔除 p>0.05 的變數",
            "3. **高相關配對擇優**：|r|>0.9 的變數對，依樹模型重要性保留較優者",
            "4. **RFE 遞歸消除**收斂至精簡特徵集（可用 AIC 逐步法替代）",
        ]
        screen_actions = [
            _mk_action("📊 跑 WOE/IV 變數預測力排名", "iv_table", **screen_params),
            _mk_action("🔍 一鍵跑完整特徵海選", "feature_select_suite", **screen_params),
        ]
        recs.append({
            "category": "feature",
            "title": f"變數篩選路徑 — 目標 `{tgt}`",
            "reason": "實戰驗證的海選流程：\n" + "\n".join(screen_steps),
            "actions": screen_actions,
        })

    # ── 分箱建議 ──
    bin_lines = []
    bin_targets_skew = []
    for item in profile.get("high_skew_columns", []):
        name = item["column"]
        p = profile["columns"].get(name, {})
        if name in numeric_feats and p.get("unique", 0) > 10:
            bin_targets_skew.append(item)
            method = "tree" if has_binary_target else "quantile"
            method_note = ("決策樹分箱（自動找最佳切點，配合 WOE 使用）"
                           if method == "tree" else "等頻分箱（每箱樣本均勻）")
            bin_lines.append(
                f"- `{name}`：偏態 **{item['skewness']}** → 建議{method_note}，"
                f"或先做對數轉換"
            )
    for name, p in profile["columns"].items():
        if "many_outliers" in p.get("flags", []) and name in numeric_feats:
            if name not in [b["column"] for b in bin_targets_skew]:
                bin_lines.append(
                    f"- `{name}`：IQR 離群值 {p['outliers_iqr']} 個"
                    f"（{p['outlier_pct']:.1%}）→ 建議等寬分箱前先截斷（winsorize）極端值"
                )
    for name, p in profile["columns"].items():
        if "high_cardinality" in p.get("flags", []):
            bin_lines.append(
                f"- `{name}`：{p['unique']} 個類別屬高基數 → 建議合併稀有類別"
                f"（出現次數 < {max(profile['n_rows'] // 500, 10)}）後再編碼"
            )

    if bin_lines:
        woe_action = ([_mk_action("📊 查看 WOE 分箱詳情", "statistics",
                                  analysis_type="WOE/IV 分析", **screen_params)]
                      if has_binary_target else [])
        recs.append({
            "category": "feature",
            "title": "變數分箱 / 轉換建議",
            "reason": "\n".join(bin_lines),
            "actions": woe_action[:2],
        })

    return recs


# ═══════════════════════════════════════════════
# 主入口：產生完整健檢報告
# ═══════════════════════════════════════════════
CATEGORY_LABELS = {
    "missing": "🔬 遺漏值與資料品質",
    "modeling": "🤖 演算法與建模策略",
    "feature": "📐 變數篩選與分箱",
}


def generate_advice_report(df: pd.DataFrame, profile: dict = None) -> dict:
    """
    產生完整分析建議報告。

    Returns:
        {
          "markdown": str,           # 完整 Markdown 報告（顯示於聊天區）
          "recommendations": [...],  # 結構化建議清單
          "all_actions": [...],      # 匯總的所有 action（供按鈕渲染）
          "profile": {...},
        }
    """
    if profile is None:
        profile = build_data_profile(df)

    recommendations = []
    recommendations.extend(recommend_missing_handling(df, profile))
    recommendations.extend(recommend_modeling_strategy(df, profile))
    recommendations.extend(recommend_binning_and_selection(df, profile))

    # ── 組裝 Markdown ──
    lines = ["# 📋 AI 資料健檢與分析建議", ""]
    lines.append(f"> 資料規模：**{profile['n_rows']:,}** 筆 × **{profile['n_cols']}** 欄｜"
                 f"缺失 {profile['total_missing']:,}｜重複列 {profile['duplicate_rows']:,}")
    lines.append("")

    current_cat = None
    all_actions = []
    for rec in recommendations:
        if rec["category"] != current_cat:
            current_cat = rec["category"]
            lines.append("---")
            lines.append(f"## {CATEGORY_LABELS.get(current_cat, current_cat)}")
            lines.append("")
        lines.append(f"### {rec['title']}")
        lines.append(rec["reason"])
        lines.append("")
        for act in rec.get("actions", []):
            all_actions.append(act)

    return {
        "markdown": "\n".join(lines),
        "recommendations": recommendations,
        "all_actions": all_actions,
        "profile": profile,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame({
        "年齡": rng.integers(20, 70, 300).astype(float),
        "收入": rng.lognormal(11, 0.8, 300),   # 高偏態
        "城市": rng.choice(["台北", "台中", "高雄"], 300),
        "違約": rng.choice([0, 1], 300, p=[0.96, 0.04]),
        "負債比": rng.uniform(0, 1, 300),
        "客戶ID": [f"C{i:04d}" for i in range(300)],
    })
    test_df.loc[test_df.index[:50], "年齡"] = np.nan
    test_df.loc[test_df.index[:200], "備註"] = None
    test_df["備註"] = test_df["備註"].fillna(rng.choice(["A", "B"], 100))

    report = generate_advice_report(test_df)
    print(report["markdown"])
    print("\n--- Actions ---")
    for a in report["all_actions"]:
        print(a["module"], "|", a["label"])
