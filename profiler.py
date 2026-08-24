"""
資料剖析引擎 (Data Profiling Engine)
─────────────────────────────────────
深度分析 DataFrame 的每個欄位，產生結構化剖析報告，
供 AI 顧問與建議引擎 (advisor.py) 引用真實數據。

核心原則：
- 程式端保證「數字正確」，LLM 只負責解讀與潤飾。
- 寬表自動截斷（依訊息量排序），控制送給 LLM 的 token 量。
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── 閾值設定 ─────────────────────────────────────
HIGH_MISSING_PCT = 0.50      # 缺失率超過此值 → 建議捨棄欄位
HIGH_CARDINALITY_RATIO = 0.5  # 類別欄唯一值/筆數 超過此比例 → 高基數警示
SKEW_THRESHOLD = 1.0          # 偏態絕對值超過此值 → 建議中位數補值 / 分箱 / 對數轉換
OUTLIER_ALERT_RATIO = 0.05    # IQR 離群值比例超過此值 → 警示
IMBALANCE_THRESHOLD = 0.20    # 二元欄少數類佔比低於此值 → 類別不平衡


def _is_binary(series: pd.Series) -> bool:
    """判斷是否為二元欄位（0/1 或兩個唯一值）。"""
    uniq = series.dropna().unique()
    if len(uniq) != 2:
        return False
    # 允許 0/1、True/False、任意兩類別
    return True


def profile_column(series: pd.Series, n_rows: int) -> dict:
    """
    剖析單一欄位。

    Returns:
        dict 欄位剖析結果，依型別包含不同 keys。
    """
    info = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "missing": int(series.isnull().sum()),
        "missing_pct": round(float(series.isnull().mean()), 4),
        "unique": int(series.nunique()),
    }

    is_numeric = pd.api.types.is_numeric_dtype(series)
    info["role"] = "numeric" if is_numeric else "categorical"

    # ── ID-like：唯一值幾乎等於筆數 ──
    clean_n = int(series.notna().sum())
    if clean_n > 0 and info["unique"] >= clean_n * 0.95 and not is_numeric:
        info["flags"] = ["id_like"]
    elif clean_n > 0 and is_numeric and info["unique"] >= clean_n * 0.95 and info["unique"] > 50:
        info["flags"] = ["id_like"]
    else:
        info["flags"] = []

    # ── 常數欄 ──
    if info["unique"] <= 1:
        info["flags"].append("constant")

    if is_numeric:
        clean = series.dropna()
        if len(clean) > 0:
            q1 = float(clean.quantile(0.25))
            q3 = float(clean.quantile(0.75))
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (clean < lower) | (clean > upper)
            skew = float(clean.skew()) if len(clean) >= 3 else 0.0
            info.update({
                "mean": round(float(clean.mean()), 4),
                "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                "min": round(float(clean.min()), 4),
                "q1": round(q1, 4),
                "median": round(float(clean.median()), 4),
                "q3": round(q3, 4),
                "max": round(float(clean.max()), 4),
                "skewness": round(skew, 3),
                "outliers_iqr": int(outlier_mask.sum()),
                "outlier_pct": round(float(outlier_mask.mean()), 4),
            })
            # 整數型且唯一值少 → 視為離散/類別性質
            if info["unique"] <= 10:
                info["role"] = "numeric_discrete"
                vc = clean.value_counts()
                info["top_values"] = [
                    {"value": str(k), "count": int(v), "pct": round(float(v) / len(clean), 4)}
                    for k, v in vc.head(6).items()
                ]
                if len(vc) == 2:
                    minority_pct = float(vc.min() / len(clean))
                    info["minority_pct"] = round(minority_pct, 4)
                    if minority_pct < IMBALANCE_THRESHOLD:
                        info["flags"].append("imbalanced")
                    info["flags"].append("binary")
            if abs(skew) > SKEW_THRESHOLD:
                info["flags"].append("high_skew")
            if info.get("outlier_pct", 0) > OUTLIER_ALERT_RATIO:
                info["flags"].append("many_outliers")
    else:
        clean = series.dropna().astype(str)
        if len(clean) > 0:
            vc = clean.value_counts()
            info["top_values"] = [
                {"value": k[:30], "count": int(v), "pct": round(float(v) / len(clean), 4)}
                for k, v in vc.head(6).items()
            ]
            cardinality_ratio = info["unique"] / max(len(clean), 1)
            if cardinality_ratio > HIGH_CARDINALITY_RATIO and info["unique"] > 50:
                info["flags"].append("high_cardinality")
            if len(vc) == 2:
                minority_pct = float(vc.min() / len(clean))
                info["minority_pct"] = round(minority_pct, 4)
                if minority_pct < IMBALANCE_THRESHOLD:
                    info["flags"].append("imbalanced")
                info["flags"].append("binary")

    if info["missing_pct"] > HIGH_MISSING_PCT:
        info["flags"].append("high_missing")
    elif info["missing_pct"] > 0:
        info["flags"].append("has_missing")

    return info


# ── 目標欄關鍵詞（命中者大幅提升為首選目標的機率）──
TARGET_KEYWORDS = [
    "churn", "default", "risk", "claim", "fraud", "survived",
    "target", "label", "response", "converted", "buy", "subscribed",
    "y", "is_bad", "is_good", "outcome", "diagnosis",
    "disease", "sick", "death", "dead", "stroke", "attack",
    "attrition", "exited", "readmit", "recurrence",
    "違約", "理賠", "流失", "目標", "風險", "詐欺", "存活", "患病",
]


def detect_candidate_targets(df: pd.DataFrame, exclude_cols=None) -> list:
    """
    偵測候選目標欄位（二元欄＋事件率），
    對應實戰流程中「claimindex」這類 0/1 目標的自動識別。

    排序評分：
    - 名稱命中關鍵詞（churn/risk/claim/disease…）→ 強力優先
    - 數值型二元欄 → 小幅優先（比字串欄更可能是建模目標）
    - 位於最後一欄 → 加分（建模資料的目標慣例放最後）
    - 其餘依事件率接近 15% 程度排序
    """
    if exclude_cols is None:
        exclude_cols = []
    candidates = []
    last_col = str(df.columns[-1]) if len(df.columns) > 0 else None
    for col in df.columns:
        if col in exclude_cols:
            continue
        s = df[col]
        if s.nunique(dropna=True) != 2:
            continue
        vc = s.value_counts()
        minority_label = vc.idxmin()
        minority_pct = float(vc.min() / len(s.dropna()))
        is_numeric = bool(pd.api.types.is_numeric_dtype(s))

        # ── 評分（越小越優先）──
        score = abs(minority_pct - 0.15)
        name_lower = str(col).lower()
        if any(kw in name_lower for kw in TARGET_KEYWORDS):
            score -= 1.0          # 關鍵詞強力加分
        if is_numeric:
            score -= 0.05         # 數值型小幅加分
        if str(col) == last_col:
            score -= 0.10         # 慣例：目標常在最後一欄

        candidates.append({
            "column": col,
            "minority_class": str(minority_label),
            "event_rate": round(minority_pct, 5),
            "counts": {str(k): int(v) for k, v in vc.items()},
            "is_numeric": is_numeric,
            "_score": round(score, 4),
        })
    # 事件率過於極端（<0.1%）的多半不是建模目標而是稀疏旗標，往後排
    candidates.sort(key=lambda c: c["_score"])
    return candidates


def build_data_profile(df: pd.DataFrame, max_cols: int = 60) -> dict:
    """
    建立完整資料剖析報告。

    Args:
        df: 目標 DataFrame
        max_cols: 最多剖析的欄位數（寬表截斷，優先保留有缺失/有問題的欄位）

    Returns:
        dict 全域統計 + 每欄剖析 + 問題清單 + 候選目標
    """
    n_rows, n_cols = df.shape

    # 寬表截斷策略：先全部粗略評分，挑最有資訊量的欄位
    cols = list(df.columns)
    if n_cols > max_cols:
        def col_priority(c):
            s = df[c]
            miss_ratio = float(s.isnull().mean())
            uniq = int(s.nunique())
            score = miss_ratio * 2.0
            if uniq <= 1 or uniq >= n_rows * 0.95:
                score += 1.0
            return -score  # 分數高者優先
        cols = sorted(cols, key=col_priority)[:max_cols]

    column_profiles = {}
    for c in cols:
        try:
            column_profiles[c] = profile_column(df[c], n_rows)
        except Exception as e:
            column_profiles[c] = {"name": str(c), "error": str(e)}

    numeric_cols = [c for c, p in column_profiles.items()
                    if p.get("role") in ("numeric", "numeric_discrete")]
    cat_cols = [c for c, p in column_profiles.items()
                if p.get("role") == "categorical"]

    duplicate_rows = int(df.duplicated().sum())

    profile = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "truncated": bool(n_cols > max_cols),
        "duplicate_rows": duplicate_rows,
        "total_missing": int(df.isnull().sum().sum()),
        "overall_missing_pct": round(float(df.isnull().mean().mean()), 4),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "columns": column_profiles,
        "candidate_targets": detect_candidate_targets(df),
        "constant_columns": [c for c, p in column_profiles.items()
                             if "constant" in p.get("flags", [])],
        "id_like_columns": [c for c, p in column_profiles.items()
                            if "id_like" in p.get("flags", [])],
        "high_missing_columns": [c for c, p in column_profiles.items()
                                 if "high_missing" in p.get("flags", [])],
        "imbalanced_candidates": [
            {"column": c, "minority_pct": p.get("minority_pct")}
            for c, p in column_profiles.items()
            if "imbalanced" in p.get("flags", [])
        ],
        "high_skew_columns": [
            {"column": c, "skewness": p.get("skewness"),
             "missing_pct": p.get("missing_pct"), 0: None}
            for c, p in column_profiles.items()
            if "high_skew" in p.get("flags", [])
        ],
    }
    # 清理上面 dict comprehension 的殘留 key
    for item in profile["high_skew_columns"]:
        item.pop(0, None)

    return profile


def format_profile_for_llm(profile: dict, max_chars: int = 5000) -> str:
    """
    將剖析報告轉為精簡文字，供注入 LLM context。
    控制在 max_chars 內，確保不會爆 token。
    """
    lines = []
    lines.append(f"【資料剖析】{profile['n_rows']:,} 筆 × {profile['n_cols']} 欄"
                 + ("（僅顯示重點欄位）" if profile.get("truncated") else ""))
    lines.append(f"缺失值總計: {profile['total_missing']:,} ({profile['overall_missing_pct']:.1%}) | "
                 f"完全重複列: {profile['duplicate_rows']:,}")

    # 候選目標
    targets = profile.get("candidate_targets", [])[:3]
    if targets:
        t_strs = [f"`{t['column']}`(事件率{t['event_rate']:.2%})" for t in targets]
        lines.append(f"二元目標候選: {', '.join(t_strs)}")

    # 有問題的欄位優先列出
    issue_lines = []
    for name, p in profile["columns"].items():
        flags = p.get("flags", [])
        if not flags or flags == ["has_missing"]:
            continue
        parts = []
        if "high_missing" in flags:
            parts.append(f"缺失率高達 {p['missing_pct']:.0%}")
        if "constant" in flags:
            parts.append("常數欄(無變異)")
        if "id_like" in flags:
            parts.append("疑似ID欄位")
        if "high_skew" in flags:
            parts.append(f"偏態 {p.get('skewness', 'N/A')}")
        if "imbalanced" in flags:
            parts.append(f"類別不平衡(少數類僅 {p.get('minority_pct', '?'):.1%})"
                         if isinstance(p.get('minority_pct'), float)
                         else "類別不平衡")
        if "many_outliers" in flags:
            parts.append(f"IQR 離群值 {p['outliers_iqr']} 個({p['outlier_pct']:.1%})")
        if parts:
            issue_lines.append(f"- `{name}`: {'、'.join(parts)}")
    if issue_lines:
        lines.append("⚠️ 資料品質問題：")
        lines.extend(issue_lines[:12])

    # 重點欄位摘要（每欄一行）
    summary_lines = []
    for name, p in list(profile["columns"].items()):
        if p.get("error"):
            summary_lines.append(f"- `{name}`: 剖析失敗")
            continue
        role = p.get("role")
        miss = f"缺失{p['missing_pct']:.0%}" if p["missing_pct"] > 0 else ""
        if role in ("numeric", "numeric_discrete"):
            desc = (f"{p.get('median')}中位/{p.get('mean')}均"
                    f"/偏態{p.get('skewness')}/{p['unique']}唯一值")
        else:
            tops = p.get("top_values", [])
            top_str = ", ".join(f"{t['value']}({t['pct']:.0%})" for t in tops[:3])
            desc = f"[{top_str}] {p['unique']}類"
        flag_str = "/".join(f for f in p.get("flags", []) if f != "has_missing")
        summary_lines.append(
            f"- `{name}`({role}): {desc} {miss}" + (f" ⚑{flag_str}" if flag_str else "")
        )
    lines.append("欄位摘要：")
    base = "\n".join(lines + summary_lines)
    if len(base) > max_chars:
        # 截斷欄位摘要行
        keep = []
        cur_len = len("\n".join(lines))
        for sl in summary_lines:
            if cur_len + len(sl) > max_chars:
                keep.append("- ...（其餘欄位省略）")
                break
            keep.append(sl)
            cur_len += len(sl) + 1
        base = "\n".join(lines + keep)
    return base


if __name__ == "__main__":
    # 煙霧測試
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame({
        "年齡": rng.integers(20, 70, 200).astype(float),
        "收入": rng.lognormal(11, 0.8, 200),
        "城市": rng.choice(["台北", "台中", "高雄"], 200),
        "違約": rng.choice([0, 1], 200, p=[0.96, 0.04]),
        "客戶ID": [f"C{i:04d}" for i in range(200)],
        "備註": [None] * 120 + list(rng.choice(["A", "B"], 80)),
    })
    test_df.loc[test_df.index[:30], "年齡"] = np.nan

    prof = build_data_profile(test_df)
    print(format_profile_for_llm(prof))
    print("\n--- 候選目標 ---")
    for t in prof["candidate_targets"]:
        print(t)
