# -*- coding: utf-8 -*-
"""
煙霧測試：feature_selection.py 用公開資料集驗證
- german credit：對應人工筆記的信用評分場景（卡方+相關+RFE+AIC 全流程）
- titanic：混合型別、缺失值情境
- california：迴歸路徑
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from feature_selection import (
    univariate_screen, iv_ranking, correlated_pairs_prune,
    rfe_select, aic_stepwise, run_full_suite, format_suite_report,
)

# ═════════ german credit ═════════
cols = ["checking_acc", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment_since", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age",
        "other_installment_plans", "housing", "n_credits", "job",
        "n_people_liable", "telephone", "foreign_worker", "risk"]
df = pd.read_csv("test_datasets/german.data", sep=" ", header=None, names=cols)
df["risk"] = (df["risk"] == 2).astype(int)  # 2=bad → 1

num_feats = ["duration", "credit_amount", "installment_rate", "residence_since",
             "age", "n_credits", "n_people_liable"]
cat_feats = [c for c in cols if c not in num_feats + ["risk"]]

print("═" * 66)
print("▶ german credit — 完整海選流程")
print("═" * 66)
t0 = time.time()
report = run_full_suite(df, "risk",
                        numeric_features=num_feats,
                        categorical_features=cat_feats,
                        task_type="classification")
print(format_suite_report(report))
print(f"\n[耗時 {report['elapsed_sec']}s]")

# ── IV 排名（二元目標適用）──
print("\n" + "═" * 66)
print("▶ german credit — WOE/IV 排名")
print("═" * 66)
iv_table, drop_iv, leak_iv = iv_ranking(df, "risk", features=num_feats + cat_feats)
print(iv_table.head(8).to_string(index=False))
if drop_iv:
    print(f"\nIV<{0.02} 建議剔除: {drop_iv}")
if leak_iv:
    print(f"⚠️ IV>={0.5} 洩漏警示: {leak_iv}")

# ── AIC 對照人工窮舉速度 ──
print("\n" + "═" * 66)
print("▶ AIC 貪婪法 vs 人工窮舉速度對照")
print("═" * 66)
t0 = time.time()
selected, trace = aic_stepwise(df, "risk", features=num_feats, direction="both")
aic_t = time.time() - t0
print(f"AIC 逐步選擇 ({len(num_feats)} 變數): {aic_t:.2f}s → 選出 {len(selected)} 個")
print("軌跡:")
for feats, aic in trace:
    print(f"  AIC {aic} ← [{', '.join(feats) if feats else '(空)'}]")
print("(人工筆記暴力窮舉 15 變數耗時約 2430 秒)")

# ═════════ titanic ═════════
print("\n" + "═" * 66)
print("▶ titanic — 含缺失值的分類篩選")
print("═" * 66)
titanic = pd.read_csv("test_datasets/titanic.csv")
report2 = run_full_suite(titanic, "Survived",
                         numeric_features=["Pclass", "Age", "SibSp", "Parch", "Fare"],
                         categorical_features=["Sex", "Embarked"],
                         task_type="classification")
print(format_suite_report(report2))

# ═════════ california 迴歸 ═════════
print("\n" + "═" * 66)
print("▶ california housing — 迴歸路徑")
print("═" * 66)
cal = pd.read_csv("test_datasets/california_housing.csv").sample(5000, random_state=42)
report3 = run_full_suite(cal, "MedHouseVal", task_type="regression")
final3 = report3["final_features"]
print(f"原始 {report3['n_original']} 特徵 → 最終 {report3['n_final']} 個 "
      f"(耗時 {report3['elapsed_sec']}s)")
print("最終特徵集:", ", ".join(final3))

print("\n✅ feature_selection.py 煙霧測試完成")
