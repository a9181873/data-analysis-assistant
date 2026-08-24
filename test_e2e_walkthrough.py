# -*- coding: utf-8 -*-
"""
端到端完整走一遍 (End-to-End Walkthrough)
═══════════════════════════════════════
資料集：UCI Heart Disease（克利夫蘭）— 303 筆、14 欄
場景模擬：使用者拿到一份全新的原始資料，從載入到建模的完整旅程

Step 0  前處理原始檔（欄位命名、?→缺失、目標二元化）
Step 1  載入 data_loader.load_data
Step 2  資料剖析 profiler.build_data_profile
Step 3  AI 健檢報告 advisor.generate_advice_report
Step 4  套用遺漏值建議（action: apply_missing_strategy）
Step 5  WOE/IV 變數預測力排名（action: iv_table）
Step 6  特徵海選 feature_selection.run_full_suite
Step 7  模型比較 compare_models_cv + 過擬合診斷
Step 8  分析記憶 findings 回顧
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

JOURNEY = []   # 模擬 session_state.findings


def finding(tool, summary):
    JOURNEY.append({"tool": tool, "summary": summary})
    print(f"       📝 findings += [{tool}] {summary}")


# ══════════════════════════════════════════════════════════
print("▛" + "─" * 64)
print("▌ Step 0｜前處理原始檔")
print("▙" + "─" * 64)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
raw = pd.read_csv("test_datasets/cleveland_raw.csv", header=None, names=COLS,
                  na_values="?")
n_missing_raw = int(raw.isnull().sum().sum())
raw["heart_disease"] = (raw["num"] > 0).astype(int)   # 0=健康, 1-4=患病 → 二元化
raw = raw.drop(columns=["num"])
raw.to_csv("test_datasets/heart_disease.csv", index=False)
print(f"原始 303 筆 × 14 欄｜'?' 轉缺失: {n_missing_raw} 個｜"
      f"目標二元化 num→heart_disease")
print(f"目標分布: {dict(raw['heart_disease'].value_counts())} "
      f"(患病率 {raw['heart_disease'].mean():.1%})")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 1｜載入（data_loader.load_data）")
print("▙" + "─" * 64)
from data_loader import load_data
df = load_data("test_datasets/heart_disease.csv")
print(f"✅ 載入成功: {df.shape[0]} 筆 × {df.shape[1]} 欄｜"
      f"缺失值 {int(df.isnull().sum().sum())} 個")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 2｜資料剖析（profiler）")
print("▙" + "─" * 64)
from profiler import build_data_profile, format_profile_for_llm
t0 = time.time()
profile = build_data_profile(df)
llm_ctx = format_profile_for_llm(profile)
print(f"剖析耗時 {time.time()-t0:.2f}s｜LLM context {len(llm_ctx)} chars")
targets = profile["candidate_targets"]
if targets:
    t = targets[0]
    print(f"🎯 首選目標偵測: `{t['column']}`（事件率 {t['event_rate']:.1%}）")
    assert t["column"] == "heart_disease", "目標偵測錯誤！"
    print("✅ 目標欄偵測正確")
finding("資料健檢", f"候選目標 `heart_disease`（事件率 {t['event_rate']:.1%}）")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 3｜AI 健檢報告（advisor）")
print("▙" + "─" * 64)
from advisor import generate_advice_report
report = generate_advice_report(df, profile=profile)
print(report["markdown"])
print()
print(f"📋 產生 {len(report['all_actions'])} 個可執行 actions:")
for a in report["all_actions"]:
    print(f"   - [{a['module']}] {a['label']}")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 4｜套用遺漏值補值建議（一鍵執行）")
print("▙" + "─" * 64)
from data_preprocessing import handle_missing_values
missing_actions = [a for a in report["all_actions"]
                   if a["module"] == "apply_missing_strategy"]
if missing_action := next(iter(missing_actions), None):
    strategies = missing_action["params"]["column_strategies"]
    print(f"建議策略: {strategies}")
    df = handle_missing_values(df, column_strategies=strategies)
    remaining = int(df.isnull().sum().sum())
    print(f"✅ 補值完成，剩餘缺失: {remaining}")
    assert remaining == 0, "仍有缺失未處理！"
    finding("遺漏值處理", f"逐欄策略補值 {list(strategies.keys())}，剩餘缺失 0")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 5｜WOE/IV 變數預測力排名")
print("▙" + "─" * 64)
from feature_selection import iv_ranking
feats_all = [c for c in df.columns if c != "heart_disease"]
iv_table, drop_iv, leak_iv = iv_ranking(df, "heart_disease", features=feats_all)
print(iv_table.head(10).to_string(index=False))
if drop_iv:
    print(f"\n❌ IV<0.02 建議剔除: {drop_iv}")
if leak_iv:
    print(f"⚠️ IV≥0.5 洩漏警示: {leak_iv}")
top3 = ", ".join(iv_table.head(3)["feature"])
finding("WOE/IV 篩選", f"IV 前3名: {top3}")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 6｜特徵海選（單變量→相關擇優→RFE→AIC）")
print("▙" + "─" * 64)
from feature_selection import run_full_suite, format_suite_report
cat_feats = [c for c in feats_all if not pd.api.types.is_numeric_dtype(df[c])]
suite = run_full_suite(
    df, "heart_disease",
    numeric_features=[c for c in feats_all if pd.api.types.is_numeric_dtype(df[c])],
    categorical_features=cat_feats,
    task_type="classification")
print(format_suite_report(suite))
final_feats = list(suite["final_features"])
finding("特徵海選",
        f"{suite['n_original']} 變數收斂至 {suite['n_final']} 個: "
        f"{', '.join(final_feats[:6])}" + ("..." if len(final_feats) > 6 else ""))

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 7｜模型比較（K-Fold CV + class_weight）+ 過擬合診斷")
print("▙" + "─" * 64)
from ml_models import (compare_models_cv, train_test_roc_check,
                       prepare_data, get_balanced_models)
X_train, X_test, y_train, y_test, le, pre = prepare_data(
    df, "heart_disease", final_feats,
    test_size=0.2, task_type="classification")

t0 = time.time()
comp_df, cv_list, best_name, fitted = compare_models_cv(
    X_train, y_train, n_folds=5, task_type="classification",
    balancing_strategy="class_weight")
print(comp_df.to_string(index=False))
print(f"\n[CV 耗時 {time.time()-t0:.1f}s]　🏆 最佳模型: {best_name}")
finding("模型比較", f"CV 最佳模型: {best_name}")

check = train_test_roc_check(best_name, X_train, y_train, X_test, y_test)
print(f"\n🔍 過擬合診斷 — {check['model_name']}")
print(f"   Train AUC: {check['train']['auc']:.4f}")
print(f"   Test  AUC: {check['test']['auc']:.4f}")
print(f"   Gap: {check['auc_gap']:.4f} → {check['overfit_level']}度過擬合")
print(f"   診斷: {check['diagnosis']}")
finding("過擬合檢查",
        f"AUC gap {check['auc_gap']:.3f}（{check['overfit_level']}度）")

# ══════════════════════════════════════════════════════════
print()
print("▛" + "─" * 64)
print("▌ Step 8｜分析記憶回顧（findings）")
print("▙" + "─" * 64)
for i, f in enumerate(JOURNEY, 1):
    print(f"  {i}. [{f['tool']}] {f['summary']}")

print()
print("═" * 58)
print(f"🎉 完整流程走完 — 共 {len(JOURNEY)} 條分析發現累積於記憶")
print("═" * 58)
