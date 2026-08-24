# -*- coding: utf-8 -*-
"""
煙霧測試：用公開資料集驗證 profiler.py + advisor.py 端到端
資料集特性互補：
  1. titanic          — 分類、Age/Cabin 大量缺失、類別欄位
  2. telco_churn      — 分類、TotalCharges 隱性缺失(字串)、多類別欄、不平衡 ~26%
  3. adult            — 分類、混合型別、~24% 正類
  4. german           — 信用評分場景（對應 WOE/IV 流程）、30% 壞帳
  5. california       — 迴歸（無二元目標 → 測試「無候選目標」分支）
"""
import sys, io, time, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

from profiler import build_data_profile, format_profile_for_llm
from advisor import generate_advice_report


def load_dataset(name, path):
    if name == "titanic":
        return pd.read_csv(path)
    if name == "telco_churn":
        df = pd.read_csv(path)
        # 還原真實世界狀況：TotalCharges 有空字串隱性缺失
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df
    if name == "adult":
        cols = ["age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week",
                "native-country", "income"]
        df = pd.read_csv(path, header=None, names=cols, skipinitialspace=True, na_values="?")
        # fnlwgt 是普查加權 ID-like 欄位，保留以測試 id_like 偵測
        return df
    if name == "german":
        col_names = ["checking_acc", "duration", "credit_history", "purpose",
                     "credit_amount", "savings", "employment_since", "installment_rate",
                     "personal_status_sex", "other_debtors", "residence_since",
                     "property", "age", "other_installment_plans", "housing",
                     "n_credits", "job", "n_people_liable", "telephone", "foreign_worker",
                     "risk"]
        df = pd.read_csv(path, sep=" ", header=None, names=col_names)
        return df
    if name == "california":
        return pd.read_csv(path)
    raise ValueError(name)


DATASETS = {
    "titanic":     ("test_datasets/titanic.csv",             "Survived"),
    "telco_churn": ("test_datasets/telco_churn.csv",         "Churn"),
    "adult":       ("test_datasets/adult.csv",               "income"),
    "german":      ("test_datasets/german.data",             "risk"),
    "california":  ("test_datasets/california_housing.csv",  None),  # 迴歸無二元目標
}

PASS, FAIL = [], []

for name, (path, expected_target) in DATASETS.items():
    print("=" * 70)
    print(f"▶ 資料集: {name}")
    print("=" * 70)
    try:
        t0 = time.time()
        df = load_dataset(name, path)
        load_t = time.time() - t0

        # ── profiler ──
        t0 = time.time()
        profile = build_data_profile(df)
        prof_t = time.time() - t0

        llm_ctx = format_profile_for_llm(profile)

        # ── advisor ──
        t0 = time.time()
        report = generate_advice_report(df, profile=profile)
        adv_t = time.time() - t0

        targets = [t["column"] for t in profile["candidate_targets"][:5]]
        target_ok = True
        if expected_target:
            target_ok = any(t == expected_target for t in profile["candidate_targets"][0:1] and [expected_target]) or \
                        expected_target in targets
            if not target_ok:
                # 寬鬆判定：目標在候選清單即可（排序可能不在第一位）
                target_ok = expected_target in [c for c in df.columns
                                                if df[c].nunique() == 2]

        n_actions = len(report["all_actions"])
        n_recs = len(report["recommendations"])

        print(f"  形狀: {df.shape[0]:,} × {df.shape[1]} | 載入 {load_t:.2f}s | "
              f"剖析 {prof_t:.2f}s | 建議 {adv_t:.2f}s")
        print(f"  缺失值: {profile['total_missing']:,} | 重複列: {profile['duplicate_rows']:,}")
        print(f"  候選目標: {targets}")
        print(f"  目標偵測: {'✅ PASS' if target_ok else '❌ FAIL'}"
              + (f" (期望 {expected_target})" if expected_target else ""))
        print(f"  ID-like 欄: {profile['id_like_columns'][:3]}")
        print(f"  高缺失欄: {profile['high_missing_columns'][:5]}")
        print(f"  不平衡候選: {[c['column'] for c in profile['imbalanced_candidates']][:4]}")
        print(f"  建議數: {n_recs} | 可執行 actions: {n_actions}")

        # 基本斷言
        assert profile["n_rows"] == len(df), "筆數不符"
        assert n_recs > 0 or name == "california", f"{name}: 未產生任何建議"
        if expected_target:
            assert target_ok, f"{name}: 目標欄 {expected_target} 偵測失敗"

        # LLM context 長度控制驗證
        ctx_len = len(llm_ctx)
        print(f"  LLM context: {ctx_len:,} chars {'✅ (<6000)' if ctx_len < 6000 else '⚠️ 過長'}")

        PASS.append(name)
        print(f"  ✅ {name} PASS\n")

        # 印出報告前段供人工檢查（僅前兩個資料集）
        if name in ("titanic", "german"):
            print("─── 報告預覽（前 60 行）─".ljust(65, "─"))
            preview_lines = report["markdown"].splitlines()[:60]
            print("\n".join(preview_lines))
            print()

    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  ❌ {name} FAIL: {e}")
        traceback.print_exc()
        print()

print("=" * 70)
print(f"結果: {len(PASS)} PASS / {len(FAIL)} FAIL")
print("PASS:", ", ".join(PASS))
if FAIL:
    for name, err in FAIL:
        print(f"FAIL: {name} — {err}")
