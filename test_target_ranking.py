# -*- coding: utf-8 -*-
"""驗證目標欄偵測排序：關鍵詞命中的欄位應排第一"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from profiler import build_data_profile


def check(name, df, expect):
    p = build_data_profile(df)
    top = p["candidate_targets"][0]["column"] if p["candidate_targets"] else None
    ok = top is not None and str(top).lower() == expect.lower()
    print(f"{name:10s} 首選: {str(top):18s} 期望: {expect:12s} {'✅' if ok else '❌'}")
    return ok


results = []

df = pd.read_csv("test_datasets/titanic.csv")
results.append(check("titanic", df, "Survived"))

df = pd.read_csv("test_datasets/telco_churn.csv")
results.append(check("telco_churn", df, "Churn"))

cols = ["checking_acc", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment_since", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age",
        "other_installment_plans", "housing", "n_credits", "job",
        "n_people_liable", "telephone", "foreign_worker", "risk"]
df = pd.read_csv("test_datasets/german.data", sep=" ", header=None, names=cols)
results.append(check("german", df, "risk"))

df = pd.read_csv(
    "test_datasets/adult.csv", header=None, skipinitialspace=True, na_values="?",
    names=["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week",
           "native-country", "income"])
results.append(check("adult", df, "income"))

# german 欠採樣筆數修正驗證（bad=300 → 應建議抽 3000 筆）
from advisor import recommend_modeling_strategy
p = build_data_profile(df_german := pd.read_csv("test_datasets/german.data",
                                                sep=" ", header=None, names=cols))
report = recommend_modeling_strategy(df_german, p)
reason = report[0]["reason"]
line = [l for l in reason.splitlines() if "欠採樣" in l]
print("\n欠採樣建議:", line[0].strip() if line else "(無)")
ok_sample = "3,000" in line[0] if line else False
print("欠採樣筆數 (期望 3,000):", "✅" if ok_sample else "❌")

print("\n總計:", f"{sum(results)}/{len(results)} PASS")
