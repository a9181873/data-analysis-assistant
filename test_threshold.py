# -*- coding: utf-8 -*-
"""測試 optimize_threshold：用水險級極端不平衡 (0.49%) 驗證"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from ml_models import optimize_threshold, prepare_data, get_balanced_models
from sklearn.base import clone

# 模擬水險場景：0.49% 事件率、20000 筆
rng = np.random.default_rng(42)
n = 20000
df_imb = pd.DataFrame({
    "f1": rng.normal(0, 1, n),
    "f2": rng.normal(0, 1, n),
    "f3": rng.choice([0, 1], n, p=[0.6, 0.4]),
})
# 讓事件與特徵有關聯（可學習）
score = df_imb["f1"] * 1.2 + df_imb["f2"] * 0.8 + df_imb["f3"] * 0.5
prob_true = 1 / (1 + np.exp(-(score - score.mean()) / 0.8))
df_imb["target"] = (rng.random(n) < prob_true * 0.02).astype(int)
event_rate = df_imb["target"].mean()
print(f"模擬資料: {n:,} 筆｜事件率 {event_rate:.3%}（{df_imb['target'].sum()} 筆）")
assert event_rate < 0.01, "模擬資料不夠極端"

X_train, X_test, y_train, y_test, le, pre = prepare_data(
    df_imb, "target", ["f1", "f2", "f3"],
    test_size=0.2, task_type="classification")

# 用 class_weight balanced 訓練 LR
model = clone(get_balanced_models("class_weight")["Logistic Regression (邏輯迴歸)"])
model.fit(X_train.values, y_train)
y_prob = model.predict_proba(X_test.values)[:, 1]

print()
print("═" * 60)
print("▶ Youden J 策略")
print("═" * 60)
r1 = optimize_threshold(y_test, y_prob, strategy="youden")
print(f"預設閾值 0.50 → P={r1['default_05']['precision']:.3f} "
      f"R={r1['default_05']['recall']:.3f} F1={r1['default_05']['f1']:.3f}")
print(f"最佳閾值 {r1['threshold']:.3f} → P={r1['precision']:.3f} "
      f"R={r1['recall']:.3f} F1={r1['f1']:.3f}")
print(f"💬 {r1['note']}")
assert r1["threshold"] != 0.5, "Youden 未找到不同閾值"
# Youden 優化 TPR-FPR：Recall 應明顯提升（不平衡下預設 0.5 召回偏低）
assert r1["recall"] > r1["default_05"]["recall"], "Youden 應提升召回"

print()
print("═" * 60)
print("▶ F1 最大化策略")
print("═" * 60)
r2 = optimize_threshold(y_test, y_prob, strategy="f1")
print(f"最佳閾值 {r2['threshold']:.3f} → F1={r2['f1']:.3f}")
assert r2["f1"] >= r1["default_05"]["f1"]

print()
print("═" * 60)
print("▶ Recall 目標策略（風控情境：寧可錯殺）")
print("═" * 60)
r3 = optimize_threshold(y_test, y_prob, strategy="recall", target_recall=0.80)
print(f"達成 Recall≥0.80 的最小閾值: {r3['threshold']:.3f}")
print(f"→ Recall={r3['recall']:.3f} Precision={r3['precision']:.3f}")
assert r3["recall"] >= 0.79, "Recall 目標未達成"

print()
print("✅ optimize_threshold 三種策略全部通過")
