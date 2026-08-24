# -*- coding: utf-8 -*-
"""OpenRouter 金鑰診斷（安全版：全程遮罩，不顯示金鑰內容）"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os


def mask(v):
    """遮罩金鑰：只顯示前 8 碼與後 4 碼"""
    if not v:
        return "(空)"
    return v[:8] + "****" + v[-4:] if len(v) > 16 else "****(過短)"


print("═" * 60)
print("▶ 1. dotenv 載入後，環境中偵測到的 API 變數名稱")
print("═" * 60)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', ".env"),
            encoding="utf-8-sig")

KEY_NAMES = ["OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]
found_any = False
for k in KEY_NAMES:
    v = os.environ.get(k)
    status = "✅" if v else "—"
    print(f"  {status} {k}: {mask(v)}")
    if v:
        found_any = True
if not found_any:
    print("\n  ⚠️ 三個變數都是空的！可能原因：")
    print("     a) .env 內變數名拼錯（需完全一致：OPENROUTER_API_KEY）")
    print("     b) 檔名實際上是 .env.txt（Windows 隱藏副檔名）")
    print("     c) 行首有 # 註解掉")

print()
print("═" * 60)
print("▶ 2. OpenRouter 實際認證測試（用偵測到的金鑰）")
print("═" * 60)
key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("❌ 無法測試：OPENROUTER_API_KEY 未載入")
    sys.exit(0)

import requests
r = requests.get("https://openrouter.ai/api/v1/auth/key",
                 headers={"Authorization": f"Bearer {key}"}, timeout=15)
print(f"HTTP 狀態: {r.status_code}")
if r.status_code == 200:
    info = r.json().get("data", {})
    print("✅ 金鑰有效！")
    label = info.get("label", "")
    usage = info.get("usage")
    limit = info.get("limit")
    free_tier = info.get("is_free_tier")
    print(f"   標籤: {label}｜已用: {usage}｜額度上限: {limit}｜免費層: {free_tier}")
elif r.status_code == 401:
    print("❌ 金鑰無效（401）——請到 openrouter.ai/keys 重新確認或重建金鑰")
else:
    print(f"⚠️ 未預期狀態: {r.text[:200]}")

print()
print("═" * 60)
print("▶ 3. 免費模型實際呼叫測試（max_tokens=5，幾乎零成本）")
print("═" * 60)
r2 = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={"model": "deepseek/deepseek-chat-v3-0324:free",
          "messages": [{"role": "user", "content": "hi"}],
          "max_tokens": 5},
    timeout=30)
print(f"HTTP 狀態: {r2.status_code}")
if r2.status_code == 200:
    content = r2.json()["choices"][0]["message"]["content"]
    print(f"✅ 模型回覆: {content!r} —— LLM 對話層驗證成功！")
else:
    body = r2.json() if r2.headers.get("content-type", "").startswith("application/json") else {}
    err = body.get("error", {}).get("message", r2.text[:200])
    print(f"❌ 呼叫失敗: {err}")
