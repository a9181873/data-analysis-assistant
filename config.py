# ===================================
# 數據分析小幫手 - 統一配置管理
# ===================================
import os

# --- LLM 模型配置 ---
# 推理模型 (推薦使用 DeepSeek-R1 以獲得最佳數理邏輯能力)
#   - deepseek-r1:14b  ~9GB RAM，適合 16GB-32GB RAM 機器 (極致推理)
#   - qwen2.5:14b      備用通用模型
#   - qwen2.5:7b       輕量版模型，速度最快
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1:7b")

# --- 雲端 API 部署配置 ---
# 支援多家雲端 LLM 提供者，API Key 透過 UI 輸入（存於 session state，不落地）
USE_CLOUD_LLM = os.environ.get("USE_CLOUD_LLM", "False").lower() == "true"

# 雲端提供者配置 (名稱, 預設 base_url, 推薦模型, 環境變數 key 名稱)
CLOUD_PROVIDERS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": [
            {"id": "gpt-4o",      "rating": "⭐⭐⭐⭐",  "note": "全能均衡，生態最成熟"},
            {"id": "gpt-4o-mini", "rating": "⭐⭐⭐",   "note": "輕量便宜，簡單分析夠用"},
            {"id": "gpt-4.1",     "rating": "⭐⭐⭐⭐⭐", "note": "最新旗艦，推理與程式碼強"},
            {"id": "gpt-4.1-mini","rating": "⭐⭐⭐⭐",  "note": "4.1 輕量版，性價比高"},
            {"id": "o3-mini",     "rating": "⭐⭐⭐⭐",  "note": "推理專用，數學邏輯好"},
        ],
        "env_key": "OPENAI_API_KEY",
        "note": "生態最成熟，插件豐富",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            {"id": "gemini-2.5-flash-preview-05-20", "rating": "⭐⭐⭐⭐⭐", "note": "便宜快速，數據分析性價比最高"},
            {"id": "gemini-2.5-pro-preview-06-05",   "rating": "⭐⭐⭐⭐⭐", "note": "旗艦級推理，複雜分析首選"},
            {"id": "gemini-2.0-flash",               "rating": "⭐⭐⭐⭐",  "note": "穩定版 Flash，速度快"},
        ],
        "env_key": "GEMINI_API_KEY",
        "note": "免費額度高，多模態強",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            {"id": "anthropic/claude-sonnet-4",      "rating": "⭐⭐⭐⭐⭐", "note": "推理最強，程式碼與分析頂級"},
            {"id": "openai/gpt-4o",                  "rating": "⭐⭐⭐⭐",  "note": "全能均衡，穩定可靠"},
            {"id": "deepseek/deepseek-r1",           "rating": "⭐⭐⭐⭐⭐", "note": "深度推理，數學邏輯最強"},
            {"id": "qwen/qwen3-235b-a22b",           "rating": "⭐⭐⭐⭐",  "note": "Qwen3 旗艦，ML 能力強"},
            {"id": "anthropic/claude-haiku-4",        "rating": "⭐⭐⭐⭐",  "note": "便宜快速，簡單分析夠用"},
            {"id": "meta-llama/llama-4-maverick",     "rating": "⭐⭐⭐",   "note": "開源通用，ML 深度分析稍弱"},
        ],
        "env_key": "OPENROUTER_API_KEY",
        "note": "聚合平台，一個 API Key 使用多家模型，可比價",
    },
}

# 雲端 LLM 連線（由 sidebar 動態設定，不要手動修改）
CLOUD_API_KEY = ""
CLOUD_BASE_URL = ""

# Embedding 模型 (用於 RAG 文檔檢索，sentence-transformers 格式)
#   - BAAI/bge-m3              ~1.2GB，多語言 (中/英/日/韓)，中文表現最佳 (推薦)
#   - all-MiniLM-L6-v2         ~80MB，英文為主，輕量快速
#   - shibing624/text2vec-base-chinese  ~400MB，中文專用
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120.0"))

# --- Streamlit 配置 ---
APP_TITLE = "數據分析小幫手"
APP_ICON = "📊"
SERVER_PORT = 8501

# --- 檔案上傳配置 ---
SUPPORTED_FILE_TYPES = ['csv', 'txt', 'xlsx', 'xls', 'sas7bdat']
MAX_FILE_SIZE_MB = 200

# --- 機器學習配置 ---
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# --- 超參數調整配置 ---
DEFAULT_GRID_SEARCH_CV = 3       # GridSearchCV 預設折數（3 折速度較快）

# --- ML 任務類型常數 ---
ML_TASK_CLASSIFICATION = "classification"
ML_TASK_REGRESSION = "regression"

# --- 資料匯出配置 ---
EXPORT_ENCODING = "utf-8-sig"    # UTF-8 with BOM，Windows Excel 可正確顯示中文
