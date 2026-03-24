# 數據分析小幫手 📊

基於 LangChain + Ollama（或雲端 LLM）的本地數據分析平台，具備 Streamlit GUI 介面，涵蓋資料科學全流程：從資料清理、統計分析、機器學習，到風險建模與 RAG 知識庫管理。

## 🎯 主要功能

| 分頁 | 功能說明 |
|------|---------|
| 資料預覽 | 上傳 CSV/Excel/SAS 檔案，查看基本資訊與缺失值狀況 |
| 資料預處理 | 缺失值填充、型別轉換、重複值處理 |
| 資料視覺化 | Plotly 互動圖表：直方圖、散點圖、盒鬚圖、相關熱圖等 |
| 統計分析 | 敘述統計、t 檢定、ANOVA、卡方、相關分析、迴歸 |
| 變數分析 | WOE/IV 分析、特徵重要性、分布檢驗 |
| 機器學習 | 分類/迴歸建模（LR/RF/XGBoost/LightGBM/CatBoost/LDA）、Optuna 調參、SHAP 可解釋性、模型匯出 |
| PSI 監控 | 族群穩定性指數（PSI）計算與視覺化 |
| RAG 管理 | 上傳文件建立知識庫，供 AI 助手查詢 |
| AI 助手 | LangChain Agent，具備 **AI Copilot** 模式：自動建議模組、預填參數、一鍵執行與結果回饋 |

## 💻 系統需求

- **作業系統**: Windows 10/11（`install.bat` / `start.bat`）；macOS/Linux 請用 `install.sh` / `start.sh`
- **Python**: 3.9 或更高版本
- **RAM**: 建議 16GB 以上（本地 LLM 需要更多）
- **Ollama**（選用）: 用於在本地運行開源 LLM

## 🚀 安裝與啟動

### Windows（推薦）

**步驟 1：安裝依賴**

雙擊執行 `install.bat`，自動建立虛擬環境並安裝所有 Python 套件。

**步驟 2：安裝 Ollama（如需本地 LLM）**

前往 [ollama.ai](https://ollama.ai/) 下載安裝，然後下載模型：

```bash
ollama pull qwen2.5:14b
```

**步驟 3：啟動應用程式**

雙擊 `start.bat`，瀏覽器會自動開啟 `http://localhost:8501`。

### macOS / Linux

```bash
# 安裝
bash install.sh

# 啟動
bash start.sh
```

### Docker（備用方案）

```bash
docker-compose up --build
```

---

## 🤖 AI 助手：人機協作 (Human-AI Collaboration)

本專案升級為 **AI Copilot** 架構，具備以下先進功能：

- **結構化分析建議**：LLM 不僅提供文字建議，還會回饋 JSON 格式的動作參數。
- **行動按鈕 (Actionable Buttons)**：對話中會出現建議的操作按鈕，點擊後自動跳轉至對應模組。
- **參數預填 (Param Pre-fill)**：AI 會根據數據內容與您的意圖，預先填入工具面板中的參數（如目標變數、模型名稱、圖表類型等）。
- **自動執行與回饋**：部分動作支援一鍵執行並將執行結果（如模型準確度、統計摘要）自動發回聊天框，讓 AI 做進一步回饋與建議。
- **雙模式切換**：
    - **本地模式 (Ollama)**：完全離線，保護數據隱私（推薦使用 Qwen2.5 或 Llama 3.1 8B 以上模型）。
    - **雲端模式 (OpenRouter)**：透過 API 使用 GPT-4o、Claude 3.5 Sonnet 等雲端最強模型。

## 📦 主要依賴

- **Streamlit** — 網頁 GUI
- **LangChain 0.3.x** — Agent 框架
- **scikit-learn / XGBoost / LightGBM / CatBoost** — 機器學習
- **Optuna** — 超參數自動調整
- **SHAP** — 模型可解釋性
- **ChromaDB + sentence-transformers** — RAG 向量資料庫
- **Plotly** — 互動視覺化

## 🔧 常見問題

#### Ollama 連接失敗
確認 Ollama 服務正在運行，並已下載模型：
```bash
ollama serve
ollama list
```

#### Python 套件安裝失敗
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 記憶體不足
建議處理 < 100MB 的數據集，或先對大型資料集進行抽樣後再分析。

## 📁 專案結構

```
data_analysis_assistant/
├── streamlit_app.py        # 主程式入口（Tab 路由）
├── tabs/                   # 各分頁模組
│   ├── tab_data_preview.py
│   ├── tab_preprocessing.py
│   ├── tab_visualization.py
│   ├── tab_statistics.py
│   ├── tab_variable_analysis.py
│   ├── tab_ml.py
│   ├── tab_psi_monitoring.py
│   ├── tab_rag_management.py
│   └── tab_ai_assistant.py
├── langchain_agent.py      # ReAct Agent
├── rag_manager.py          # ChromaDB RAG
├── ml_models.py            # ML 邏輯
├── visualization.py        # Plotly 圖表
├── risk_metrics.py         # KS / Lift / Gain
├── woe_iv.py               # WOE / IV 計算
├── psi.py                  # PSI 監控
├── model_export.py         # 模型匯出（joblib）
├── config.py               # 設定（支援環境變數覆蓋）
├── requirements.txt
├── install.bat / install.sh
├── start.bat / start.sh
└── docker-compose.yml
```

## ❓ FAQ

**Q: 數據會上傳到雲端嗎？**
A: 使用本地 Ollama 模式時完全離線。使用雲端 LLM 時，提問內容會傳送至該 API 提供商，請注意敏感資料。

**Q: 支援哪些檔案格式？**
A: CSV、TXT（分隔符）、Excel（.xlsx/.xls）、SAS（.sas7bdat）。

**Q: 如何匯出機器學習模型？**
A: 在「機器學習」分頁訓練完成後，點擊「匯出模型」按鈕，即可下載 `.joblib` 檔案。

**Q: 如何更新？**
A: 拉取最新程式碼後，重新執行 `install.bat` 更新依賴即可。
