# 專案架構與說明文件：數據分析小幫手 (Data Analysis Assistant)

## 1. 專案概述
「數據分析小幫手」是一個基於 Python 與 Streamlit 開發的本機端（Local）數據分析應用程式。其主要目的為提供一個圖形化使用者介面（GUI），協助使用者在不編寫程式碼的情況下，進行資料預處理、統計分析、機器學習建模以及模型監控。此外，本專案整合了 LangChain 框架與本地端大型語言模型（Ollama），提供自然語言互動功能來輔助結構化資料的分析與非結構化文件（知識庫）的檢索。

本系統的核心設計理念為「完全在本地連線執行」，所有資料處理與模型運算皆於使用者本身的電腦硬體上完成，不強制串接外部雲端 API，藉此確保商業資料的安全性與隱私。

## 2. 系統核心架構
本專案採用模組化設計，主要由前端介面層、資料處理層、分析與建模層、AI輔助層組成。

### 2.1 前端介面層 (UI Layer)
- **技術選型**：使用 `streamlit` 框架建立互動式的網頁介面。
- **程式入口**：`streamlit_app.py` 負責整體的頁面佈局與路由分配。用戶上傳的資料檔案會被讀取並存入 Streamlit 的 `st.session_state`（工作階段狀態），供所有的子功能模組共用。
- **功能分頁 (Tabs)**：專案依功能拆分為多個獨立的子模組（位於 `tabs/` 目錄下），主要包含：
  - `tab_data_preview.py`：數據預覽與匯總
  - `tab_variable_analysis.py`：變數分析與處理（如 WOE分箱）
  - `tab_visualization.py`：資料視覺化（直方圖、散點圖等基本圖表）
  - `tab_statistics.py`：常見統計分析
  - `tab_ml.py`：機器學習模型訓練與評估
  - `tab_psi_monitoring.py`：模型群體穩定度指標（PSI）監控
  - `tab_rag_management.py`：RAG 機器學習知識庫的文件管理
  - `tab_ai_assistant.py`：AI 小幫手對話介面

### 2.2 資料處理層 (Data Processing Layer)
- **資料載入** (`data_loader.py`)：支援讀取 CSV, TXT, Excel (.xlsx, .xls) 及 SAS (.sas7bdat) 等常見檔案格式。檔案上傳時會透過暫存檔機制安全處理後，再轉換為 Pandas 函式庫的 DataFrame 資料結構。
- **資料預處理** (`data_preprocessing.py`)：
  - 缺失值處理：提供平均值 (mean)、中位數 (median)、眾數 (mode)、刪除含缺失值的列 (drop) 及常數自定義填充等策略。
  - 型別轉換：包含字串轉換為數值、日期格式化轉換、布林值轉換等操作。
  - 特徵工程：支援基礎特徵轉換與處理，供後續機器學習使用。

### 2.3 分析與建模層 (Analysis & Modeling Layer)
- **統計分析** (`data_analysis.py`)：
  - 敘述性統計：快速計算平均數、變異數、極端值、四分位數等。
  - 假設檢定：包含 t檢定、無母數檢定、卡方檢定、單因子變異數分析 (ANOVA)。
  - 相關分析與迴歸：Pearson、Spearman、Kendall的相關係數分析，以及基本的線性迴歸。
- **機器學習模型** (`ml_models.py`)：
  - 支援兩個主要任務：分類 (Classification) 與迴歸 (Regression)。
  - 整合的模型種類包含：邏輯迴歸 (Logistic Regression)、決策樹 (Decision Tree)、隨機森林 (Random Forest)、支援向量機 (SVM)、K-近鄰 (KNN)、樸素貝葉斯 (Naive Bayes)。若是環境中已安裝 XGBoost 與 LightGBM 套件則會自動支援這兩個高效能梯度提升樹模型。
  - 模型評估：涵蓋準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall)、F1 Score、ROC AUC，以及迴歸用的 RMSE、MAE、R² 等多元指標比較。
  - 不平衡資料採樣：可調用 `imbalanced-learn` 套件，提供 SMOTE、ADASYN、Undersample、SMOTETomek 等演算法改善分類樣本數量不균衡狀況。
  - 自動調參：實作基於 `GridSearchCV` 的超參數搜索，幫助找尋最佳模型參數。

### 2.4 AI 輔助與知識庫層 (AI & RAG Layer)
- **AI 代理人 (Agent)** (`langchain_agent.py`)：
  - 利用 LangChain 的 `ReAct` 推理框架，搭配本地端 Ollama（預設使用例如 `qwen2.5:7b` 系列的大型語言模型）建立智能代理人。
  - 此代理人綁定了多個內建工具 (Tools)，如：`DescriptiveStatistics`, `HandleMissingValues`, `TTest`, `CorrelationAnalysis` 等。
  - 當使用者在「AI 助手」分頁輸入自然語言指令時，代理人能夠自主推理，判斷需要執行哪些工具，並對目前載入的 DataFrame 進行運算並解釋結果。
- **RAG 知識庫管理** (`rag_manager.py`)：
  - 搭載了檢索增強生成 (Retrieval-Augmented Generation, RAG) 功能，支援將文字檔、PDF、或 Markdown 等參考資料匯入。
  - 使用 `sentence-transformers` 將文件片段轉化為向量嵌入 (Embeddings)，並透過本地端的 `ChromaDB` 作為向量資料庫來儲存。
  - 提供輔助的 `KnowledgeBase` 工具，讓代理人除了分析數據，還能檢索並閱讀外部知識文件來回答使用者的進階領域問題。

## 3. 程式碼邏輯結構重點
- **狀態管理共享**：充分仰賴 Streamlit 的 `st.session_state` 在不同切換分頁中保持全域變數狀態不遺失（例如目前正在分析的表格 `df`，或是代理人的對話紀錄 `agent_executor`）。
- **解耦與延遲加載 (Lazy Initialization)**：對於依賴較繁重的機器學習套件（例如 xgboost, lightgbm, imbalanced-learn），採取在 `ml_models.py` 中用 `try-except` 與 Lazy Initialization 的設計。此做法確保在未安裝某些擴充套件的輕量環境中，專案依然能正常啟動並使用其餘功能。
- **資料洩漏防範 (Preventing Data Leakage)**：在機器學習的資料切分實作中 (`prepare_data` 函數)，嚴格遵循：先拆分訓練集與測試集 (Train-Test Split)，再各自套用特徵轉換與標準化 (`StandardScaler`, `OneHotEncoder`) 的正規流程。此設計避免了未來的資訊（測試集分佈）提前洩漏到模型的訓練過程中，確保評估指標客觀有效。

## 4. 亮點比較：為何不單純寫程式跑機器學習？ (目標使用情境)
本系統不單追求「演算法準確率」，而是大幅提升解決問題的「綜合成功率」與「效率」。特別適合以下情況：

1. **破除黑盒子，轉化為白話文 (可解釋性)**：傳統 ML 僅給出準確率數字，本系統透過 AI 助手直接將冷冰冰的數據轉化為白話文的商業洞察與建議。
2. **零門檻的 No-Code 普及化**：缺乏程式 (Python/R) 撰寫能力，但需要直觀 GUI 快速進行探索性分析 (EDA) 與建模的業務端人員或初級分析師。
3. **防呆流程，預防人類犯錯**：傳統 ML 容易因前處理不當導致「資料洩漏 (Data Leakage)」。本架構內建標準化流程，強迫執行符合規範的動作，產出更穩健的模型。
4. **絕對的隱私與合規**：資料隱私要求嚴格，必須阻絕將真實業務資料上傳至 ChatGPT 等雲端服務的機密環境。

## 5. 解構 AI 核心架構：LangChain ReAct + RAG
如果需要向他人說明本專案的 AI 技術細節，本系統採用的是**「LangChain ReAct Agent + RAG（檢索增強生成）」混合架構**，包含三層核心技術：

1. **推理層 (Reasoning)**：使用開源大型語言模型（如 Qwen2.5 / Llama 3），透過 LangChain 框架的 **ReAct (Reasoning + Acting)** 機制，讓 AI 不單純是聊天機器人，而是能「邊思考、邊呼叫工具」，自主判斷該執行什麼分析操作的自動化分析員。
2. **工具層 (Tool Use / Function Calling)**：AI 背後綁定了多個分析函數（統計檢定、迴歸、機器學習模型），可以直接對上傳的數據進行運算，確保數據準確而非憑空捏造。
3. **知識層 (RAG)**：使用 **ChromaDB 向量資料庫** + **Sentence-Transformers 嵌入模型**，將企業內部文件向量化儲存。AI 在回答概念性問題時會先檢索關聯知識片段，確保回答有據可依、不產生幻覺 (Hallucination)。

## 6. 總結
「數據分析小幫手」透過 Streamlit 將 Pandas、Scikit-Learn 等常見的 Python 資料科學主流套件封裝為圖形介面，並在此穩固基礎上，外掛了 LangChain 與本地端 LLM（Ollama）以提供自然語言推理與 RAG 技術，構成一個架構清晰、模組化且具高度擴充性的在地化數據分析自動化工作站。
