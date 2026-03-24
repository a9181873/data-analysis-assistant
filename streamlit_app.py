"""
數據分析小幫手 — Chat-First 主應用程式
核心理念：LLM 了解需求後，導流至對應工具模組。
佈局：左側對話 + 右側動態工作區。
"""

import streamlit as st
import pandas as pd
import os
import config
from data_loader import load_data

# Tab 模組（現在作為動態工具面板使用）
from tabs import tab_data_preview, tab_variable_analysis, tab_visualization
from tabs import tab_statistics, tab_ml, tab_psi_monitoring
from tabs import tab_rag_management, tab_ai_assistant

# ─── 頁面配置 ─────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 載入自訂 CSS (外部檔案)
_css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(_css_path):
    with open(_css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 內嵌關鍵 CSS (確保一定套用，即使外部檔案快取未生效)
st.markdown("""
<style>
/* LOL Colors 4714 配色 — #9DC8C8, #58C9B9, #519D9E, #D1B6E1 */
section[data-testid="stSidebar"] {
    background-color: #519D9E !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #D1B6E1 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 6px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.25) !important;
    border-color: #D1B6E1 !important;
    color: #D1B6E1 !important;
}
/* 修正側邊欄檔案上傳區的字體顏色 */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #495057 !important;
}
/* 主要標題色 */
.main h1, .main h2, .main h3 { color: #519D9E !important; }
/* Metric 卡片頂邊線 */
div[data-testid="stMetric"] {
    border-top: 4px solid #58C9B9 !important;
    border-radius: 8px;
    background-color: #ffffff;
}
/* 主按鈕 */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #58C9B9 !important;
    color: #ffffff !important;
    border: none !important;
}
/* 工作區佔位 */
.workspace-empty {
    text-align: center; color: #9E9E9E; margin-top: 60px;
}
.workspace-empty h1 { color: #E0E0E0; font-size: 3rem; }
.workspace-empty h3 { color: #BDBDBD; }
.workspace-empty p  { color: #9E9E9E; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════
_defaults = {
    "df": None,
    "analysis_results": [],
    "agent_executor": None,
    "ml_results": {},
    "column_descriptions": {},
    # Chat-First 專用
    "messages": [],
    "active_module": None,
    "data_profiled": False,
    "last_uploaded_file": None,
    "_trigger_chat_to_dict": False,
    # LLM 選擇器
    "llm_source": "local",         # "local" or "cloud"
    "llm_cloud_model_idx": 0,      # CLOUD_MODELS 的索引
    "llm_cloud_api_keys": {},      # {env_key: key_value} 各提供者分開存
    "llm_local_model": None,
    # ── 人機協作 (Human-AI Collaboration) ──
    "ai_suggested_params": {},      # LLM 建議的工具參數
    "ai_actions": [],               # LLM 建議的可執行動作按鈕
    "ai_context_msg": "",           # LLM 建議理由（供右側面板顯示）
    "tool_result_summary": None,    # 工具執行結果摘要（供回饋給 LLM）
    "_pending_action": None,        # 待執行的 action（點按鈕後觸發）
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 初始化歡迎訊息
if not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 您好！我是您的**ML 分析導師**。\n\n"
                "我會帶您依照標準的機器學習流程，一步一步完成資料分析：\n"
                "1. 📋 **數據預覽** — 了解資料規模與欄位\n"
                "2. 🔬 **探索性分析 (EDA)** — 敘述統計 & 變數處理\n"
                "3. 📊 **數據視覺化** — 圖表挖掘規律\n"
                "4. 📐 **統計 & 特徵工程** — 檢定 & WOE/IV\n"
                "5. 🤖 **模型訓練與預測** — 機器學習建模\n"
                "6. 📡 **模型監控 (PSI)** — 確保模型品質\n\n"
                "請從側邊欄上傳資料，或點擊「🚀 使用範例數據」開始第一步！"
            ),
        }
    ]


# ═══════════════════════════════════════════════
# 工具模組對映表
# ═══════════════════════════════════════════════
MODULE_MAP = {
    "data_preview":      ("📋 數據預覽",              tab_data_preview),
    "variable_analysis": ("🔬 變數分析與處理",         tab_variable_analysis),
    "visualization":     ("📊 資料圖表化",             tab_visualization),
    "statistics":        ("📐 統計分析",               tab_statistics),
    "ml":                ("🤖 機器學習 (Machine Learning)", tab_ml),
    "psi_monitoring":    ("📡 模型監控 (PSI)",          tab_psi_monitoring),
    "rag_management":    ("📚 知識庫管理",              tab_rag_management),
    "ai_assistant":      ("🖥️ AI 狀態儀表板",           tab_ai_assistant),
}


# ═══════════════════════════════════════════════
# 意圖路由 (Intent Router)
# ═══════════════════════════════════════════════
def route_intent(user_input: str):
    text = user_input.lower()

    if any(k in text for k in ["預測", "機器學習", "模型訓練", "分類", "迴歸",
                                "ml", "xgboost", "random forest", "shap",
                                "訓練", "建模", "logistic"]):
        return "ml", (
            "好的！我將為您開啟 **『機器學習』** 模組。\n\n"
            "請在右側工作區選擇預測目標與特徵，然後點擊開始訓練。"
        )

    if any(k in text for k in ["圖", "視覺化", "散點", "直方", "箱型", "長條",
                                "圓餅", "配對", "chart", "plot", "畫"]):
        return "visualization", (
            "沒問題！我已幫您開啟 **『資料圖表化』** 面板。\n\n"
            "請在右側選擇圖表類型與對應欄位。"
        )

    if any(k in text for k in ["統計", "t 檢定", "t檢定", "迴歸分析", "卡方",
                                "anova", "相關", "woe", "iv", "敘述統計"]):
        return "statistics", (
            "了解！我已開啟 **『統計分析』** 模組。\n\n"
            "請在右側選擇分析類型（敘述統計、t 檢定、WOE/IV 等）。"
        )

    if any(k in text for k in ["變數", "欄位", "缺失", "遺漏", "型態", "轉換",
                                "離群值", "補值", "替換", "資料字典"]):
        return "variable_analysis", (
            "右側已為您開啟 **『變數分析與處理』** 面板。\n\n"
            "可以查看每個欄位的統計摘要、處理遺漏值或進行型態轉換。"
        )

    if any(k in text for k in ["預覽", "看看資料", "資料概覽", "匯出", "下載",
                                "preview", "overview"]):
        return "data_preview", (
            "已為您開啟 **『數據預覽』** 面板。\n\n"
            "右側可以查看完整數據、欄位資訊，也可以匯出 CSV / Excel / JSON。"
        )

    if any(k in text for k in ["psi", "監控", "偏移", "穩定性", "drift"]):
        return "psi_monitoring", (
            "好的！我已開啟 **『模型監控 (PSI)』** 模組。\n\n"
            "請在右側上傳監控期資料，與基準期資料進行比較。"
        )

    if any(k in text for k in ["知識庫", "rag", "文件", "向量", "文檔"]):
        return "rag_management", (
            "已為您開啟 **『知識庫管理』** 面板。\n\n"
            "您可以上傳文字檔到向量知識庫。"
        )

    if any(k in text for k in ["狀態", "服務", "ollama", "模型狀態",
                                "ai狀態", "ai 狀態"]):
        return "ai_assistant", (
            "好的！我已開啟 **『AI 狀態儀表板』**。\n\n"
            "可以查看 Ollama 服務連線、已安裝模型與知識庫狀態。"
        )

    return None, (
        "您可以更具體地描述想做的事，例如：\n"
        "- 「我想**預測**違約」→ Step 5 機器學習\n"
        "- 「幫我畫**散點圖**」→ Step 3 視覺化\n"
        "- 「做**敘述統計**」→ Step 4 統計分析\n"
        "- 「處理缺失值」→ Step 2 變數分析\n\n"
        "也可以使用下方的 **ML 分析流程** 按鈕直接跳到對應步驟 👇"
    )


def _add_msg(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def _build_data_context() -> str:
    """根據目前載入的 DataFrame 建立精簡的數據摘要（限制最多 15 個欄位以節省 Token）。"""
    df = st.session_state.df
    if df is None:
        return ""

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_total = int(df.isnull().sum().sum())

    MAX_COLS = 15
    shown_cols = list(df.columns[:MAX_COLS])
    col_names = ', '.join(shown_cols)
    if n_cols > MAX_COLS:
        col_names += f' ... (共 {n_cols} 欄)'

    context = (
        f"資料: {n_rows} 筆 x {n_cols} 欄\n"
        f"數值欄位({len(numeric_cols)}): {', '.join(numeric_cols[:MAX_COLS])}\n"
        f"類別欄位({len(cat_cols)}): {', '.join(cat_cols[:MAX_COLS]) if cat_cols else '無'}\n"
        f"缺失值: {missing_total}\n"
        f"欄位: {col_names}"
    )
    return context


def _ask_llm_structured(question: str) -> dict:
    """呼叫 LLM 產生結構化 JSON 回應（含 reply, target_module, suggested_params, actions）。
    回傳 dict，即使 LLM 回傳非法 JSON 也會 fallback 為純文字。"""
    import re, json
    _FALLBACK = {"reply": "", "target_module": None, "suggested_params": {}, "actions": []}

    try:
        # 確保 agent 已建立
        df = st.session_state.df
        df_id = id(df)
        if (st.session_state.agent_executor is None
                or st.session_state.get("_agent_df_id") != df_id):
            from langchain_agent import create_agent_executor
            st.session_state.agent_executor = create_agent_executor(df)
            st.session_state._agent_df_id = df_id

        # ── 建構對話歷史 (最近 10 輪) ──
        _SYSTEM_MSG_MARKERS = ["Step 1 完成", "已進入 **", "已切換至 **", "⚙️ 自動執行"]
        history_messages = []
        recent = st.session_state.messages[-10:]
        for msg in recent:
            content = msg["content"]
            if msg["role"] == "assistant" and any(m in content for m in _SYSTEM_MSG_MARKERS):
                continue
            if msg["role"] == "user":
                history_messages.append(("user", content))
            elif msg["role"] == "assistant":
                history_messages.append(("assistant", content))

        # ── 注入數據上下文 ──
        data_context = _build_data_context()
        augmented = question
        if data_context:
            augmented = f"{data_context}\n\n使用者問題: {question}"

        # ── RAG 增強 ──
        try:
            from rag_manager import get_chroma_collection, query_rag
            _, collection = get_chroma_collection()
            if collection.count() > 0:
                context = query_rag(question, collection, n_results=3)
                if context != "找不到相關文檔。":
                    augmented = f"以下是知識庫中的參考資料:\n{context}\n\n{augmented}"
        except Exception:
            pass

        # ── 呼叫 agent (含歷史) ──
        agent = st.session_state.agent_executor
        all_messages = history_messages + [("user", augmented)]
        result = agent.invoke({"messages": all_messages})

        messages = result.get("messages", [])
        raw = messages[-1].content if messages else str(result)

        # 清除 <think> 標籤
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        # ── 嘗試解析 JSON ──
        # 移除可能的 markdown code block 包裹
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        # 嘗試找到 JSON 物件
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict) and "reply" in parsed:
                # 確保所有 key 都存在
                parsed.setdefault("target_module", None)
                parsed.setdefault("suggested_params", {})
                parsed.setdefault("actions", [])
                return parsed

        # JSON 解析失敗 → fallback：用原有的 keyword routing
        module_key, _ = route_intent(question)
        return {"reply": raw, "target_module": module_key, "suggested_params": {}, "actions": []}

    except Exception as e:
        if getattr(config, "USE_CLOUD_LLM", False):
            hint = (f"請確認 API Key 正確且與所選提供者匹配。\n\n"
                    f"目前模型: `{config.LLM_MODEL}` | "
                    f"Base URL: `{getattr(config, 'CLOUD_BASE_URL', '')}`")
        else:
            hint = f"請確認 Ollama 服務正在運行且模型 `{config.LLM_MODEL}` 已安裝。"
        fallback = _FALLBACK.copy()
        fallback["reply"] = f"⚠️ AI 回應失敗: {e}\n\n{hint}\n\n您也可以使用下方的快捷按鈕直接操作工具模組。"
        return fallback


def _execute_action(action: dict):
    """自動執行 LLM 建議的 action — 呼叫程式內建工具並將結果回饋到聊天。"""
    module = action.get("module", "")
    params = action.get("params", {})
    label = action.get("label", "執行動作")
    df = st.session_state.df
    if df is None:
        _add_msg("assistant", "⚠️ 尚未載入資料，無法執行此動作。")
        return

    result_text = ""
    try:
        if module == "statistics":
            analysis_type = params.get("analysis_type", "敘述統計")
            target_columns = params.get("target_columns", [])
            selected_df = df[target_columns] if target_columns else df

            if analysis_type == "敘述統計":
                from data_analysis import descriptive_statistics
                result = descriptive_statistics(selected_df)
                # 也產生格式化摘要
                num_desc = selected_df.describe(include="number")
                cols_str = ", ".join(target_columns) if target_columns else "全部欄位"
                result_text = f"**📐 敘述統計 — {cols_str}**\n\n"
                if not num_desc.empty:
                    result_text += num_desc.T.to_markdown() + "\n\n"
                missing = selected_df.isnull().sum()
                if missing.sum() > 0:
                    result_text += f"⚠️ 缺失值: {missing[missing > 0].to_dict()}\n"
                else:
                    result_text += "✅ 無缺失值\n"
                st.session_state.analysis_results.append(("敘述統計", result))

            elif analysis_type == "相關分析":
                from data_analysis import perform_correlation_analysis
                method = params.get("method", "pearson")
                cols = target_columns or df.select_dtypes(include=["number"]).columns.tolist()
                if len(cols) >= 2:
                    result = perform_correlation_analysis(df, cols, method)
                    result_text = f"**📐 相關分析 ({method})**\n\n{result}\n"
                else:
                    result_text = "⚠️ 需要至少 2 個數值欄位進行相關分析。\n"

            elif analysis_type in ("t 檢定", "ANOVA 變異數分析", "卡方檢定", "線性迴歸", "WOE/IV 分析"):
                # 這些需要更多參數，導向工具面板讓使用者微調
                st.session_state.active_module = "statistics"
                st.session_state.ai_suggested_params = params
                result_text = f"已開啟 **統計分析** 面板並預設為「{analysis_type}」，請在右側確認參數後執行。"

            else:
                st.session_state.active_module = "statistics"
                st.session_state.ai_suggested_params = params
                result_text = f"已開啟 **統計分析** 面板，分析類型：{analysis_type}"

        elif module == "ml":
            model_name = params.get("model_name", "Random Forest")
            target_col = params.get("target_col")
            task_type = params.get("task_type", "classification")
            feature_cols = params.get("feature_cols", [])

            if not target_col:
                # 猜測目標欄位（取唯一值最少的數值欄位）
                for c in df.columns:
                    if df[c].nunique() <= 5 and pd.api.types.is_numeric_dtype(df[c]):
                        target_col = c
                        break
            if not target_col:
                target_col = df.columns[-1]

            if not feature_cols:
                feature_cols = [c for c in df.columns if c != target_col]

            # 自動訓練模型
            from ml_models import prepare_data, train_single_model, train_regression_model, apply_balancing
            try:
                _task = config.ML_TASK_REGRESSION if task_type == "regression" else config.ML_TASK_CLASSIFICATION
                X_train, X_test, y_train, y_test, le, preprocessor = prepare_data(
                    df, target_col, feature_cols, config.DEFAULT_TEST_SIZE, _task)

                if task_type == "regression":
                    res = train_regression_model(model_name, X_train, y_train, X_test, y_test)
                    result_text = (
                        f"**🤖 {model_name} 迴歸模型訓練完成！**\n\n"
                        f"| 指標 | 數值 |\n|------|------|\n"
                        f"| R² | {res['r2']:.4f} |\n"
                        f"| RMSE | {res['rmse']:.4f} |\n"
                        f"| MAE | {res['mae']:.4f} |\n"
                    )
                else:
                    X_train_bal, y_train_bal = apply_balancing(X_train, y_train, "none")
                    res = train_single_model(model_name, X_train_bal, y_train_bal, X_test, y_test, le)
                    result_text = (
                        f"**🤖 {model_name} 分類模型訓練完成！**\n\n"
                        f"| 指標 | 數值 |\n|------|------|\n"
                        f"| Accuracy | {res['accuracy']:.4f} |\n"
                        f"| Precision | {res['precision']:.4f} |\n"
                        f"| Recall | {res['recall']:.4f} |\n"
                        f"| F1 Score | {res['f1']:.4f} |\n"
                    )
                    if res.get('roc_auc') is not None:
                        result_text += f"| AUC | {res['roc_auc']:.4f} |\n"

                # 存入 ML 結果供面板展示
                feature_names_out = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else feature_cols
                st.session_state.ml_results_single = {
                    "type": task_type, "res": res, "model_name": model_name,
                    "feature_names_out": feature_names_out, "le": le,
                    "X_train_bal": X_train_bal if task_type != "regression" else X_train,
                    "X_test_df": X_test, "y_test": y_test,
                    "preprocessor": preprocessor,
                }
                st.session_state.active_module = "ml"
                st.session_state.ai_suggested_params = params
                result_text += "\n📊 詳細結果已在右側 **機器學習** 面板中展示。"

            except Exception as train_err:
                result_text = f"⚠️ 模型訓練失敗: {train_err}\n\n已開啟 ML 面板，請手動調整參數後重試。"
                st.session_state.active_module = "ml"
                st.session_state.ai_suggested_params = params

        elif module == "visualization":
            chart_type = params.get("chart_type", "散點圖")
            x_col = params.get("x_col")
            y_col = params.get("y_col")

            # 自動產生圖表
            from visualization import (plot_histogram, plot_scatter, plot_boxplot,
                                       plot_bar_chart, plot_pie_chart, plot_correlation_heatmap)
            fig = None
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if chart_type == "散點圖" and x_col and y_col:
                fig = plot_scatter(df, x_col, y_col)
                result_text = f"**📊 散點圖 — {x_col} vs {y_col}**"
            elif chart_type == "直方圖" and x_col:
                fig = plot_histogram(df, x_col, 30)
                result_text = f"**📊 直方圖 — {x_col}**"
            elif chart_type == "箱型圖" and (x_col or y_col):
                col = y_col or x_col
                fig = plot_boxplot(df, col)
                result_text = f"**📊 箱型圖 — {col}**"
            elif chart_type == "相關熱圖":
                cols = [x_col, y_col] if x_col and y_col else numeric_cols[:6]
                cols = [c for c in cols if c in numeric_cols]
                if len(cols) >= 2:
                    fig = plot_correlation_heatmap(df, cols)
                    result_text = f"**📊 相關熱圖 — {', '.join(cols)}**"
            else:
                # 導向面板
                st.session_state.active_module = "visualization"
                st.session_state.ai_suggested_params = params
                result_text = f"已開啟 **資料圖表化** 面板，建議圖表類型：{chart_type}"

            if fig is not None:
                st.session_state["_auto_fig"] = fig
                st.session_state.active_module = "visualization"
                st.session_state.ai_suggested_params = params

        elif module == "variable_analysis":
            st.session_state.active_module = "variable_analysis"
            st.session_state.ai_suggested_params = params
            result_text = "已開啟 **變數分析與處理** 面板。"

        elif module == "data_preview":
            st.session_state.active_module = "data_preview"
            result_text = "已開啟 **數據預覽** 面板。"

        elif module == "psi_monitoring":
            st.session_state.active_module = "psi_monitoring"
            result_text = "已開啟 **模型監控 (PSI)** 面板。"

        else:
            result_text = f"未知的模組: {module}"

    except Exception as e:
        result_text = f"⚠️ 自動執行失敗: {e}"

    # 將執行結果寫入聊天
    if result_text:
        _add_msg("assistant", f"⚙️ 自動執行 **{label}** 完成：\n\n{result_text}")
        # 存入 tool_result_summary 供後續 LLM 回饋
        st.session_state.tool_result_summary = result_text


def _feedback_from_tool_result():
    """如果有工具執行結果摘要，呼叫 LLM 做進一步評論與建議。"""
    summary = st.session_state.get("tool_result_summary")
    if not summary:
        return
    st.session_state.tool_result_summary = None  # 清除，避免重複觸發

    feedback_prompt = (
        f"以下是剛才自動執行工具的結果：\n{summary}\n\n"
        f"請根據結果簡短評論（2-3 句），並建議使用者下一步可以做什麼。"
        f"reply 中直接給出評論和建議即可，actions 可以給 1-2 個下一步建議。"
    )
    with st.spinner("🤖 AI 正在分析結果..."):
        result = _ask_llm_structured(feedback_prompt)
    _add_msg("assistant", result["reply"])
    st.session_state.ai_actions = result.get("actions", [])
    if result.get("target_module"):
        st.session_state.ai_suggested_params = result.get("suggested_params", {})


def _profile_data():
    """資料載入後，AI 主動產生資料摘要與下一步引導。"""
    if st.session_state.df is None or st.session_state.data_profiled:
        return

    df = st.session_state.df
    n_rows, n_cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col_list = ", ".join(f"`{c}`" for c in df.columns[:10])
    if len(df.columns) > 10:
        col_list += f" ... (共 {n_cols} 欄)"

    missing_warning = ""
    if missing_total > 0:
        missing_warning = f"\n⚠️ 發現 **{missing_total}** 個缺失值，建議在 Step 2 中處理。"

    summary = (
        f"✅ 資料已成功載入！\n\n"
        f"| 項目 | 數值 |\n"
        f"|------|------|\n"
        f"| 資料筆數 | **{n_rows:,}** 筆 |\n"
        f"| 變數數量 | **{n_cols}** 個 |\n"
        f"| 數值型欄位 | {len(numeric_cols)} 個 |\n"
        f"| 類別型欄位 | {len(cat_cols)} 個 |\n"
        f"| 缺失值 | {missing_total} 個 |\n"
        f"{missing_warning}\n"
        f"\n欄位清單：{col_list}\n\n"
    )
    
    # 呼叫 LLM 獲取動態洞察
    insight_prompt = (
        f"資料已經載入，包含 {n_rows} 筆資料，{n_cols} 個欄位。其中有 {missing_total} 個缺失值。"
        f"請用專業數據顧問的語氣，簡短描述這份資料的基本樣貌，並明確給出若要進行機器學習「建模」前，"
        f"該如何進行資料調整（例如：針對缺失值該如何補值、數值欄位是否需要標準化、類別欄位編碼等具體建議）。"
    )
    
    with st.spinner("AI 顧問正在速覽資料樣貌..."):
        llm_result = _ask_llm_structured(insight_prompt)
        llm_insight = llm_result["reply"]
        
    full_msg = f"{summary}\n---\n**🧠 顧問洞察與建模建議**\n\n{llm_insight}\n\n您可以點擊上方的 **🗺️ ML 分析流程** 或直接在此詢問我下一步！"
    _add_msg("assistant", full_msg)
    st.session_state.ai_actions = llm_result.get("actions", [])
    st.session_state.data_profiled = True


# ═══════════════════════════════════════════════
# 側邊欄：LLM 模型選擇器（須在 _profile_data 之前，確保上傳資料時已套用使用者選定的模型）
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 LLM 模型設定")

    llm_source = st.radio(
        "運算模式",
        ["🖥️ 地端 (Local)", "☁️ 雲端 (Cloud)"],
        index=0 if st.session_state.llm_source == "local" else 1,
        key="llm_source_radio",
        horizontal=True,
    )
    _is_cloud = "雲端" in llm_source

    if _is_cloud:
        st.session_state.llm_source = "cloud"

        # 扁平模型列表：一個下拉選單直接選模型
        _models = config.CLOUD_MODELS
        _display = [f"{m['rating']} {m['id']}  ({m['provider']})" for m in _models]
        _cur_idx = st.session_state.llm_cloud_model_idx
        if _cur_idx >= len(_models):
            _cur_idx = 0

        _sel_label = st.selectbox("模型 (Model)", _display, index=_cur_idx, key="cloud_model_select")
        _sel_idx = _display.index(_sel_label)
        st.session_state.llm_cloud_model_idx = _sel_idx
        _sel_model = _models[_sel_idx]

        st.caption(f"📋 {_sel_model['note']}")

        # API Key 輸入（根據選定模型的 env_key 自動切換）
        _env_key = _sel_model["env_key"]
        _existing_key = os.environ.get(_env_key, "")
        if _existing_key:
            st.success(f"✅ 已偵測到環境變數 `{_env_key}`")
            st.session_state.llm_cloud_api_keys[_env_key] = _existing_key
        else:
            _saved_key = st.session_state.llm_cloud_api_keys.get(_env_key, "")
            _key_input = st.text_input(
                f"🔑 {_sel_model['provider']} API Key",
                type="password",
                value=_saved_key,
                placeholder=f"請輸入 {_env_key}",
                key="api_key_input",
            )
            st.session_state.llm_cloud_api_keys[_env_key] = _key_input
            if not _key_input:
                st.warning(f"⚠️ 請輸入 {_sel_model['provider']} API Key")

        _active_key = st.session_state.llm_cloud_api_keys.get(_env_key, "")

        # 驗證 API Key（真實發送請求）
        if _active_key:
            if st.button("🔑 驗證 API Key", key="verify_api_key", use_container_width=True):
                import requests as _req
                _test_url = f"{_sel_model['base_url']}/chat/completions"
                _headers = {
                    "Authorization": f"Bearer {_active_key}",
                    "Content-Type": "application/json",
                }
                if "openrouter.ai" in _sel_model["base_url"]:
                    _headers["HTTP-Referer"] = "https://github.com/a9181873/data-analysis-assistant"
                    _headers["X-Title"] = "Data Analysis Assistant"
                _payload = {
                    "model": _sel_model["id"],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                }
                try:
                    _r = _req.post(_test_url, headers=_headers, json=_payload, timeout=15)
                    _detail = ""
                    try:
                        _detail = _r.json().get("error", {}).get("message", "")
                    except Exception:
                        _detail = _r.text[:200] if _r.text else ""
                    if _r.status_code == 200:
                        st.success("✅ API Key 驗證成功！")
                    elif _r.status_code == 401:
                        st.error(f"❌ 驗證失敗 (401): {_detail or 'API Key 無效或已過期'}")
                    elif _r.status_code == 402:
                        st.error(f"❌ 餘額不足 (402): {_detail or '請至提供者網站儲值'}")
                    else:
                        st.warning(f"⚠️ 狀態碼 {_r.status_code}: {_detail}")
                except Exception as _e:
                    st.error(f"❌ 連線失敗: {_e}")

        st.caption("🔒 API Key 僅存於記憶體，關閉頁面即消失，不會寫入任何檔案。")

        # 動態更新 config
        config.USE_CLOUD_LLM = True
        config.LLM_MODEL = _sel_model["id"]
        config.CLOUD_API_KEY = _active_key
        config.CLOUD_BASE_URL = _sel_model["base_url"]

    else:
        st.session_state.llm_source = "local"
        config.USE_CLOUD_LLM = False

        # 偵測本地 Ollama 可用模型
        _ollama_models = []
        try:
            import requests
            resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                _ollama_models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass

        if _ollama_models:
            default_idx = 0
            current = st.session_state.llm_local_model or config.LLM_MODEL
            if current in _ollama_models:
                default_idx = _ollama_models.index(current)

            selected_local = st.selectbox(
                "本地模型 (Local Model)",
                _ollama_models,
                index=default_idx,
                key="local_model_select",
            )
            st.session_state.llm_local_model = selected_local
            config.LLM_MODEL = selected_local
            st.success(f"✅ Ollama 已連線，{len(_ollama_models)} 個模型可用")
        else:
            st.error("❌ 無法連線 Ollama，請確認服務已啟動")
            st.code("ollama serve", language="bash")
            st.caption(f"目前預設模型: {config.LLM_MODEL}")

    # 當模型來源變更時，重置 agent
    _current_llm_sig = f"{st.session_state.llm_source}:{config.LLM_MODEL}"
    if st.session_state.get("_llm_sig") != _current_llm_sig:
        st.session_state.agent_executor = None
        st.session_state._llm_sig = _current_llm_sig

    st.markdown("---")


# ═══════════════════════════════════════════════
# 主動數據摘要
# ═══════════════════════════════════════════════
_profile_data()

# ═══════════════════════════════════════════════
# 待執行 Action 處理 (按鈕觸發後在此 rerun 時執行)
# ═══════════════════════════════════════════════
if st.session_state.get("_pending_action"):
    _act = st.session_state._pending_action
    st.session_state._pending_action = None
    _execute_action(_act)
    # 執行完後呼叫 LLM 回饋
    _feedback_from_tool_result()
    st.rerun()


# ═══════════════════════════════════════════════
# 對話 → 資料字典 (由按鈕觸發)
# ═══════════════════════════════════════════════
if st.session_state.get("_trigger_chat_to_dict"):
    st.session_state["_trigger_chat_to_dict"] = False
    _df = st.session_state.df
    if _df is not None:
        import re as _re2
        import json as _json2
        # 收集對話中 assistant 的訊息
        _chat_texts = []
        for _m in st.session_state.messages:
            if _m["role"] == "assistant":
                _clean = _re2.sub(r'<think>.*?</think>', '', _m["content"], flags=_re2.DOTALL).strip()
                _chat_texts.append(_clean)
        _all_chat = "\n\n".join(_chat_texts[-6:])  # 取最近 6 則 assistant 訊息

        _col_list = list(_df.columns)
        _extract_prompt = (
            f"以下是 AI 架構師與使用者的對話內容，其中可能包含對資料欄位的描述與解釋。\n"
            f"資料集的欄位名稱為：{_col_list}\n\n"
            f"--- 對話內容 ---\n{_all_chat}\n--- 結束 ---\n\n"
            f"請從對話中擷取每個欄位的描述。專業術語請同時提供繁體中文和英文。\n"
            f"請嚴格以 JSON 格式回覆，格式如下（不要加 markdown code block）：\n"
            f'{{"欄位名": "描述文字", ...}}\n'
            f"若對話中未提及某欄位，請跳過該欄位。只回覆 JSON，不要加任何其他文字。"
        )
        with st.spinner("🤖 AI 正在從對話中擷取變數描述⋯"):
            try:
                if getattr(config, "USE_CLOUD_LLM", False) and getattr(config, "CLOUD_API_KEY", ""):
                    from langchain_openai import ChatOpenAI
                    _extra_headers = {}
                    if "openrouter.ai" in getattr(config, "CLOUD_BASE_URL", ""):
                        _extra_headers = {
                            "HTTP-Referer": "https://github.com/a9181873/data-analysis-assistant",
                            "X-Title": "Data Analysis Assistant",
                        }
                    _llm = ChatOpenAI(
                        model=config.LLM_MODEL,
                        api_key=config.CLOUD_API_KEY,
                        base_url=config.CLOUD_BASE_URL,
                        timeout=60,
                        default_headers=_extra_headers or None,
                    )
                    _resp = _llm.invoke(_extract_prompt).content.strip()
                else:
                    from langchain_ollama import OllamaLLM
                    _llm = OllamaLLM(
                        model=config.LLM_MODEL,
                        base_url=config.OLLAMA_BASE_URL,
                        timeout=60,
                    )
                    _resp = _llm.invoke(_extract_prompt).strip()

                # 清理可能的 markdown code block
                _resp = _re2.sub(r'^```json\s*', '', _resp)
                _resp = _re2.sub(r'\s*```$', '', _resp)
                _parsed = _json2.loads(_resp)
                if isinstance(_parsed, dict):
                    _count = 0
                    for _k, _v in _parsed.items():
                        if _k in _col_list and isinstance(_v, str) and _v.strip():
                            st.session_state.column_descriptions[_k] = _v.strip()
                            _count += 1
                    _add_msg("assistant",
                             f"✅ 已從對話中擷取 **{_count}** 個欄位描述存入資料字典！\n\n"
                             f"前往 **🔬 變數分析與處理** 可查看並匯出。")
                else:
                    _add_msg("assistant", "⚠️ AI 回傳格式異常，請重試。")
            except Exception as _e:
                _add_msg("assistant", f"⚠️ 擷取失敗: {_e}")
        st.rerun()


# ═══════════════════════════════════════════════
# 主內容區：左側對話 ｜ 右側動態工作區
# ═══════════════════════════════════════════════
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.markdown("---")

# 增強資料視覺化與工作區版面 (比例 1:1.8)
col_chat, col_work = st.columns([1, 1.8])

# ─── 左側：對話區與控制項 ──────────────────────────
with col_chat:
    # --- 工作流程與資料上傳 (改放在對話框上方) ---
    with st.expander("🛠️ 資料載入與功能選擇", expanded=(st.session_state.df is None)):
        uploaded_file = st.file_uploader(
            "選擇數據文件",
            type=config.SUPPORTED_FILE_TYPES,
            help="支援 CSV, TXT, Excel, SAS 格式",
        )

        if uploaded_file is not None:
            current_file_info = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("last_uploaded_file") != current_file_info:
                import tempfile
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    df = load_data(tmp_path)
                    st.session_state.df = df
                    st.session_state.data_profiled = False
                    st.session_state.last_uploaded_file = current_file_info
                    st.success(f"成功載入 {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"載入失敗: {str(e)}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

        if st.session_state.df is not None:
            _df = st.session_state.df
            st.caption(f"載入完成: {len(_df)} 行 x {len(_df.columns)} 列 | 缺失值: {_df.isnull().sum().sum()}")
        else:
            if st.button("🚀 使用範例數據", use_container_width=True):
                sample_data = pd.DataFrame({
                    "年齡": [25, 30, 35, 40, 45, 50, 55, 60],
                    "收入": [50000, 60000, 70000, 80000, 90000, 85000, 75000, 65000],
                    "支出": [30000, 35000, 40000, 45000, 50000, 48000, 42000, 38000],
                    "負債比": [0.2, 0.3, 0.15, 0.4, 0.1, 0.35, 0.25, 0.5],
                    "違約": [0, 0, 0, 1, 0, 1, 0, 1],
                    "城市": ["台北", "台中", "高雄", "台南", "桃園", "台北", "台中", "高雄"],
                })
                st.session_state.df = sample_data
                st.session_state.data_profiled = False
                st.session_state.last_uploaded_file = "sample_data"
                st.rerun()

        st.markdown("---")
        st.markdown("**🗺️ ML 分析流程**")
        ml_workflow = [
            ("📋 預覽", "data_preview"), ("🔬 EDA", "variable_analysis"),
            ("📊 圖表", "visualization"), ("📐 統計", "statistics"),
            ("🤖 建模", "ml"), ("📡 PSI", "psi_monitoring")
        ]
        
        # 使用 3x2 格子排版
        cols = st.columns(3)
        for i, (lbl, key) in enumerate(ml_workflow):
            with cols[i % 3]:
                if st.button(lbl, key=f"wf_{key}", use_container_width=True):
                    st.session_state.active_module = key
                    _add_msg("assistant", f"已進入 **{MODULE_MAP[key][0]}**。")
                    st.rerun()
                    
        other_tools = [("📚 知識庫", "rag_management"), ("🖥️ 系統狀態", "ai_assistant")]
        cols2 = st.columns(2)
        for i, (lbl, key) in enumerate(other_tools):
            with cols2[i]:
                if st.button(lbl, key=f"ot_{key}", use_container_width=True):
                    st.session_state.active_module = key
                    _add_msg("assistant", f"已切換至 **{MODULE_MAP[key][0]}** 工作區。")
                    st.rerun()

    st.subheader("💬 與 AI 架構師對話")

    # 對話歷程 (用固定高度容器可捲動)
    chat_box = st.container(height=420)
    with chat_box:
        import re
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    clean_content = re.sub(r'<think>.*?</think>', '', msg["content"], flags=re.DOTALL).strip()
                    st.markdown(clean_content)
                else:
                    st.markdown(msg["content"])

    # ── AI 建議的行動按鈕（顯示在對話框下方）──
    _ai_actions = st.session_state.get("ai_actions", [])
    if _ai_actions and st.session_state.df is not None:
        st.caption("🤖 AI 建議的操作（點擊將自動執行）：")
        _act_cols = st.columns(min(len(_ai_actions), 3))
        for _i, _act in enumerate(_ai_actions):
            with _act_cols[_i % 3]:
                if st.button(_act.get("label", "執行"), key=f"ai_act_{_i}",
                             use_container_width=True, type="primary"):
                    st.session_state._pending_action = _act
                    st.session_state.ai_actions = []  # 清除按鈕
                    st.rerun()

    # 匯出按鈕列（橫向排列於對話框下方）
    if st.session_state.messages:
        import re as _re
        from datetime import datetime as _dt
        _lines = [f"# AI 架構師對話紀錄\n", f"匯出時間: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        if st.session_state.df is not None:
            _d = st.session_state.df
            _lines.append(f"資料集: {len(_d)} 行 x {len(_d.columns)} 列\n")
        _lines.append("---\n")
        for _m in st.session_state.messages:
            _role = "🧑 使用者" if _m["role"] == "user" else "🤖 AI 架構師"
            _content = _re.sub(r'<think>.*?</think>', '', _m["content"], flags=_re.DOTALL).strip()
            _lines.append(f"### {_role}\n\n{_content}\n\n---\n")
        _md_text = "\n".join(_lines)

        _exp1, _exp2 = st.columns(2)
        with _exp1:
            st.download_button(
                "📥 匯出對話",
                data=_md_text.encode("utf-8"),
                file_name=f"chat_export_{_dt.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with _exp2:
            if st.session_state.df is not None:
                if st.button("📖 對話→資料字典", use_container_width=True,
                             help="讓 AI 從對話中擷取變數描述，存入資料字典"):
                    st.session_state["_trigger_chat_to_dict"] = True
                    st.rerun()

    # 快速建議按鈕 (有資料時才顯示)
    if st.session_state.df is not None:
        st.caption("💡 快速開始：")
        sg1, sg2 = st.columns(2)
        suggestions = [
            ("📊 資料視覺化",    "幫我做視覺化分析"),
            ("📐 敘述統計",      "幫我做敘述統計"),
            ("🤖 建立預測模型",  "我想建立機器學習預測模型"),
            ("🔬 變數處理",      "我想看變數分析與處理"),
        ]
        for i, (btn_label, btn_text) in enumerate(suggestions):
            col = sg1 if i % 2 == 0 else sg2
            if col.button(btn_label, key=f"suggest_{i}", use_container_width=True):
                _add_msg("user", btn_text)
                with st.spinner("AI 顧問分析中..."):
                    result = _ask_llm_structured(btn_text)
                _add_msg("assistant", result["reply"])
                if result.get("target_module"):
                    st.session_state.active_module = result["target_module"]
                    st.session_state.ai_suggested_params = result.get("suggested_params", {})
                    st.session_state.ai_context_msg = result["reply"]
                st.session_state.ai_actions = result.get("actions", [])
                st.rerun()

    # 對話輸入框
    if prompt := st.chat_input("告訴我您想分析什麼？ (例如：我想預測違約)"):
        _add_msg("user", prompt)

        if st.session_state.df is None:
            _add_msg("assistant", "⚠️ 尚未載入資料。請先從側邊欄上傳數據文件或使用範例數據！")
        else:
            # 呼叫 LLM → 結構化回應
            with st.spinner("AI 顧問分析中..."):
                result = _ask_llm_structured(prompt)
            _add_msg("assistant", result["reply"])

            # 處理結構化結果
            if result.get("target_module"):
                st.session_state.active_module = result["target_module"]
                st.session_state.ai_suggested_params = result.get("suggested_params", {})
                st.session_state.ai_context_msg = result["reply"]
            st.session_state.ai_actions = result.get("actions", [])

        st.rerun()


# ─── 右側：動態工作區 ─────────────────────────
with col_work:
    active = st.session_state.active_module

    if active and active in MODULE_MAP:
        module_label, module_ref = MODULE_MAP[active]
        st.subheader(f"🛠️ {module_label}")

        if st.session_state.df is not None:
            module_ref.render(st.session_state.df)
        else:
            st.warning("⚠️ 請先載入資料後才能使用此工具。")
    else:
        # 空白狀態 — 使用原生元件避免 unsafe_allow_html DOM 衝突
        st.subheader("🛠️ 動態工作區")
        if st.session_state.df is None:
            st.markdown("")
            _ws = st.container()
            with _ws:
                st.markdown("### 📂")
                st.markdown("#### 歡迎使用數據分析小幫手")
                st.markdown(
                    "全本機運行的數據分析平台，資料不離開你的電腦。\n\n"
                    "- 多格式支援: CSV, TXT, Excel, SAS\n"
                    "- 互動式圖表 (Plotly) + 統計分析\n"
                    "- 8+ 機器學習模型 + SHAP 解釋\n"
                    "- LangChain + Ollama 智能問答\n\n"
                    "**👈 從側邊欄上傳資料或使用範例數據開始**"
                )
        else:
            st.markdown("")
            _ws = st.container()
            with _ws:
                st.markdown("### ✨")
                st.markdown("#### 資料已就緒")
                st.markdown(
                    "請在左側對話框告訴我您的分析需求，\n"
                    "或使用快捷按鈕開啟工具。\n"
                    "對應的分析面板將在此處展開。"
                )


# ─── 頁腳 ─────────────────────────────────────
st.markdown("---")
st.markdown(
    f"**{config.APP_TITLE}** | Chat-First AI 導流架構 | "
    f"基於 LangChain + {config.LLM_MODEL} | "
    "支援多種數據格式、統計分析與機器學習"
)
