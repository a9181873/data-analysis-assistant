"""
LangChain Agent 模組
基於 LangGraph + Ollama 的 AI 助手，整合數據分析工具。
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

import config

from data_loader import load_data
from data_preprocessing import handle_missing_values
from data_analysis import (
    descriptive_statistics, perform_ttest, perform_linear_regression,
    perform_chi_square_test, perform_anova, perform_correlation_analysis
)
from rag_manager import get_chroma_collection, query_rag

# ═══════════════════════════════════════════════
# 顧問式 System Prompt
# ═══════════════════════════════════════════════
CONSULTANT_SYSTEM_PROMPT = """你是一位資深的數據分析顧問，擁有豐富的統計分析、機器學習和商業分析經驗。
你的任務是協助使用者進行數據分析，像一位耐心的顧問一樣提供指導。

## 你的角色定位
- 你是「數據分析導師」，不是簡單的指令執行器
- 根據使用者的**具體問題**給出有針對性的回答，而非罐頭訊息
- 用繁體中文回覆，語氣專業但親切

## 回覆格式（重要，必須遵守）
你必須以 JSON 格式回覆，不要在 JSON 外面加任何文字或 markdown code block。
JSON 結構如下：
{"reply": "顧問式的回覆文字（Markdown 格式）", "target_module": "模組key或null", "suggested_params": {}, "actions": []}

### reply 原則：
1. 針對使用者問題，給出專業分析建議，用實際欄位名舉例。
2. 解釋為什麼建議這樣做（例如為何用 XGBoost 而非 Logistic Regression）。
3. 簡潔精準，使用列點整理。
4. 如果涉及建模，主動提醒資料預處理步驟。

### target_module 可用值：
- "data_preview" — 數據預覽
- "variable_analysis" — 變數分析處理
- "visualization" — 資料圖表化
- "statistics" — 統計分析
- "ml" — 機器學習
- "psi_monitoring" — 模型監控
- null — 不切換模組（純對話）

### suggested_params：根據 target_module 填入建議參數：
- statistics: {"analysis_type": "敘述統計|t 檢定|線性迴歸|卡方檢定|ANOVA 變異數分析|相關分析|WOE/IV 分析", "target_columns": ["col1"]}
- ml: {"task_type": "classification|regression|clustering", "model_name": "XGBoost|Random Forest|...", "target_col": "col", "feature_cols": ["col1"]}
- visualization: {"chart_type": "直方圖|散點圖|箱型圖|長條圖|圓餅圖|相關熱圖|配對散點圖", "x_col": "col", "y_col": "col"}
- 其他模組或純對話：{}

### actions：提供 1-3 個建議動作按鈕，點擊後會「自動執行」系統內建的分析工具（不會生成程式碼）：
[{"label": "📊 年齡與收入的敘述統計", "module": "statistics", "params": {"analysis_type": "敘述統計", "target_columns": ["年齡", "收入"]}}, {"label": "🤖 XGBoost 預測違約", "module": "ml", "params": {"model_name": "XGBoost", "target_col": "違約", "task_type": "classification"}}]
label 要具體描述動作內容（如「📊 年齡 vs 收入 散點圖」而非「畫圖」）。
如果是純問答不需要按鈕，設為空陣列 []。

## 可用的分析模組（你可以透過 actions 呼叫）
- 📋 數據預覽 (data_preview)：查看資料概況
- 🔬 變數分析 (variable_analysis)：處理缺失值、型態轉換、離群值
- 📊 資料圖表化 (visualization)：散點圖、直方圖、箱型圖等互動圖表
- 📐 統計分析 (statistics)：敘述統計、t 檢定、ANOVA、相關分析、WOE/IV
- 🤖 機器學習 (ml)：Random Forest、XGBoost、LightGBM、CatBoost 等
- 📡 模型監控 (psi_monitoring)：PSI 穩定性追蹤

## 注意事項
- 只回覆 JSON，不要在 JSON 外面加任何文字
- 如果使用者的問題不明確，在 reply 中詢問，actions 可以給出猜測性建議
- 如果提到統計概念，在 reply 中用淺顯的方式解釋
- 不要只說「已開啟模組」，要在 reply 中給出實質分析建議"""

import pandas as pd


def make_tools(df: pd.DataFrame) -> list:
    """
    建立 LangChain 工具清單，透過閉包 (closure) 綁定當前 DataFrame。
    """

    def load_data_tool(file_path: str) -> str:
        """載入指定路徑的數據文件。輸入是文件的絕對路徑。"""
        try:
            loaded_df = load_data(file_path.strip())
            return f"成功載入文件: {file_path}。數據框的前5行:\n{loaded_df.head().to_string()}"
        except Exception as e:
            return f"載入文件失敗: {e}"

    def descriptive_statistics_tool(_: str) -> str:
        """計算數據的敘述性統計。無需輸入參數（輸入任意字串即可）。"""
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            return descriptive_statistics(df)
        except Exception as e:
            return f"計算敘述統計失敗: {e}"

    def handle_missing_values_tool(input_str: str) -> str:
        """
        處理缺失值。
        輸入格式: "strategy" 或 "strategy,col1,col2"
        strategy 可以是: mean / median / mode / drop
        """
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            parts = [p.strip() for p in input_str.split(",")]
            strategy = parts[0] if parts else 'mean'
            cols = parts[1:] if len(parts) > 1 else None
            processed_df = handle_missing_values(df, strategy=strategy, columns=cols)
            return f"缺失值處理完成 (策略: {strategy})。剩餘缺失值: {processed_df.isnull().sum().sum()}"
        except Exception as e:
            return f"處理缺失值失敗: {e}"

    def ttest_tool(input_str: str) -> str:
        """
        執行 t 檢定。
        輸入格式: "column1" 或 "column1,column2"
        """
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            parts = [p.strip() for p in input_str.split(",")]
            col1 = parts[0]
            col2 = parts[1] if len(parts) > 1 else None
            return perform_ttest(df, col1, col2)
        except Exception as e:
            return f"t 檢定失敗: {e}"

    def linear_regression_tool(input_str: str) -> str:
        """
        執行線性迴歸。
        輸入格式: "target,feature1,feature2,..."
        """
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            parts = [p.strip() for p in input_str.split(",")]
            if len(parts) < 2:
                return "輸入格式錯誤。請使用: target,feature1,feature2,..."
            target = parts[0]
            features = parts[1:]
            return perform_linear_regression(df, target, features)
        except Exception as e:
            return f"線性迴歸失敗: {e}"

    def chi_square_tool(input_str: str) -> str:
        """
        執行卡方檢定。
        輸入格式: "col1,col2"（兩個類別型變數名）
        """
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            parts = [p.strip() for p in input_str.split(",")]
            if len(parts) < 2:
                return "需要兩個類別型變數，格式: col1,col2"
            return perform_chi_square_test(df, parts[0], parts[1])
        except Exception as e:
            return f"卡方檢定失敗: {e}"

    def anova_tool(input_str: str) -> str:
        """
        執行 ANOVA 變異數分析。
        輸入格式: "group_col,value_col"（分組變數,數值變數）
        """
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            parts = [p.strip() for p in input_str.split(",")]
            if len(parts) < 2:
                return "需要分組變數和數值變數，格式: group_col,value_col"
            return perform_anova(df, parts[0], parts[1])
        except Exception as e:
            return f"ANOVA 失敗: {e}"

    def correlation_tool(input_str: str) -> str:
        """
        執行相關分析。
        輸入格式: "col1,col2,col3" 或 "col1,col2,method"（method: pearson/spearman/kendall）
        """
        if df is None:
            return "錯誤：尚未載入數據。"
        try:
            parts = [p.strip() for p in input_str.split(",")]
            method = 'pearson'
            if parts and parts[-1] in ('pearson', 'spearman', 'kendall'):
                method = parts.pop()
            if len(parts) < 2:
                return "至少需要兩個欄位，格式: col1,col2 或 col1,col2,method"
            return perform_correlation_analysis(df, parts, method)
        except Exception as e:
            return f"相關分析失敗: {e}"

    def knowledge_base_tool(query: str) -> str:
        """
        查詢知識庫。輸入自然語言問題，返回相關的文件段落。
        適合查詢數據分析方法、統計概念、模型說明等背景知識。
        """
        try:
            _, collection = get_chroma_collection()
            if collection.count() == 0:
                return "知識庫目前為空。請先在「知識庫管理」頁面上傳文件。"
            return query_rag(query.strip(), collection, n_results=3)
        except Exception as e:
            return f"知識庫查詢失敗: {e}"

    return [
        Tool(name="LoadData", func=load_data_tool,
             description="載入指定路徑的數據文件。輸入是文件的絕對路徑字串。"),
        Tool(name="DescriptiveStatistics", func=descriptive_statistics_tool,
             description="計算當前數據的敘述性統計。輸入任意字串（例如 'run'）即可執行。"),
        Tool(name="HandleMissingValues", func=handle_missing_values_tool,
             description="處理缺失值。輸入格式: 'strategy' 或 'strategy,col1,col2'。strategy: mean/median/mode/drop。"),
        Tool(name="TTest", func=ttest_tool,
             description="執行 t 檢定。輸入格式: 'column1' 或 'column1,column2'。"),
        Tool(name="LinearRegression", func=linear_regression_tool,
             description="執行線性迴歸。輸入格式: 'target,feature1,feature2,...'"),
        Tool(name="ChiSquareTest", func=chi_square_tool,
             description="執行卡方檢定。輸入格式: 'col1,col2'（兩個類別型變數）。"),
        Tool(name="ANOVA", func=anova_tool,
             description="執行 ANOVA 變異數分析。輸入格式: 'group_col,value_col'。"),
        Tool(name="CorrelationAnalysis", func=correlation_tool,
             description="執行相關分析。輸入格式: 'col1,col2,col3' 或 'col1,col2,method'（method: pearson/spearman/kendall）。"),
        Tool(name="KnowledgeBase", func=knowledge_base_tool,
             description="查詢知識庫中的相關文件。輸入自然語言問題即可。適合查詢數據分析方法、統計概念等背景知識。"),
    ]


from langchain_core.runnables import RunnableLambda

def create_agent_executor(df: pd.DataFrame = None):
    """
    創建並返回 LangGraph ReAct Agent (或 Fallback Runnable)。
    若模型不支援 Tool Calling (如 deepseek-r1)，會退化為純問答模式。
    注入 CONSULTANT_SYSTEM_PROMPT 使 AI 扮演數據分析顧問角色。
    """
    tools = make_tools(df)

    if getattr(config, "USE_CLOUD_LLM", False):
        from langchain_openai import ChatOpenAI
        _extra_headers = {}
        if "openrouter.ai" in config.CLOUD_BASE_URL:
            _extra_headers = {
                "HTTP-Referer": "https://github.com/a9181873/data-analysis-assistant",
                "X-Title": "Data Analysis Assistant",
            }
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.CLOUD_API_KEY,
            base_url=config.CLOUD_BASE_URL,
            timeout=config.OLLAMA_TIMEOUT,
            default_headers=_extra_headers or None,
        )
        return create_react_agent(llm, tools, prompt=CONSULTANT_SYSTEM_PROMPT)
    else:
        llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            timeout=config.OLLAMA_TIMEOUT,
        )

        # 若為地端 DeepSeek 模型，由於會有 <think> 標籤，LangGraph 的 ReAct Parser 很容易解析失敗報錯
        # 因此直接強制退化為純問答模式
        is_deepseek = "deepseek" in config.LLM_MODEL.lower()

        if is_deepseek:
            print(f"⚠️ 偵測到地端 DeepSeek 模型 ({config.LLM_MODEL})，將退化為純問答模式以避免工具解析錯誤。")
            def fallback_chain(state: dict) -> dict:
                messages = state.get("messages", [])
                full_messages = [SystemMessage(content=CONSULTANT_SYSTEM_PROMPT)] + messages
                response = llm.invoke(full_messages)
                return {"messages": messages + [response]}
            return RunnableLambda(fallback_chain)

        try:
            # 測試模型是否支援 bind_tools
            llm.bind_tools(tools)
            return create_react_agent(llm, tools, prompt=CONSULTANT_SYSTEM_PROMPT)
        except Exception as e:
            # 模型不支援 Tools
            print(f"⚠️ 模型 {config.LLM_MODEL} 不支援 Tool Calling，將退化為純問答模式。({e})")
            
            def fallback_chain(state: dict) -> dict:
                messages = state.get("messages", [])
                full_messages = [SystemMessage(content=CONSULTANT_SYSTEM_PROMPT)] + messages
                response = llm.invoke(full_messages)
                return {"messages": messages + [response]}
            
            return RunnableLambda(fallback_chain)


if __name__ == '__main__':
    test_data = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [2, 4, 5, 8, 10, 12, 14, 16, 18, 20],
        'C': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(test_data)

    tools = make_tools(df)
    print("工具清單:", [t.name for t in tools])

    print("\n--- 測試敘述統計 ---")
    desc_tool = next(t for t in tools if t.name == "DescriptiveStatistics")
    print(desc_tool.func("run"))

    print("\n--- 測試相關分析 ---")
    corr_tool = next(t for t in tools if t.name == "CorrelationAnalysis")
    print(corr_tool.func("A,B,pearson"))
