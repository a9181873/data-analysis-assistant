import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Prototype UI: Chat-First 數據分析小幫手 ---
st.set_page_config(layout="wide", page_title="AI 數據架構師 (Prototype)", page_icon="💡")

# 自訂 CSS，美化排版
st.markdown("""
<style>
.stChatMessage { border-radius: 10px; padding: 10px; }
.css-1d391kg { padding-top: 1rem; }
.workspace-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    background-color: #FAFAFA;
    height: 80vh;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# ─── 初始化 State ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是您的專屬數據架構師。\n\n請從側邊欄上傳您的資料，或是點擊按鈕使用『範例數據』開始我們今天的分析旅程。"}
    ]
if "active_tool" not in st.session_state:
    st.session_state.active_tool = None
if "df" not in st.session_state:
    st.session_state.df = None
if "step" not in st.session_state:
    st.session_state.step = "init"

# ─── 左側邊欄 ─────────────────────────────────
with st.sidebar:
    st.header("📂 資料與設定")
    uploaded_file = st.file_uploader("上傳資料集 (CSV/Excel/TXT)")
    
    if st.button("使用範例數據 (銀行違約)", use_container_width=True):
        st.session_state.df = pd.DataFrame({
            "年齡": [25, 30, 35, 40, 45, 50],
            "收入": [50000, 60000, 70000, 80000, 90000, 65000],
            "支出": [30000, 35000, 40000, 45000, 50000, 45000],
            "違約": [0, 0, 0, 1, 0, 1]
        })
        msg = "我已經成功讀取了範例數據 (包含 6 筆資料, 4 個變數：年齡、收入、支出、違約)。\n\n看起來這是一份包含客戶屬性與是否違約的資料。您可以告訴我您想先進行哪種分析？例如：\n- 幫我視覺化收入與違約的關係\n- 幫我做敘述性統計\n- 我想建立機器學習模型預測違約"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.step = "data_loaded"
        st.rerun()
        
    st.markdown("---")
    st.caption("目前預設 LLM 引擎: qwen2.5:7b (為提升效能切換)")

# ─── 頂部標題 ─────────────────────────────────
st.title("💡 Chat-First 數據分析小幫手 (Prototype)")
st.markdown("這是一個展示**「對話導流 -> 工具面板動態召喚」**的 UI 概念原型。不再被 8 個空洞的 Tab 淹沒，而是讓 LLM 首先了解您的意圖。")
st.markdown("---")

# ─── 雙欄配置 ─────────────────────────────────
# 左半邊顯示聊天室，右半邊顯示動態工作區 UI
col_chat, col_work = st.columns([1, 1.2])

with col_chat:
    st.subheader("💬 與 AI 架構師對話")
    
    # 對話容器
    chat_container = st.container(height=550)
    
    with chat_container:
        # 顯示歷史訊息
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    # 對話輸入
    if prompt := st.chat_input("請告訴我您想分析什麼？ (例如：我想預測違約)"):
        # 加入 User Input
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 簡單的關鍵字 Intent Router 模擬 LLM 思考
        if "預測" in prompt or "違約" in prompt or "機器學習" in prompt or "ML" in prompt:
            response = "好的，原來您想進行**預測任務**！\n我將為您啟動 **『機器學習』** 模組，請看右側的工作區。請您在右側選擇：\n1. 預測目標 (Target)\n2. 演算法模型\n3. 特徵 (Features)"
            st.session_state.active_tool = "ml"
        elif "圖" in prompt or "視覺化" in prompt or "分布" in prompt or "關係" in prompt:
            response = "沒問題，我已經為您開啟 **『視覺化分析』** 面板。\n請在右側選擇您想繪製的圖表類型與對應的欄位。"
            st.session_state.active_tool = "plot"
        elif "統計" in prompt or "敘述" in prompt:
            response = "好的，我為您啟動 **『統計分析』**。\n您可以在右側查看每個變數的平均值、標準差與缺失值分佈。"
            st.session_state.active_tool = "stats"
        else:
            response = "我明白了。您可以具體告訴我您想看哪個變數的分布？或是想建立什麼主題的模型嗎？這有助於我為您準備正確的工具面板。"
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ─── 右側工作區 (Workspace) ────────────────────
with col_work:
    st.subheader("🛠️ 動態工具工作區")
    
    with st.container():
        # 這裡改用 st.markdown + div 來模擬美化背景
        
        if st.session_state.active_tool == "ml":
            st.info("🎯 **AI 提示:** 這裡原本是 tab_ml 的龐大內容，現在只有在您需要預測時才會被召喚出來。")
            st.write("### 🤖 機器學習模型設定")
            
            if st.session_state.df is not None:
                cols = list(st.session_state.df.columns)
            else:
                cols = ["缺席", "尚未有資料"]
                
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("請選擇預測目標 (Target Y)", cols, index=len(cols)-1)
                model_type = st.selectbox("模型演算法", ["Random Forest", "Logistic Regression", "XGBoost", "LightGBM"])
            with col2:
                features = st.multiselect("請選擇特徵 (Features X)", cols, default=cols[:-1])
                cv_folds = st.slider("CV 交叉驗證折數", 2, 10, 5)
                
            st.markdown("---")
            if st.button("火箭發射！🚀 開始訓練模型", use_container_width=True, type="primary"):
                with st.spinner("模型訓練中，請稍候..."):
                    time.sleep(1.5)
                st.success(f"**{model_type}** 模型訓練完成！")
                st.metric("Test Accuracy (測試集準確率)", "88.5%")
                st.metric("ROC AUC", "0.91")
                st.write("*(這裡是模型的 SHAP 解釋性圖表或特徵重要度...)*")
                
        elif st.session_state.active_tool == "plot":
            st.info("🎯 **AI 提示:** 這是 tab_visualization 的精簡版，適合用來探索資料分布。")
            st.write("### 📊 視覺化圖表配置")
            
            if st.session_state.df is not None:
                cols = list(st.session_state.df.columns)
            else:
                cols = ["預設欄位"]
                
            plot_type = st.selectbox("圖表類型", ["散佈圖 (Scatter)", "長條圖 (Bar)", "折線圖 (Line)", "圓餅圖 (Pie)"])
            x_axis = st.selectbox("X 軸欄位", cols)
            y_axis = st.selectbox("Y 軸欄位", cols, index=1 if len(cols) > 1 else 0)
            
            if st.button("生成圖表", type="primary"):
                with st.spinner("繪製中..."):
                    time.sleep(0.5)
                # Mock a chart
                chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
                st.scatter_chart(chart_data)
                
        elif st.session_state.active_tool == "stats":
            st.info("🎯 **AI 提示:** 這是 tab_statistics 裡的敘述性統計。")
            st.write("### 📉 變數摘要資訊")
            if st.session_state.df is not None:
                st.dataframe(st.session_state.df.describe(), use_container_width=True)
            else:
                st.warning("請先在側邊欄上傳或載入資料。")
            
        else:
            # 預設空白狀態
            st.markdown(
                """
                <div style='text-align: center; color: #9E9E9E; margin-top: 150px;'>
                <h1 style='color: #E0E0E0;'>✨</h1>
                <h3>您的專屬 AI 數據工作區</h3>
                <p>請在左側與 AI 助手對話，討論您的分析需求。<br>需要的圖表、統計與模型工具將會在此處自動展開。</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
