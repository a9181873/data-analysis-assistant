"""Tab 8: AI 狀態儀表板 — 顯示 Ollama 服務狀態與模型資訊"""

import streamlit as st
import pandas as pd
import config


def render(df: pd.DataFrame):
    st.subheader("🖥️ AI 服務狀態")

    import requests

    # ── 連線偵測 ──
    is_connected = False
    models_info = []
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            is_connected = True
            models_info = response.json().get("models", [])
    except Exception:
        pass

    # ── 狀態卡片 ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Ollama 服務", "✅ 已連線" if is_connected else "❌ 離線")
    with c2:
        st.metric("推理模型", config.LLM_MODEL)
    with c3:
        st.metric("Embedding 模型", config.EMBED_MODEL.split("/")[-1])

    st.markdown("---")

    if is_connected:
        st.success("Ollama 服務運行中，AI 架構師（左側對話）可正常使用。")

        # ── 已安裝的模型列表 ──
        if models_info:
            st.subheader("📦 已安裝模型")
            rows = []
            for m in models_info:
                name = m.get("name", "N/A")
                size_bytes = m.get("size", 0)
                size_gb = f"{size_bytes / (1024**3):.1f} GB" if size_bytes else "N/A"
                modified = m.get("modified_at", "N/A")
                if isinstance(modified, str) and "T" in modified:
                    modified = modified.split("T")[0]
                rows.append({"模型名稱": name, "大小": size_gb, "修改日期": modified})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── 資料集摘要 ──
        st.markdown("---")
        st.subheader("📊 當前資料集")
        if df is not None:
            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1:
                st.metric("筆數", f"{len(df):,}")
            with dc2:
                st.metric("欄位數", len(df.columns))
            with dc3:
                st.metric("數值型", len(df.select_dtypes(include=["number"]).columns))
            with dc4:
                st.metric("缺失值", int(df.isnull().sum().sum()))
        else:
            st.info("尚未載入資料。請從側邊欄上傳或使用範例數據。")

        # ── 知識庫狀態 ──
        st.markdown("---")
        st.subheader("📚 知識庫 (RAG)")
        try:
            from rag_manager import get_chroma_collection
            _, collection = get_chroma_collection()
            doc_count = collection.count()
            st.metric("已索引文檔段落", doc_count)
            if doc_count == 0:
                st.caption("知識庫為空。前往「知識庫管理」上傳文件以啟用 RAG 增強。")
        except Exception as e:
            st.warning(f"無法取得知識庫狀態: {e}")

    else:
        st.error("無法連接到 Ollama 服務")
        st.markdown(f"""
        **如何啟動 Ollama:**
        1. 確保已安裝 Ollama
        2. 在終端執行: `ollama serve`
        3. 下載模型: `ollama pull {config.LLM_MODEL}`
        4. 嵌入模型 `{config.EMBED_MODEL}` 會在首次使用 RAG 時自動下載
        """)
