"""Tab 7: 知識庫管理 (RAG)"""

import streamlit as st
from rag_manager import (
    get_chroma_collection, add_documents, delete_documents,
    get_collection_stats, query_rag_with_scores, chunk_text,
)


def render(df=None):
    st.subheader("知識庫管理")
    st.caption("上傳文件至向量知識庫，讓 AI 助手能參考這些資料回答問題。")

    _, collection = get_chroma_collection()

    # --- 集合狀態 ---
    stats = get_collection_stats(collection)
    st.metric("知識庫文件數", stats["count"])

    # --- 上傳文件 ---
    st.markdown("---")
    st.write("**上傳文件**")
    uploaded = st.file_uploader(
        "上傳文字文件 (.txt / .md / .csv)",
        type=["txt", "md", "csv"],
        accept_multiple_files=True,
        key="rag_upload",
    )

    chunk_size = st.slider("段落大小 (字元)", 200, 2000, 500, 50)

    if uploaded and st.button("加入知識庫", key="rag_add"):
        total_chunks = 0
        for f in uploaded:
            raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("big5", errors="ignore")

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)
            add_documents(collection, chunks)
            total_chunks += len(chunks)

        st.success(f"已新增 {total_chunks} 個段落（來自 {len(uploaded)} 個文件）")
        st.rerun()

    # --- 查詢測試 ---
    st.markdown("---")
    st.write("**查詢測試**")
    test_query = st.text_input("輸入測試查詢:", placeholder="例如：什麼是 XGBoost？")
    n_results = st.slider("返回筆數", 1, 10, 3, key="rag_n")

    if test_query and st.button("查詢", key="rag_query"):
        results = query_rag_with_scores(test_query, collection, n_results)
        if not results:
            st.info("知識庫為空或找不到相關內容。")
        else:
            for i, (doc, dist) in enumerate(results):
                similarity = max(0, 1 - dist)
                st.write(f"**結果 {i+1}** (相似度: {similarity:.2%})")
                st.text(doc)
                st.markdown("---")

    # --- 管理 ---
    st.markdown("---")
    st.write("**管理**")
    if stats["count"] > 0:
        if st.button("清空知識庫", type="secondary", key="rag_clear"):
            # 取得所有 ID 並刪除
            all_data = collection.get()
            all_ids = all_data.get("ids", [])
            if all_ids:
                delete_documents(collection, all_ids)
                st.success("知識庫已清空。")
                st.rerun()

        # 顯示現有文件預覽
        with st.expander("現有文件預覽"):
            if "sample_docs" in stats:
                for sid, sdoc in zip(stats["sample_ids"], stats["sample_docs"]):
                    st.write(f"- **{sid}**: {sdoc}")
            if stats["count"] > 5:
                st.caption(f"（僅顯示前 5 筆，共 {stats['count']} 筆）")
