"""
RAG Manager — 使用 ChromaDB + sentence-transformers 實作文件向量檢索。
"""

import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid
import config

_embed_model = None


def _get_embed_model():
    """延遲載入嵌入模型（避免啟動時過慢）。"""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def get_chroma_collection(collection_name: str = "data_analysis_docs"):
    """
    取得或建立 ChromaDB 持久化集合。
    返回 (client, collection)。
    """
    client = chromadb.PersistentClient(path="./storage")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def add_documents(collection, documents: list, doc_ids: list = None):
    """將文字文件加入向量集合。"""
    if not documents:
        return
    model = _get_embed_model()
    embeddings = model.encode(documents, show_progress_bar=False).tolist()
    if doc_ids is None:
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=doc_ids,
    )


def delete_documents(collection, doc_ids: list):
    """從集合中刪除指定 ID 的文件。"""
    if doc_ids:
        collection.delete(ids=doc_ids)


def get_collection_stats(collection) -> dict:
    """返回集合統計資訊。"""
    count = collection.count()
    result = {"count": count}
    if count > 0:
        peek = collection.peek(limit=min(5, count))
        result["sample_ids"] = peek.get("ids", [])
        result["sample_docs"] = [
            d[:100] + "..." if len(d) > 100 else d
            for d in peek.get("documents", [])
        ]
    return result


def query_rag(query_text: str, collection, n_results: int = 3) -> str:
    """用自然語言查詢向量集合，返回最相關的文件段落。"""
    count = collection.count()
    if count == 0:
        return "找不到相關文檔。"
    model = _get_embed_model()
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, count),
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "找不到相關文檔。"
    return "\n\n---\n\n".join(docs)


def query_rag_with_scores(query_text: str, collection, n_results: int = 3) -> list:
    """查詢並返回結果 + 距離分數。返回 list of (doc, distance)。"""
    count = collection.count()
    if count == 0:
        return []
    model = _get_embed_model()
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, count),
        include=["documents", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return list(zip(docs, distances))


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """將長文字切分為固定大小的段落，帶重疊。"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


if __name__ == '__main__':
    print("初始化 ChromaDB 集合...")
    _, collection = get_chroma_collection("rag_test")

    sample_docs = [
        "這是一份關於數據分析的範例文件。包含統計學與機器學習基礎知識。",
        "XGBoost 是一種梯度提升演算法，常用於結構化數據的分類與迴歸任務。",
        "Pandas 是 Python 的資料分析套件，提供 DataFrame 資料結構。",
    ]
    print("加入測試文件...")
    add_documents(collection, sample_docs, doc_ids=["doc0", "doc1", "doc2"])

    print("執行 RAG 查詢...")
    result = query_rag("什麼是 XGBoost？", collection)
    print(f"RAG 回應:\n{result}")

    print("\n帶分數查詢...")
    results = query_rag_with_scores("什麼是 XGBoost？", collection)
    for doc, dist in results:
        print(f"  距離={dist:.4f}: {doc[:60]}...")
