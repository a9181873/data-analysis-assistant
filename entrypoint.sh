#!/bin/bash
# 等待 Ollama 服務啟動
echo "Waiting for Ollama service..."
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo "Ollama is ready."

# 自動拉取模型（如果尚未下載）
MODEL="${LLM_MODEL:-qwen2.5:14b}"
if ! curl -s http://ollama:11434/api/tags | grep -q "$MODEL"; then
  echo "Pulling model: $MODEL ..."
  curl -s -X POST http://ollama:11434/api/pull -d "{\"name\":\"$MODEL\"}"
  echo "Model pulled."
fi

# 啟動 Streamlit
exec streamlit run streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
