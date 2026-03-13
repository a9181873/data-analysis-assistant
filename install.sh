#!/bin/bash
echo "==================================="
echo " 數據分析小幫手 v3.0 - 安裝程式"
echo "==================================="
echo

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 請先安裝 Python 3.9 或更高版本"
    exit 1
fi

echo "[1/3] 建立 Python 虛擬環境..."
python3 -m venv data_assistant_env
source data_assistant_env/bin/activate

echo "[2/3] 升級 pip..."
pip install --upgrade pip

echo "[3/3] 安裝 Python 依賴 (可能需要幾分鐘)..."
pip install -r requirements.txt

echo
echo "==================================="
echo " 安裝完成!"
echo "==================================="
echo
echo "下一步:"
echo "  1. 安裝 Ollama: https://ollama.ai/"
echo "  2. Pull model: ollama pull qwen2.5:14b"
echo "  3. 啟動: ./start.sh"
