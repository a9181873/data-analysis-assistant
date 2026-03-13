#!/bin/bash
echo "==================================="
echo " 數據分析小幫手 v3.0"
echo "==================================="
echo

if [ -d "data_assistant_env" ]; then
    source data_assistant_env/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[ERROR] 找不到虛擬環境，請先執行 install.sh"
    exit 1
fi

echo "啟動中... 瀏覽器將自動開啟 http://localhost:8501"
echo "按 Ctrl+C 停止服務"
echo
streamlit run streamlit_app.py
