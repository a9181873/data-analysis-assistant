#!/usr/bin/env python3
"""
數據分析小幫手 - 主要啟動腳本
基於 LangChain + Llama-3-8B 的本地數據分析工具

使用方法:
    python run_data_assistant.py

功能:
- 支援 CSV, TXT, Excel, SAS 檔案格式
- 數據預處理和清理
- 敘述統計、統計檢定、建模分析
- 基於 LangChain 的 AI 助手
- Streamlit GUI 介面

作者: Manus AI Assistant
日期: 2025-09-06
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_ollama_service():
    """檢查 Ollama 服務是否運行"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_dependencies():
    """檢查必要的依賴是否已安裝"""
    package_map = {
        'streamlit': 'streamlit',
        'langchain': 'langchain',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'scipy': 'scipy',
        'statsmodels': 'statsmodels',
        'openpyxl': 'openpyxl'
    }
    
    missing_packages = []
    for install_name, import_name in package_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(install_name)
    
    return missing_packages

def main():
    """主要啟動函數"""
    print("🚀 數據分析小幫手啟動中...")
    print("=" * 50)
    
    # 檢查當前工作目錄
    current_dir = Path.cwd()
    print(f"📁 當前工作目錄: {current_dir}")
    
    # 檢查必要文件是否存在
    required_files = [
        'streamlit_app.py', 'data_loader.py', 
        'data_preprocessing.py', 'data_analysis.py',
        'langchain_agent.py', 'rag_manager.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n請確保所有必要文件都在當前目錄中。")
        return
    
    # 檢查依賴
    print("🔍 檢查 Python 依賴...")
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ 缺少必要的 Python 套件:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n請先安裝缺少的套件:")
        print(f"pip install {' '.join(missing_deps)}")
        return
    
    print("✅ Python 依賴檢查完成")
    
    # 檢查 Ollama 服務
    print("🔍 檢查 Ollama 服務...")
    if check_ollama_service():
        print("✅ Ollama 服務運行正常")
    else:
        print("⚠️  Ollama 服務未運行")
        print("   AI 助手功能將無法使用")
        print("   請啟動 Ollama 服務以使用完整功能")
        print("   啟動命令: ollama serve")
    
    # 啟動 Streamlit 應用
    print("\n🌐 啟動 Streamlit 應用...")
    print("=" * 50)
    
    try:
        # 使用 subprocess 啟動 Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"]
        
        print("📱 應用程式將在瀏覽器中開啟...")
        print("🔗 本地網址: http://localhost:8501")
        print("\n💡 使用提示:")
        print("   1. 上傳您的數據文件 (CSV, TXT, Excel, SAS)")
        print("   2. 使用不同標籤頁進行數據預覽和分析")
        print("   3. 在 AI 助手頁面與 LangChain Agent 互動")
        print("   4. 按 Ctrl+C 停止應用程式")
        print("=" * 50)
        
        # 等待一下再開啟瀏覽器
        time.sleep(2)
        
        # 嘗試自動開啟瀏覽器
        try:
            webbrowser.open("http://localhost:8501")
        except:
            pass
        
        # 執行 Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 數據分析小幫手已停止")
    except Exception as e:
        print(f"\n❌ 啟動失敗: {e}")
        print("請檢查錯誤信息並重試")

if __name__ == "__main__":
    main()

