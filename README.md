# 數據分析小幫手 📊

基於 LangChain + Qwen2.5 的本地數據分析工具，具備 GUI 介面，支援多種檔案格式，提供完整的資料科學分析與機器學習功能。

## 🎯 主要特色

- **多格式支援**: CSV, TXT, Excel (.xlsx/.xls), SAS (.sas7bdat)
- **數據預處理**: 缺失值處理、數據類型轉換、數據清理
- **統計分析**: 敘述統計、t 檢定、線性迴歸、相關分析
- **AI 助手**: 基於 LangChain + Qwen2.5 的智能問答
- **GUI 介面**: 直觀的 Streamlit 網頁介面
- **本地運行**: 完全在本地運行，保護數據隱私

## 💻 系統需求

### 硬體需求
- **RAM**: 最少 16GB，建議 20GB 以上
- **儲存空間**: 至少 10GB 可用空間
- **處理器**: 支援 x64 架構的現代處理器

### 軟體需求
- **作業系統**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 或更高版本
- **Ollama**: 用於運行 Llama-3-8B 模型



## 🚀 安裝步驟

### 步驟 1: 安裝 Python 環境

確保您的系統已安裝 Python 3.8 或更高版本：

```bash
python --version
# 或
python3 --version
```

如果沒有安裝 Python，請從 [python.org](https://www.python.org/downloads/) 下載並安裝。

### 步驟 2: 安裝 Ollama

Ollama 是運行 Llama-3-8B 模型的必要工具。

**Windows:**
1. 前往 [Ollama 官網](https://ollama.ai/) 下載 Windows 版本
2. 執行安裝程式並按照指示完成安裝

**macOS:**
```bash
# 使用 Homebrew 安裝
brew install ollama

# 或直接下載安裝包
# 前往 https://ollama.ai/ 下載 macOS 版本
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 步驟 3: 下載 Llama-3-8B 模型

安裝 Ollama 後，下載 Llama-3-8B 模型：

```bash
ollama pull qwen2.5:7b
```

這個步驟可能需要一些時間，因為模型檔案約 4.7GB。

### 步驟 4: 啟動 Ollama 服務

```bash
ollama serve
```

保持這個終端視窗開啟，Ollama 服務需要在背景運行。

### 步驟 5: 下載數據分析小幫手

將所有必要的檔案下載到一個資料夾中：

- `streamlit_app.py` - 主要的 GUI 應用程式
- `data_loader.py` - 數據載入模組
- `data_preprocessing.py` - 數據預處理模組
- `data_analysis.py` - 統計分析模組
- `langchain_agent.py` - LangChain Agent
- `rag_manager.py` - RAG 管理模組
- `run_data_assistant.py` - 啟動腳本
- `requirements.txt` - Python 依賴清單

### 步驟 6: 建立 Python 虛擬環境

```bash
# 進入專案資料夾
cd /path/to/data-assistant

# 建立虛擬環境
python -m venv data_assistant_env

# 啟動虛擬環境
# Windows:
data_assistant_env\Scripts\activate
# macOS/Linux:
source data_assistant_env/bin/activate
```

### 步驟 7: 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

如果沒有 `requirements.txt` 檔案，請手動安裝：

```bash
pip install streamlit langchain langchain-community langchain-ollama pandas numpy scikit-learn scipy statsmodels openpyxl sas7bdat chromadb sentence-transformers
```


## 🎮 使用指南

### 啟動應用程式

有兩種方式啟動數據分析小幫手：

**方式 1: 使用啟動腳本 (推薦)**
```bash
python run_data_assistant.py
```

**方式 2: 直接啟動 Streamlit**
```bash
streamlit run streamlit_app.py
```

應用程式將在瀏覽器中自動開啟，網址為 `http://localhost:8501`。

### 基本使用流程

#### 1. 上傳數據文件

- 點擊「選擇數據文件」或直接拖拽文件到上傳區域
- 支援的格式：CSV, TXT, Excel (.xlsx/.xls), SAS (.sas7bdat)
- 檔案大小限制：200MB

#### 2. 數據預覽

上傳成功後，您可以：
- 查看數據框的前幾行
- 檢視基本統計信息
- 了解各欄位的數據類型
- 查看缺失值情況

#### 3. 數據預處理

在「數據預處理」標籤頁中：
- **處理缺失值**: 選擇填充策略（平均值、中位數、眾數、刪除）
- **轉換數據類型**: 將文字轉為數值、日期等
- **數據清理**: 移除重複值、異常值處理

#### 4. 統計分析

在「統計分析」標籤頁中進行：

**敘述統計**
- 自動計算平均值、標準差、四分位數等
- 支援數值型和類別型變數
- 生成統計摘要表格

**假設檢定**
- t 檢定：比較兩組數據的平均值差異
- 相關分析：探索變數間的關聯性
- 正態性檢定：檢驗數據分佈

**迴歸建模**
- 線性迴歸：探索變數間的線性關係
- 多元迴歸：同時考慮多個預測變數
- 模型診斷：R²、p 值、殘差分析

#### 5. AI 助手互動

在「AI 助手」標籤頁中：
- 用自然語言提問關於您的數據
- 例如：「這個數據集中年齡和收入有什麼關係？」
- AI 會自動選擇合適的分析方法並提供解釋
- 支援中文和英文提問

### 範例使用情境

#### 情境 1: 員工薪資分析
```
1. 上傳員工數據 CSV 檔案
2. 查看薪資分佈的敘述統計
3. 進行男女薪資差異的 t 檢定
4. 建立年齡、教育程度對薪資的迴歸模型
5. 向 AI 助手提問：「哪些因素最影響員工薪資？」
```

#### 情境 2: 銷售數據分析
```
1. 上傳銷售數據 Excel 檔案
2. 處理缺失的銷售金額數據
3. 分析不同產品類別的銷售表現
4. 探索季節性對銷售的影響
5. 向 AI 助手提問：「如何提高下季度的銷售？」
```


## 📋 功能詳細說明

### 支援的檔案格式

| 格式 | 副檔名 | 說明 | 範例 |
|------|--------|------|------|
| CSV | .csv | 逗號分隔值檔案 | data.csv |
| 文字檔 | .txt | 分隔符號文字檔案 | data.txt |
| Excel | .xlsx, .xls | Microsoft Excel 檔案 | data.xlsx |
| SAS | .sas7bdat | SAS 數據檔案 | data.sas7bdat |

### 數據預處理功能

#### 缺失值處理
- **平均值填充**: 用該欄位的平均值填充缺失值
- **中位數填充**: 用該欄位的中位數填充缺失值
- **眾數填充**: 用該欄位的眾數填充缺失值
- **刪除**: 刪除包含缺失值的行
- **自定義值**: 用指定的值填充缺失值

#### 數據類型轉換
- **數值轉換**: 將文字轉換為整數或浮點數
- **日期轉換**: 將文字轉換為日期格式
- **類別轉換**: 將數值轉換為類別變數
- **布林轉換**: 將文字轉換為 True/False

### 統計分析功能

#### 敘述統計
- **數值變數**: 平均值、標準差、最小值、最大值、四分位數
- **類別變數**: 頻次、眾數、唯一值數量
- **分佈視覺化**: 直方圖、盒鬚圖、密度圖

#### 假設檢定
- **單樣本 t 檢定**: 檢驗樣本平均值是否等於特定值
- **獨立樣本 t 檢定**: 比較兩組獨立樣本的平均值
- **配對樣本 t 檢定**: 比較配對樣本的平均值差異
- **卡方檢定**: 檢驗類別變數間的獨立性
- **正態性檢定**: Shapiro-Wilk 檢定、Kolmogorov-Smirnov 檢定

#### 相關分析
- **Pearson 相關**: 線性相關係數
- **Spearman 相關**: 等級相關係數
- **Kendall 相關**: Tau 相關係數
- **相關矩陣**: 多變數相關分析
- **相關熱圖**: 視覺化相關關係

#### 迴歸分析
- **簡單線性迴歸**: 單一預測變數
- **多元線性迴歸**: 多個預測變數
- **邏輯迴歸**: 二元分類問題
- **多項式迴歸**: 非線性關係建模
- **模型診斷**: 殘差分析、影響點檢測

### AI 助手功能

#### 自然語言查詢
支援的查詢類型：
- 數據概覽：「這個數據集有什麼特徵？」
- 統計分析：「計算年齡的平均值和標準差」
- 關係探索：「收入和教育程度有關係嗎？」
- 預測建模：「建立一個預測薪資的模型」
- 數據清理：「處理這個數據集的缺失值」

#### 智能建議
- 根據數據特性推薦合適的分析方法
- 提供統計結果的解釋和建議
- 識別數據品質問題並提供解決方案
- 建議進一步的分析方向

### 視覺化功能

#### 基本圖表
- **直方圖**: 數值分佈視覺化
- **散點圖**: 變數關係探索
- **盒鬚圖**: 分佈和異常值檢測
- **長條圖**: 類別變數頻次
- **圓餅圖**: 比例關係展示

#### 進階圖表
- **相關熱圖**: 變數相關性矩陣
- **配對圖**: 多變數關係探索
- **殘差圖**: 迴歸模型診斷
- **QQ 圖**: 正態性檢驗
- **時間序列圖**: 時間趨勢分析


## 🔧 故障排除

### 常見問題及解決方案

#### 問題 1: Ollama 連接失敗
**錯誤訊息**: `ConnectionError: Failed to connect to Ollama service`

**解決方案**:
1. 確認 Ollama 服務正在運行：
   ```bash
   ollama serve
   ```
2. 檢查 Llama-3-8B 模型是否已下載：
   ```bash
   ollama list
   ```
3. 如果模型未下載，執行：
   ```bash
   ollama pull qwen2.5:7b
   ```

#### 問題 2: 記憶體不足
**錯誤訊息**: `MemoryError` 或系統變慢

**解決方案**:
1. 關閉其他不必要的應用程式
2. 處理較小的數據集（< 100MB）
3. 使用數據抽樣：
   ```python
   df_sample = df.sample(n=10000)  # 抽取 10,000 行
   ```

#### 問題 3: 檔案上傳失敗
**錯誤訊息**: `File upload failed` 或 `Unsupported file format`

**解決方案**:
1. 檢查檔案格式是否支援（CSV, TXT, Excel, SAS）
2. 確認檔案大小 < 200MB
3. 檢查檔案是否損壞
4. 嘗試用其他格式儲存檔案

#### 問題 4: Python 套件安裝失敗
**錯誤訊息**: `pip install failed` 或 `ModuleNotFoundError`

**解決方案**:
1. 更新 pip：
   ```bash
   pip install --upgrade pip
   ```
2. 使用國內鏡像源：
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ package_name
   ```
3. 檢查 Python 版本是否 >= 3.8

#### 問題 5: SAS 檔案讀取失敗
**錯誤訊息**: `SAS file reading error`

**解決方案**:
1. 確認 SAS 檔案版本（支援 SAS7BDAT 格式）
2. 嘗試用 SAS 軟體重新匯出檔案
3. 轉換為 CSV 格式後再上傳

### 效能優化建議

#### 記憶體優化
- 處理大型數據集時，考慮分批處理
- 及時刪除不需要的變數：`del df_old`
- 使用適當的數據類型（如 `category` 代替 `object`）

#### 運算速度優化
- 關閉不必要的瀏覽器標籤頁
- 使用 SSD 硬碟儲存數據
- 確保有足夠的可用 RAM（建議 > 8GB）

## ❓ 常見問題 (FAQ)

### Q1: 這個工具是否免費？
A: 是的，這是一個開源工具，完全免費使用。

### Q2: 數據會被上傳到雲端嗎？
A: 不會。所有數據處理都在您的本地電腦上進行，確保數據隱私和安全。

### Q3: 支援哪些作業系統？
A: 支援 Windows 10/11、macOS 10.15+ 和 Linux (Ubuntu 18.04+)。

### Q4: 可以處理多大的數據集？
A: 建議數據集大小 < 200MB，具體取決於您的電腦記憶體。對於更大的數據集，建議先進行抽樣。

### Q5: AI 助手支援哪些語言？
A: 主要支援繁體中文和英文，也可以理解簡體中文。

### Q6: 如何更新到最新版本？
A: 重新下載最新的程式檔案，並重新安裝 Python 依賴即可。

### Q7: 可以自定義分析功能嗎？
A: 可以。您可以修改 `data_analysis.py` 檔案來添加自定義的統計分析功能。

### Q8: 如何匯出分析結果？
A: 目前支援複製文字結果。未來版本將支援匯出 PDF 和 Excel 報告。

### Q9: 遇到 bug 如何回報？
A: 請記錄錯誤訊息和重現步驟，並聯繫開發者。

### Q10: 是否有使用教學影片？
A: 目前提供文字說明。建議先閱讀本說明文件並嘗試使用示例數據。

## 📞 技術支援

如果您遇到無法解決的問題，請提供以下資訊：
- 作業系統版本
- Python 版本
- 錯誤訊息截圖
- 使用的數據檔案格式和大小
- 重現問題的步驟

## 🎉 開始使用

現在您已經了解了數據分析小幫手的完整功能，可以開始您的數據分析之旅了！

1. 確保 Ollama 服務正在運行
2. 執行 `python run_data_assistant.py`
3. 在瀏覽器中開啟 `http://localhost:8501`
4. 上傳您的第一個數據檔案
5. 開始探索數據的奧秘！

祝您使用愉快！ 🚀

