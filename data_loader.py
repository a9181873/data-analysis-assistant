
import pandas as pd
import os
import chardet


def _detect_encoding(file_path, sample_size=200000):
    """讀取檔案前 N bytes，用 chardet 偵測編碼。"""
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    result = chardet.detect(raw)
    encoding = result.get('encoding') or 'utf-8'
    confidence = result.get('confidence', 0)

    enc_lower = encoding.lower().replace('-', '').replace('_', '')

    # CJK 編碼：只要 chardet 偵測到就信任（信心度門檻低）
    cjk_map = {
        'big5': 'cp950', 'big5hkscs': 'cp950', 'big5tw': 'cp950',
        'gb2312': 'gb18030', 'gb18030': 'gb18030', 'gbk': 'gb18030',
        'euccn': 'gb18030', 'eucjp': 'euc_jp', 'euckr': 'euc_kr',
        'shiftjis': 'shift_jis', 'iso2022jp': 'iso2022_jp',
    }
    if enc_lower in cjk_map:
        return cjk_map[enc_lower]

    # 非 CJK 編碼：低信心度退回 utf-8
    if confidence < 0.5:
        return 'utf-8'

    return encoding


# 明確的 fallback 順序（不含 latin-1，避免靜默產生亂碼）
_FALLBACK_ENCODINGS = ['utf-8', 'utf-8-sig', 'cp950', 'big5', 'gb18030', 'euc_jp', 'shift_jis']


def load_data(file_path, delimiter=None, encoding=None):
    """
    根據文件擴展名載入數據。
    支援 .csv, .txt, .xlsx, .xls, .sas7bdat 格式。
    使用 chardet 自動偵測編碼，確保中文欄位名與內容正確讀取。
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ('.csv', '.txt'):
        sep = delimiter
        if file_extension == '.txt' and not delimiter:
            sep = r'\s+'

        kwargs = {}
        if sep:
            kwargs['sep'] = sep

        # 使用者指定編碼：直接使用
        if encoding:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)

        # 自動偵測編碼
        detected = _detect_encoding(file_path)

        # 第一次嘗試：chardet 偵測結果
        try:
            df = pd.read_csv(file_path, encoding=detected, **kwargs)
            return _validate_dataframe(df, file_path, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            pass

        # Fallback：逐一嘗試
        for fallback_enc in _FALLBACK_ENCODINGS:
            if fallback_enc.lower() == detected.lower():
                continue
            try:
                df = pd.read_csv(file_path, encoding=fallback_enc, **kwargs)
                return _validate_dataframe(df, file_path, **kwargs)
            except (UnicodeDecodeError, UnicodeError):
                continue

        # 最後手段：latin-1（保證不會失敗，但可能亂碼）
        return pd.read_csv(file_path, encoding='latin-1', **kwargs)

    elif file_extension in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)

    elif file_extension == '.sas7bdat':
        return _load_sas7bdat(file_path)

    else:
        raise ValueError(f"不支援的文件格式: {file_extension}")


def _decode_bytes_cell(x):
    """將 SAS 讀出的 bytes 儲存格解碼為字串（cp950 → utf-8 順序嘗試）。"""
    if isinstance(x, bytes):
        for enc in ('cp950', 'big5', 'utf-8'):
            try:
                return x.decode(enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return x.decode('cp950', errors='replace')
    return x


def _load_sas7bdat(file_path):
    """
    載入 .sas7bdat 檔案。
    編碼嘗試順序：big5（台灣 SAS 最常見）→ cp950 → gb18030 → utf-8。
    全部失敗時，改讀成 bytes 後逐欄手動解碼，
    避免 encoding=None 直接回傳 b'...' 位元組欄位導致後續分析全壞。
    """
    last_error = None
    for enc in ['big5', 'cp950', 'gb18030', 'utf-8']:
        try:
            df = pd.read_sas(file_path, format='sas7bdat', encoding=enc)
            return _validate_dataframe(df, file_path)
        except (ValueError, UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue

    # 最後手段：encoding=None 讀出 bytes，逐欄解碼，確保下游拿到 str
    try:
        df = pd.read_sas(file_path, format='sas7bdat', encoding=None)
    except Exception as e:
        raise ValueError(f"無法讀取 SAS 檔案: {e}（最後錯誤: {last_error}）")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(_decode_bytes_cell)
    return _validate_dataframe(df, file_path)


def _validate_dataframe(df, file_path, **kwargs):
    """
    驗證讀取結果是否合理。
    若欄位數異常多（可能是編碼錯誤導致分隔符被誤讀），重新嘗試。
    """
    # 讀取前幾行檢查：如果只有 1 欄且行數很多，可能分隔符有問題
    # 如果欄位數 > 200，幾乎肯定是編碼錯誤
    if len(df.columns) > 200:
        raise UnicodeError(f"Column count {len(df.columns)} suspiciously high, likely encoding error")
    return df


if __name__ == '__main__':
    pass
