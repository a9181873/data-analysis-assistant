import pandas as pd

def handle_missing_values(df, strategy='mean', columns=None, column_strategies=None):
    """
    處理數據框中的缺失值。
    strategy: 若提供字串，則對所有數值型欄位套用同一策略 ('mean', 'median', 'mode', 'drop', 'fill_value')
    columns: 要處理的列名列表，如果為 None 則處理所有數值列。此選項僅在 column_strategies 為 None 時生效。
    column_strategies: 字典，格式為 {column_name: {'strategy': '...', 'fill_value': ...}}，可針對特定欄位指定策略。
    """
    df_processed = df.copy()
    
    # 處理逐變數設定的策略
    if column_strategies is not None:
        for col, col_config in column_strategies.items():
            if col not in df_processed.columns:
                continue
            
            col_strat = col_config.get('strategy')
            
            if col_strat == 'mean':
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif col_strat == 'median':
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            elif col_strat == 'mode':
                if not df_processed[col].mode().empty:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            elif col_strat == 'drop':
                df_processed.dropna(subset=[col], inplace=True)
            elif col_strat == 'fill_value':
                fill_val = col_config.get('fill_value')
                if fill_val is not None:
                    df_processed[col] = df_processed[col].fillna(fill_val)
                else:
                    raise ValueError(f"欄位 {col} 使用 'fill_value' 策略時，必須提供 fill_value 參數。")
            else:
                raise ValueError(f"欄位 {col} 使用不支援的缺失值處理策略: {col_strat}")
                
        return df_processed
        
    # 原本的統一處理邏輯
    if columns is None:
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
    else:
        numeric_cols = [col for col in columns if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    if strategy == 'mean':
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    elif strategy == 'median':
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            if not df_processed[col].mode().empty:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    elif strategy == 'drop':
        df_processed.dropna(subset=numeric_cols, inplace=True)
    elif strategy == 'fill_value':
        raise ValueError("使用 'fill_value' 策略時，必須提供 fill_value 參數。")
    else:
        raise ValueError(f"不支援的缺失值處理策略: {strategy}")
    return df_processed

def convert_data_type(df, column, dtype):
    """
    轉換指定列的數據類型。
    """
    df_processed = df.copy()
    try:
        df_processed[column] = df_processed[column].astype(dtype)
    except Exception as e:
        raise TypeError(f"無法將列 '{column}' 轉換為 '{dtype}' 類型: {e}")
    return df_processed

if __name__ == '__main__':
    # 測試代碼
    data = {'A': [1, 2, None, 4, 5],
            'B': [None, 20, 30, 40, 50],
            'C': ['a', 'b', None, 'd', 'e']}
    df = pd.DataFrame(data)
    print("原始數據框:\n", df)

    # 測試缺失值處理 (mean) 統一策略
    df_mean_filled = handle_missing_values(df.copy(), strategy='mean')
    print("\n處理缺失值 (mean, 統一) 後:\n", df_mean_filled)

    # 測試缺失值處理 (Drop)
    df_dropped = handle_missing_values(df.copy(), strategy='drop')
    print("\n處理缺失值 (drop, 統一) 後:\n", df_dropped)
    
    # 測試逐變數處理策略
    strategies = {
        'A': {'strategy': 'mean'},
        'B': {'strategy': 'median'},
        'C': {'strategy': 'fill_value', 'fill_value': 'Unknown'}
    }
    df_col_filled = handle_missing_values(df.copy(), column_strategies=strategies)
    print("\n逐變數處理缺失值後:\n", df_col_filled)

    # 測試數據類型轉換
    df_converted = convert_data_type(df.copy(), 'A', 'float')
    print("\n列 'A' 轉換為 float 後:\n", df_converted)

    try:
        df_invalid_convert = convert_data_type(df.copy(), 'C', 'int')
    except TypeError as e:
        print(f"\n類型轉換錯誤測試: {e}")


