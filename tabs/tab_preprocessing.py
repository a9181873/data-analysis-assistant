"""Tab 2: 數據預處理"""

import streamlit as st
import pandas as pd
from data_preprocessing import handle_missing_values, convert_data_type


def render(df: pd.DataFrame):
    st.subheader("數據預處理")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**處理缺失值**")
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            st.write("缺失值統計與處理策略:")
            
            # 使用字典儲存各個變數的處理策略
            column_strategies = {}
            
            for col_name, count in missing_stats.items():
                if count > 0:
                    st.markdown(f"**{col_name}** ({count} 個缺失值)")
                    
                    # 判斷資料型態以提供預設選項
                    is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
                    
                    options = ['mean', 'median', 'mode', 'drop', 'fill_value'] if is_numeric else ['mode', 'drop', 'fill_value']
                    # 提供建議的文字說明
                    suggestion_text = "建議使用 'mean' (數值型)" if is_numeric else "建議使用 'mode' (類別/字串型)"
                    
                    strategy = st.selectbox(
                        f"選擇 {col_name} 的處理策略 ({suggestion_text})",
                        options,
                        key=f"strategy_{col_name}"
                    )
                    
                    fill_value = None
                    if strategy == 'fill_value':
                        fill_value = st.text_input(f"輸入 {col_name} 的自訂填補值", key=f"fillval_{col_name}")
                        # 簡單的型別轉換嘗試
                        if is_numeric and fill_value:
                            try:
                                fill_value = float(fill_value)
                            except ValueError:
                                st.warning("數值欄位請輸入數值格式")
                                fill_value = None
                    
                    column_strategies[col_name] = {
                        'strategy': strategy,
                        'fill_value': fill_value
                    }
                    st.divider()

            if st.button("處理缺失值"):
                try:
                    processed_df = handle_missing_values(df, column_strategies=column_strategies)
                    st.session_state.df = processed_df
                    st.success("缺失值處理完成!")
                    st.rerun()
                except Exception as e:
                    st.error(f"處理失敗: {str(e)}")
        else:
            st.info("數據中沒有缺失值")

    with col2:
        st.write("**數據類型轉換**")
        col_to_convert = st.selectbox("選擇要轉換的列", df.columns)
        new_dtype = st.selectbox(
            "選擇新的數據類型",
            ['int', 'float', 'str', 'bool'],
        )

        if st.button("轉換數據類型"):
            try:
                converted_df = convert_data_type(df, col_to_convert, new_dtype)
                st.session_state.df = converted_df
                st.success(f"列 '{col_to_convert}' 已轉換為 {new_dtype} 類型!")
                st.rerun()
            except Exception as e:
                st.error(f"轉換失敗: {str(e)}")
