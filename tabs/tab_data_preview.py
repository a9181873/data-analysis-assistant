"""Tab 1: 數據預覽"""

import streamlit as st
import pandas as pd
import io
import config


def render(df: pd.DataFrame):
    st.subheader("數據預覽")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**數據框概覽:**")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.write("**基本信息:**")
        st.write(f"- 行數: {len(df)}")
        st.write(f"- 列數: {len(df.columns)}")
        st.write(f"- 記憶體使用: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

        st.write("**列名與類型:**")
        for col in df.columns:
            st.write(f"- `{col}` ({df[col].dtype})")

        # 缺失值摘要
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.write("**缺失值:**")
            for col_name, count in missing.items():
                if count > 0:
                    st.write(f"- {col_name}: {count} ({count/len(df)*100:.1f}%)")

    # 資料匯出區
    st.markdown("---")
    st.write("**資料匯出**")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_bytes = df.to_csv(index=False).encode(config.EXPORT_ENCODING)
        st.download_button(
            label="下載 CSV",
            data=csv_bytes,
            file_name="data_export.csv",
            mime="text/csv",
        )
    with dl2:
        excel_buf = io.BytesIO()
        df.to_excel(excel_buf, index=False, engine='openpyxl')
        st.download_button(
            label="下載 Excel",
            data=excel_buf.getvalue(),
            file_name="data_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dl3:
        json_str = df.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="下載 JSON",
            data=json_str.encode('utf-8'),
            file_name="data_export.json",
            mime="application/json",
        )
