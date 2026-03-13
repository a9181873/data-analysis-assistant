"""Tab 6: 模型監控 (PSI)"""

import streamlit as st
import pandas as pd
import os
from data_loader import load_data
from psi import calculate_psi, calculate_psi_report
from visualization import plot_psi_comparison, plot_distribution_shift
import config


def render(df: pd.DataFrame):
    st.subheader("模型監控 (PSI)")
    st.caption("上傳監控期資料，比較變數分佈是否偏移。基準期使用目前載入的資料。")

    st.write(f"**基準期資料:** {len(df)} 行 x {len(df.columns)} 列")

    # 上傳監控期資料
    uploaded = st.file_uploader(
        "上傳監控期資料",
        type=config.SUPPORTED_FILE_TYPES,
        key="psi_upload",
    )

    if uploaded is None:
        st.info("請上傳監控期資料以計算 PSI。")
        return

    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        df_actual = load_data(tmp_path)
    except Exception as e:
        st.error(f"載入監控期資料失敗: {str(e)}")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    st.write(f"**監控期資料:** {len(df_actual)} 行 x {len(df_actual.columns)} 列")

    # 共同數值欄位
    numeric_expected = set(df.select_dtypes(include=["number"]).columns)
    numeric_actual = set(df_actual.select_dtypes(include=["number"]).columns)
    common_cols = sorted(numeric_expected & numeric_actual)

    if not common_cols:
        st.error("兩份資料沒有共同的數值欄位。")
        return

    selected_cols = st.multiselect("選擇要比較的欄位", common_cols, default=common_cols)
    bins = st.slider("分組數", 5, 20, 10)

    if st.button("計算 PSI", key="psi_calc"):
        if not selected_cols:
            st.warning("請至少選擇一個欄位進行比較。")
            return
        with st.spinner("計算中..."):
            report = calculate_psi_report(df, df_actual, columns=selected_cols, bins=bins)

            st.write("**PSI 報告**")
            st.dataframe(report, use_container_width=True)

            st.caption("PSI < 0.1: 穩定 | 0.1~0.25: 輕微偏移 | > 0.25: 顯著偏移")

            fig_psi = plot_psi_comparison(report)
            st.plotly_chart(fig_psi, use_container_width=True)

            # 顯示偏移較大的變數分佈
            shifted = report[report["psi"].notna() & (report["psi"] >= 0.1)]
            if not shifted.empty:
                st.write("**偏移變數分佈比較:**")
                for _, row in shifted.iterrows():
                    var = row["variable"]
                    fig_dist = plot_distribution_shift(
                        df[var].dropna(), df_actual[var].dropna(), var
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
