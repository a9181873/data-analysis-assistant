"""Tab 3: 數據視覺化"""

import streamlit as st
import pandas as pd
from visualization import (
    plot_histogram, plot_scatter, plot_boxplot, plot_bar_chart,
    plot_pie_chart, plot_correlation_heatmap, plot_pairplot,
)


def render(df: pd.DataFrame):
    st.subheader("📊 數據視覺化 (Tableau 探索風格)")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()

    # 建立左右兩欄：左側為控制區(相當於 Tableau 的 Data/Marks 面板)，右側為主畫布(Canvas)
    ctrl_col, canvas_col = st.columns([1, 2.5])

    with ctrl_col:
        st.markdown("**⚙️ 圖表與標籤設定**")
        chart_type = st.selectbox(
            "圖表類型 (Marks)",
            ["直方圖", "散點圖", "箱型圖", "長條圖", "圓餅圖", "相關熱圖", "配對散點圖"]
        )
        
        fig = None # 儲存生成的圖表

        if chart_type == "直方圖":
            if not numeric_cols:
                st.warning("數據中沒有數值型欄位。")
            else:
                col_select = st.selectbox("數值欄位", numeric_cols)
                color_by = st.selectbox("分組著色", ["無"] + categorical_cols)
                nbins = st.slider("分組數", 10, 100, 30)
                fig = plot_histogram(df, col_select, nbins, color_by if color_by != "無" else None)

        elif chart_type == "散點圖":
            if not numeric_cols:
                st.warning("數據中沒有數值型欄位。")
            else:
                x_col = st.selectbox("X 軸 (Columns)", numeric_cols)
                y_col = st.selectbox("Y 軸 (Rows)", numeric_cols, index=min(1, len(numeric_cols) - 1))
                color_by = st.selectbox("分組著色 (Color)", ["無"] + categorical_cols)
                trend = st.selectbox("趨勢線", ["ols", "lowess", "無"])
                fig = plot_scatter(df, x_col, y_col, color_by if color_by != "無" else None, trend if trend != "無" else None)

        elif chart_type == "箱型圖":
            if not numeric_cols:
                st.warning("數據中沒有數值型欄位。")
            else:
                y_col = st.selectbox("數值欄位 (Rows)", numeric_cols)
                group_col = st.selectbox("分組欄位 (Columns)", ["無"] + categorical_cols)
                fig = plot_boxplot(df, y_col, group_col if group_col != "無" else None)

        elif chart_type == "長條圖":
            x_col = st.selectbox("類別欄位", categorical_cols if categorical_cols else all_cols)
            fig = plot_bar_chart(df, x_col)

        elif chart_type == "圓餅圖":
            col_select = st.selectbox("類別欄位", categorical_cols if categorical_cols else all_cols)
            fig = plot_pie_chart(df, col_select)

        elif chart_type == "相關熱圖":
            selected_cols = st.multiselect("選擇數值欄位", numeric_cols, default=numeric_cols[:min(6, len(numeric_cols))])
            method = st.selectbox("相關方法", ["pearson", "spearman", "kendall"])
            if len(selected_cols) >= 2:
                fig = plot_correlation_heatmap(df, selected_cols, method)
            else:
                st.info("請至少選擇 2 個數值欄位")

        elif chart_type == "配對散點圖":
            selected_cols = st.multiselect("選擇數值欄位 (2-5個)", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
            color_by = st.selectbox("分組著色", ["無"] + categorical_cols)
            if 2 <= len(selected_cols) <= 5:
                fig = plot_pairplot(df, selected_cols, color_by if color_by != "無" else None)
            else:
                st.info("請選擇 2 到 5 個欄位")

    with canvas_col:
        st.markdown("**🎨 視覺化畫布**")
        # 直接根據左側變數變化而動態重繪 (移除獨立的繪圖按鈕)
        if fig is not None:
            # 讓圖表隨容器擴展
            fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            if chart_type not in ["相關熱圖", "配對散點圖"]:  # 那些有自己的提示
                st.info("👈 請在左側設定圖表參數")
