"""Tab: 統計分析 & 特徵工程 (Step 4 in ML Workflow)"""

import streamlit as st
import pandas as pd
from data_analysis import (
    descriptive_statistics, perform_ttest, perform_linear_regression,
    perform_chi_square_test, perform_anova, perform_correlation_analysis
)
from visualization import plot_correlation_heatmap, plot_woe_chart, plot_iv_ranking
from woe_iv import calculate_woe_iv, calculate_iv_table


# ─────────────────────────────────────────────────────────
# 輔助：呈現統計結果（自動判斷字串 vs DataFrame）
# ─────────────────────────────────────────────────────────

def _show_result(result, label: str):
    """統一輸出分析結果，DataFrame 用表格，字串用程式碼區塊。"""
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True, hide_index=True)
    else:
        st.code(str(result), language="")


def _save_result(label: str, result):
    """把結果存入 session_state 分析歷史，固定為 (label, result) tuple。"""
    st.session_state.analysis_results.append((label, result))


# ─────────────────────────────────────────────────────────
# 渲染主函式
# ─────────────────────────────────────────────────────────

def render(df: pd.DataFrame):
    # ── Step 標題 ──
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#519D9E 0%,#58C9B9 100%);
                    border-radius:10px;padding:14px 20px;margin-bottom:16px;">
            <span style="color:#fff;font-size:1.1rem;font-weight:700;">
                📐 Step 4：統計分析 & 特徵工程
            </span><br>
            <span style="color:#dff6f4;font-size:0.87rem;">
                透過統計方法了解變數分佈、變數間關係，以及各特徵對目標的預測力（WOE/IV），
                為機器學習模型的特徵篩選提供科學依據。
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    analysis_type = st.selectbox(
        "📋 選擇分析類型",
        ["敘述統計", "t 檢定", "線性迴歸", "卡方檢定", "ANOVA 變異數分析",
         "相關分析", "WOE/IV 分析"],
        help="由左至右依序對應 ML 工作流中的探索 → 特徵篩選步驟。"
    )

    st.markdown("---")

    # ╔════════════════════════════════════════╗
    # ║  1. 敘述統計                           ║
    # ╚════════════════════════════════════════╝
    if analysis_type == "敘述統計":
        st.markdown("""
        > **📖 學習重點**：敘述統計是 EDA (探索性資料分析) 的第一步。
        > 透過平均數、中位數、標準差、最大值/最小值等，迅速掌握每個變數的「樣貌」。
        """)

        col_filter = st.multiselect(
            "選擇要分析的欄位（留空表示全部）",
            df.columns.tolist(), default=[]
        )
        selected_df = df[col_filter] if col_filter else df

        if st.button("📊 計算敘述統計", type="primary"):
            try:
                result = descriptive_statistics(selected_df)

                # ── 數值欄位敘述統計 ──
                num_desc = selected_df.describe(include="number")
                if not num_desc.empty:
                    st.markdown("#### 📈 數值型欄位摘要")
                    # 轉置後每列為一個欄位
                    num_df = num_desc.T.reset_index().rename(columns={"index": "欄位"})
                    # 將數值欄位格式化
                    for c in num_df.columns:
                        if c != "欄位":
                            num_df[c] = pd.to_numeric(num_df[c], errors="ignore")
                    st.dataframe(
                        num_df.style.format(
                            {col: "{:.4f}" for col in num_df.select_dtypes("number").columns},
                            na_rep="—"
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                # ── 類別欄位敘述統計 ──
                cat_desc = selected_df.describe(include=["object", "category"])
                if not cat_desc.empty:
                    st.markdown("#### 🔤 類別型欄位摘要")
                    cat_df = cat_desc.T.reset_index().rename(columns={"index": "欄位"})
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)

                # ── 缺失值一覽 ──
                missing = selected_df.isnull().sum()
                if missing.sum() > 0:
                    st.markdown("#### ⚠️ 缺失值一覽")
                    miss_df = pd.DataFrame({
                        "欄位": missing.index,
                        "缺失數": missing.values,
                        "缺失率": (missing.values / len(selected_df) * 100).round(2),
                    }).query("缺失數 > 0").reset_index(drop=True)
                    st.dataframe(miss_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ 選取欄位中無缺失值")

                _save_result("敘述統計", result)
                st.caption("💡 下一步建議：確認數值分佈後，往 📊 Step 3 視覺化，或直接到 🤖 Step 5 建模。")

            except Exception as e:
                st.error(f"計算失敗: {str(e)}")

    # ╔════════════════════════════════════════╗
    # ║  2. t 檢定                             ║
    # ╚════════════════════════════════════════╝
    elif analysis_type == "t 檢定":
        st.markdown("""
        > **📖 學習重點**：t 檢定用來比較兩組數值的**平均值是否有顯著差異**。
        > 常用於：驗證「違約 vs. 正常客戶」的年齡是否真的不同。
        """)
        if not numeric_cols:
            st.warning("數據中沒有數值型欄位。")
        else:
            col1, col2 = st.columns(2)
            with col1:
                column1 = st.selectbox("選擇第一個變數", numeric_cols)
            with col2:
                column2 = st.selectbox("選擇第二個變數 (可選)", ["（單樣本）"] + numeric_cols)
                column2 = None if column2 == "（單樣本）" else column2
            alpha = st.slider("顯著水準 α", 0.01, 0.10, 0.05, 0.01)
            if st.button("執行 t 檢定", type="primary"):
                try:
                    result = perform_ttest(df, column1, column2, alpha)
                    st.code(str(result), language="")
                    _save_result("t 檢定", result)
                except Exception as e:
                    st.error(f"檢定失敗: {str(e)}")

    # ╔════════════════════════════════════════╗
    # ║  3. 線性迴歸                           ║
    # ╚════════════════════════════════════════╝
    elif analysis_type == "線性迴歸":
        st.markdown("""
        > **📖 學習重點**：線性迴歸探討**連續型目標**（Y）與特徵（X）的線性關係。
        > 在 ML 前作為「基準線模型」非常有用。
        """)
        if not numeric_cols:
            st.warning("數據中沒有數值型欄位。")
        else:
            target_col = st.selectbox("選擇目標變數 (Y)", numeric_cols)
            feature_cols = st.multiselect("選擇特徵變數 (X)", [c for c in numeric_cols if c != target_col])
            if st.button("執行線性迴歸", type="primary"):
                if not feature_cols:
                    st.error("請至少選擇一個特徵變數")
                else:
                    try:
                        result = perform_linear_regression(df, target_col, feature_cols)
                        st.code(str(result), language="")
                        _save_result("線性迴歸", result)
                    except Exception as e:
                        st.error(f"迴歸分析失敗: {str(e)}")

    # ╔════════════════════════════════════════╗
    # ║  4. 卡方檢定                           ║
    # ╚════════════════════════════════════════╝
    elif analysis_type == "卡方檢定":
        st.markdown("""
        > **📖 學習重點**：卡方檢定用來測試**兩個類別變數是否獨立**（無關聯）。
        > 例如：城市與違約狀態是否相關？
        """)
        if len(categorical_cols) < 2:
            st.warning("卡方檢定需要至少 2 個類別型欄位。請先在 Step 2 變數分析中轉換資料型態。")
        else:
            c1, c2 = st.columns(2)
            with c1:
                chi_col1 = st.selectbox("選擇變數 1", categorical_cols)
            with c2:
                chi_col2 = st.selectbox("選擇變數 2", [c for c in categorical_cols if c != chi_col1])
            alpha = st.slider("顯著水準 α", 0.01, 0.10, 0.05, 0.01, key="chi_alpha")
            if st.button("執行卡方檢定", type="primary"):
                try:
                    result = perform_chi_square_test(df, chi_col1, chi_col2, alpha)
                    st.code(str(result), language="")
                    _save_result("卡方檢定", result)
                except Exception as e:
                    st.error(f"卡方檢定失敗: {str(e)}")

    # ╔════════════════════════════════════════╗
    # ║  5. ANOVA                              ║
    # ╚════════════════════════════════════════╝
    elif analysis_type == "ANOVA 變異數分析":
        st.markdown("""
        > **📖 學習重點**：ANOVA 是 t 檢定的延伸，可**同時比較 3 組以上**的平均值差異。
        > 例如：北中南三區客戶的平均收入是否有顯著差異？
        """)
        if not categorical_cols:
            st.warning("需要至少一個類別型欄位作為分組變數。")
        elif not numeric_cols:
            st.warning("需要至少一個數值欄位。")
        else:
            c1, c2 = st.columns(2)
            with c1:
                group_col = st.selectbox("選擇分組變數 (類別)", categorical_cols)
            with c2:
                value_col = st.selectbox("選擇數值變數", numeric_cols)
            alpha = st.slider("顯著水準 α", 0.01, 0.10, 0.05, 0.01, key="anova_alpha")
            if st.button("執行 ANOVA", type="primary"):
                try:
                    result = perform_anova(df, group_col, value_col, alpha)
                    st.code(str(result), language="")
                    _save_result("ANOVA", result)
                except Exception as e:
                    st.error(f"ANOVA 失敗: {str(e)}")

    # ╔════════════════════════════════════════╗
    # ║  6. 相關分析                           ║
    # ╚════════════════════════════════════════╝
    elif analysis_type == "相關分析":
        st.markdown("""
        > **📖 學習重點**：相關係數（-1 ~ 1）衡量兩變數的**線性關係強度與方向**。
        > 接近 ±1 表示強相關；接近 0 表示無線性關係。
        > 常用於**特徵篩選**：過高相關的特徵可能互相重疊（多元共線性）。
        """)
        if len(numeric_cols) < 2:
            st.warning("需要至少 2 個數值型欄位。")
        else:
            selected_cols = st.multiselect(
                "選擇數值欄位 (至少 2 個)",
                numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))]
            )
            method = st.selectbox("相關方法", ["pearson", "spearman", "kendall"],
                                  help="pearson=線性；spearman=排序；kendall=序數")
            if st.button("執行相關分析", type="primary"):
                if len(selected_cols) < 2:
                    st.error("請至少選擇兩個欄位")
                else:
                    try:
                        result = perform_correlation_analysis(df, selected_cols, method)
                        # 顯示相關矩陣（已格式化）
                        corr_matrix = df[selected_cols].corr(method=method).round(3)
                        st.markdown("**相關矩陣**")
                        st.dataframe(
                            corr_matrix.style.background_gradient(
                                cmap="RdYlGn", vmin=-1, vmax=1
                            ).format("{:.3f}"),
                            use_container_width=True,
                        )
                        fig = plot_correlation_heatmap(df, selected_cols, method)
                        st.plotly_chart(fig, use_container_width=True)
                        _save_result("相關分析", result)
                    except Exception as e:
                        st.error(f"相關分析失敗: {str(e)}")

    # ╔════════════════════════════════════════╗
    # ║  7. WOE / IV 分析                      ║
    # ╚════════════════════════════════════════╝
    elif analysis_type == "WOE/IV 分析":
        st.markdown("""
        > **📖 學習重點**：WOE (Weight of Evidence) 與 IV (Information Value) 是信用評分建模的核心指標。
        > - **IV < 0.02**：無預測力　**0.02–0.1**：弱　**0.1–0.3**：中　**> 0.3**：強
        > - WOE 可將類別特徵轉換為有序數值，幫助邏輯迴歸更好地建模。
        """)
        numeric_cols_woe = df.select_dtypes(include=["number"]).columns.tolist()
        all_cols = df.columns.tolist()
        target_col = st.selectbox("選擇二元目標變數 (0/1)", all_cols, key="woe_target")
        feature_options = [c for c in numeric_cols_woe if c != target_col]

        if not feature_options:
            st.warning("沒有可用的數值特徵變數。請先在 Step 2 處理資料型態。")
        else:
            r1, r2 = st.columns(2)
            with r1:
                n_bins = st.slider("分箱數", 3, 20, 10, key="woe_bins")
            with r2:
                bin_method = st.selectbox("分箱方法", ["quantile", "tree", "equal_width"],
                                          key="woe_method",
                                          help="quantile=等頻；tree=決策樹；equal_width=等距")

            if st.button("📊 計算所有變數 IV (特徵排名)", key="iv_all", type="primary"):
                try:
                    with st.spinner("計算中..."):
                        iv_table = calculate_iv_table(df, feature_options, target_col,
                                                      n_bins, bin_method)
                    st.markdown("**IV 排名表（由高至低）**")
                    st.dataframe(iv_table, use_container_width=True, hide_index=True)
                    fig_iv = plot_iv_ranking(iv_table)
                    st.plotly_chart(fig_iv, use_container_width=True)
                    st.caption("💡 建議保留 IV > 0.1 的特徵進行建模。")
                except Exception as e:
                    st.error(f"IV 計算失敗: {str(e)}")

            st.markdown("---")
            st.markdown("##### 🔍 單一變數 WOE 深入分析")
            woe_feature = st.selectbox("選擇變數查看 WOE 詳情", feature_options, key="woe_feature")
            if st.button("計算 WOE", key="woe_single"):
                try:
                    woe_table, total_iv = calculate_woe_iv(
                        df, woe_feature, target_col, n_bins, bin_method
                    )
                    st.metric("Information Value (IV)", f"{total_iv:.4f}")
                    st.dataframe(woe_table, use_container_width=True)
                    fig_woe = plot_woe_chart(woe_table, woe_feature)
                    st.plotly_chart(fig_woe, use_container_width=True)
                except Exception as e:
                    st.error(f"WOE 計算失敗: {str(e)}")

    # ─────────────────────────────────────────
    # 分析歷史
    # ─────────────────────────────────────────
    if st.session_state.get("analysis_results"):
        st.markdown("---")
        st.markdown("#### 🗂️ 分析歷史紀錄")
        for i, (name, result) in enumerate(reversed(st.session_state.analysis_results)):
            with st.expander(f"{len(st.session_state.analysis_results) - i}. {name}"):
                _show_result(result, name)
