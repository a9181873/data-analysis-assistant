"""Tab: 變數分析與處理
為每一個欄位提供統計摘要、迷你視覺化、型態轉換、值替換、遺漏值補值與欄位說明。
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import config
from data_preprocessing import handle_missing_values, convert_data_type


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def _numeric_summary(series: pd.Series) -> dict:
    """產生數值型欄位的統計摘要。"""
    return {
        "總數": int(series.count()),
        "缺失值": int(series.isnull().sum()),
        "缺失率": f"{series.isnull().mean() * 100:.1f}%",
        "唯一值數": int(series.nunique()),
        "平均值": f"{series.mean():.4f}" if series.count() > 0 else "N/A",
        "中位數": f"{series.median():.4f}" if series.count() > 0 else "N/A",
        "標準差": f"{series.std():.4f}" if series.count() > 0 else "N/A",
        "最小值": f"{series.min():.4f}" if series.count() > 0 else "N/A",
        "25%": f"{series.quantile(0.25):.4f}" if series.count() > 0 else "N/A",
        "75%": f"{series.quantile(0.75):.4f}" if series.count() > 0 else "N/A",
        "最大值": f"{series.max():.4f}" if series.count() > 0 else "N/A",
    }


def _categorical_summary(series: pd.Series) -> dict:
    """產生類別型欄位的統計摘要。"""
    mode_val = series.mode()[0] if not series.mode().empty else "N/A"
    return {
        "總數": int(series.count()),
        "缺失值": int(series.isnull().sum()),
        "缺失率": f"{series.isnull().mean() * 100:.1f}%",
        "唯一值數": int(series.nunique()),
        "眾數": str(mode_val),
        "眾數頻率": int(series.value_counts().iloc[0]) if series.count() > 0 else 0,
    }


def _detect_outliers(series: pd.Series) -> dict | None:
    """使用 IQR 與 3 倍標準差偵測離群值。"""
    if series.count() < 4:
        return None
    clean = series.dropna()
    # IQR 方法
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    iqr_lower, iqr_upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_outliers = int(((clean < iqr_lower) | (clean > iqr_upper)).sum())
    # 3 σ 方法
    mean, std = clean.mean(), clean.std()
    sigma_lower, sigma_upper = mean - 3 * std, mean + 3 * std
    sigma_outliers = int(((clean < sigma_lower) | (clean > sigma_upper)).sum())
    if iqr_outliers == 0 and sigma_outliers == 0:
        return None
    return {
        "IQR 離群值": iqr_outliers,
        "IQR 範圍": f"[{iqr_lower:.2f}, {iqr_upper:.2f}]",
        "3σ 離群值": sigma_outliers,
        "3σ 範圍": f"[{sigma_lower:.2f}, {sigma_upper:.2f}]",
    }


def _ai_generate_description(col_name: str, series: pd.Series) -> str:
    """透過 Ollama LLM (LangChain) 自動產生欄位業務說明。"""
    sample_values = series.dropna().head(5).tolist()
    dtype_str = str(series.dtype)
    unique_count = series.nunique()
    prompt = (
        f"你是一位資料分析師。以下是資料集中的一個欄位資訊：\n"
        f"- 欄位名稱：{col_name}\n"
        f"- 資料型態：{dtype_str}\n"
        f"- 唯一值數量：{unique_count}\n"
        f"- 前 5 筆樣本值：{sample_values}\n\n"
        f"請用一句簡潔的中文描述這個欄位可能代表什麼業務含義。只回答描述本身，不要加前綴。"
    )
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            timeout=30,
        )
        return llm.invoke(prompt).strip()
    except Exception as e:
        return f"(AI 生成失敗: {e})"


# ──────────────────────────────────────────────
# 主渲染函式
# ──────────────────────────────────────────────

def render(df: pd.DataFrame):
    st.subheader("變數分析與處理")

    # --- 初始化 session state ---
    if "column_descriptions" not in st.session_state:
        st.session_state.column_descriptions = {}

    # ═══ 資料字典匯出 / 匯入 ═══
    st.markdown("#### 📖 資料字典")
    dict_col1, dict_col2 = st.columns(2)
    with dict_col1:
        desc_data = st.session_state.column_descriptions
        if desc_data:
            json_str = json.dumps(desc_data, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ 匯出資料字典 (JSON)",
                data=json_str.encode("utf-8"),
                file_name="data_dictionary.json",
                mime="application/json",
            )
        else:
            st.caption("尚無欄位說明，請在下方各變數區塊中填寫。")
    with dict_col2:
        uploaded_dict = st.file_uploader(
            "⬆️ 匯入資料字典 (JSON)",
            type=["json"],
            key="dict_uploader",
        )
        if uploaded_dict is not None:
            try:
                imported = json.loads(uploaded_dict.read().decode("utf-8"))
                st.session_state.column_descriptions.update(imported)
                st.success(f"已匯入 {len(imported)} 個欄位說明！")
            except Exception as e:
                st.error(f"匯入失敗: {e}")

    st.markdown("---")

    # ═══ 全部欄位總覽表格 ═══
    overview_data = []
    for col in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        missing = int(df[col].isnull().sum())
        overview_data.append({
            "欄位": col,
            "型態": str(df[col].dtype),
            "缺失值": missing,
            "缺失率": f"{missing / len(df) * 100:.1f}%",
            "唯一值": int(df[col].nunique()),
            "類型": "數值" if is_num else "類別",
        })
    st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ═══ 逐變數展開面板 ═══
    for col_name in df.columns:
        series = df[col_name]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        dtype_label = "數值" if is_numeric else "類別"
        missing_count = int(series.isnull().sum())
        missing_badge = f" ⚠️ {missing_count} 缺失" if missing_count > 0 else ""

        with st.expander(f"📌 {col_name}  ({series.dtype} | {dtype_label}){missing_badge}"):
            # ── 摘要 + 視覺化 ──
            sum_col, chart_col = st.columns([1, 1])

            with sum_col:
                st.markdown("**📊 統計摘要**")
                if is_numeric:
                    summary = _numeric_summary(series)
                else:
                    summary = _categorical_summary(series)
                summary_df = pd.DataFrame(
                    list(summary.items()), columns=["指標", "值"]
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # 離群值偵測 (僅數值型)
                if is_numeric:
                    outlier_info = _detect_outliers(series)
                    if outlier_info:
                        st.warning("🚨 **偵測到離群值**")
                        for k, v in outlier_info.items():
                            st.caption(f"  {k}: {v}")

                # 樣本資料預覽 (3~5 筆)
                st.markdown("**🔍 樣本資料**")
                n_sample = min(5, len(series))
                sample_df = series.head(n_sample).to_frame()
                st.dataframe(sample_df, use_container_width=True, hide_index=False)

            with chart_col:
                st.markdown("**📈 分佈圖**")
                if is_numeric:
                    clean = series.dropna()
                    if len(clean) > 0:
                        try:
                            hist_values, bin_edges = np.histogram(clean, bins=min(30, max(5, len(clean) // 10)))
                            bin_labels = [f"{bin_edges[i]:.1f}" for i in range(len(hist_values))]
                            chart_df = pd.DataFrame({"區間": bin_labels, "頻率": hist_values})
                            st.bar_chart(chart_df.set_index("區間"))
                        except Exception:
                            st.caption("無法繪製直方圖")
                else:
                    vc = series.value_counts().head(15)
                    if len(vc) > 0:
                        chart_df = pd.DataFrame({"類別": vc.index.astype(str), "數量": vc.values})
                        st.bar_chart(chart_df.set_index("類別"))

            st.markdown("---")

            # ── 操作區：三欄佈局 ──
            op_col1, op_col2, op_col3 = st.columns(3)

            # ── 型態轉換 ──
            with op_col1:
                st.markdown("**🔄 型態轉換**")
                new_dtype = st.selectbox(
                    "目標型態",
                    ["int", "float", "str", "bool", "datetime"],
                    key=f"dtype_{col_name}",
                )
                if st.button("套用轉換", key=f"convert_{col_name}"):
                    try:
                        if new_dtype == "datetime":
                            st.session_state.df[col_name] = pd.to_datetime(
                                st.session_state.df[col_name], errors="coerce"
                            )
                        else:
                            converted_df = convert_data_type(
                                st.session_state.df, col_name, new_dtype
                            )
                            st.session_state.df = converted_df
                        st.success(f"已轉換為 {new_dtype}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"轉換失敗: {e}")

            # ── 值替換 (Value Replacement) ──
            with op_col2:
                st.markdown("**🔁 值替換**")
                old_val = st.text_input("原始值", key=f"old_{col_name}", placeholder="例如: 1.0")
                new_val = st.text_input("替換為", key=f"new_{col_name}", placeholder="例如: Y")
                if st.button("套用替換", key=f"replace_{col_name}"):
                    try:
                        target_series = st.session_state.df[col_name]
                        # 嘗試把 old_val 轉成與欄位相同的型態來比對
                        parsed_old = old_val
                        if pd.api.types.is_numeric_dtype(target_series):
                            try:
                                parsed_old = float(old_val)
                            except ValueError:
                                pass
                        elif pd.api.types.is_bool_dtype(target_series):
                            parsed_old = old_val.lower() in ("true", "1", "yes")

                        match_count = int((target_series == parsed_old).sum())
                        if match_count == 0:
                            st.warning(f"找不到值 '{old_val}'")
                        else:
                            st.session_state.df[col_name] = target_series.replace(
                                {parsed_old: new_val}
                            )
                            st.success(f"已替換 {match_count} 筆: {old_val} → {new_val}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"替換失敗: {e}")

            # ── 遺漏值補值 ──
            with op_col3:
                st.markdown("**🩹 缺失值處理**")
                if missing_count > 0:
                    options = (
                        ["mean", "median", "mode", "fill_value", "drop"]
                        if is_numeric
                        else ["mode", "fill_value", "drop"]
                    )
                    fill_strategy = st.selectbox(
                        "補值策略",
                        options,
                        key=f"fill_strat_{col_name}",
                    )
                    fill_value = None
                    if fill_strategy == "fill_value":
                        fill_value = st.text_input(
                            "自訂填補值",
                            key=f"fill_val_{col_name}",
                        )
                        if is_numeric and fill_value:
                            try:
                                fill_value = float(fill_value)
                            except ValueError:
                                st.warning("數值欄位請輸入數值格式")
                                fill_value = None

                    if st.button("套用補值", key=f"fill_{col_name}"):
                        try:
                            col_strategies = {
                                col_name: {
                                    "strategy": fill_strategy,
                                    "fill_value": fill_value,
                                }
                            }
                            processed_df = handle_missing_values(
                                st.session_state.df,
                                column_strategies=col_strategies,
                            )
                            st.session_state.df = processed_df
                            st.success(f"已處理 {col_name} 的缺失值！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"處理失敗: {e}")
                else:
                    st.info("此欄位無缺失值 ✅")

            st.markdown("---")

            # ── 欄位說明 ──
            st.markdown("**📝 欄位說明（供 LLM 使用）**")
            desc_col1, desc_col2 = st.columns([3, 1])
            with desc_col1:
                current_desc = st.session_state.column_descriptions.get(col_name, "")
                new_desc = st.text_input(
                    "輸入此欄位的業務說明",
                    value=current_desc,
                    key=f"desc_{col_name}",
                    placeholder="例如：此欄位代表客戶目前的婚姻狀態",
                )
                if new_desc != current_desc:
                    st.session_state.column_descriptions[col_name] = new_desc
            with desc_col2:
                if st.button("🤖 AI 生成", key=f"ai_desc_{col_name}"):
                    with st.spinner("AI 正在推測欄位說明⋯"):
                        ai_desc = _ai_generate_description(col_name, series)
                        st.session_state.column_descriptions[col_name] = ai_desc
                        st.success("AI 說明已生成！")
                        st.rerun()
