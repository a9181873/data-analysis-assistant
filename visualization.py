"""
視覺化模組
基於 Plotly 的互動式數據視覺化 + ML 模型評估圖表。
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# ============================================================
# 第一部分：數據探索視覺化
# ============================================================

def plot_histogram(df, column, nbins=30, color=None):
    """繪製直方圖（含 KDE 密度曲線效果）"""
    fig = px.histogram(
        df, x=column, nbins=nbins, color=color,
        marginal="box",
        title=f"📊 {column} 分佈直方圖",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_scatter(df, x_col, y_col, color=None, trendline="ols"):
    """繪製散點圖（含趨勢線）"""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color,
        trendline=trendline,
        title=f"📈 {x_col} vs {y_col} 散點圖",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def plot_boxplot(df, column, group_by=None):
    """繪製箱型圖"""
    fig = px.box(
        df, x=group_by, y=column, color=group_by,
        title=f"📦 {column} 箱型圖" + (f"（分組: {group_by}）" if group_by else ""),
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def plot_bar_chart(df, x_col, y_col=None, color=None):
    """繪製長條圖（類別計數或聚合）"""
    if y_col is None:
        # 計數模式
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, '計數']
        fig = px.bar(
            counts, x=x_col, y='計數', color=x_col,
            title=f"📊 {x_col} 計數長條圖",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.bar(
            df, x=x_col, y=y_col, color=color,
            title=f"📊 {x_col} vs {y_col} 長條圖",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    return fig


def plot_pie_chart(df, column):
    """繪製圓餅圖"""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, '計數']
    fig = px.pie(
        counts, names=column, values='計數',
        title=f"🥧 {column} 比例圖",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_correlation_heatmap(df, columns=None, method='pearson'):
    """繪製相關矩陣熱圖"""
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
    else:
        numeric_df = df[columns]

    corr_matrix = numeric_df.corr(method=method)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"🔥 相關矩陣熱圖 ({method.capitalize()})",
        template="plotly_white",
        width=700, height=600,
    )
    return fig


def plot_pairplot(df, columns, color=None):
    """繪製配對散點圖矩陣"""
    fig = px.scatter_matrix(
        df, dimensions=columns, color=color,
        title="🔗 配對散點圖矩陣",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(width=800, height=800)
    return fig


# ============================================================
# 第二部分：ML 模型評估視覺化
# ============================================================

def plot_confusion_matrix(cm, class_names=None):
    """繪製混淆矩陣熱圖"""
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"預測: {c}" for c in class_names],
        y=[f"實際: {c}" for c in class_names],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False,
    ))
    fig.update_layout(
        title="🔢 混淆矩陣",
        xaxis_title="預測標籤",
        yaxis_title="實際標籤",
        template="plotly_white",
        width=500, height=450,
    )
    return fig


def plot_roc_curve(fpr, tpr, auc_score, model_name="Model"):
    """繪製 ROC 曲線（單模型）"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='隨機基線',
        line=dict(dash='dash', color='grey'),
    ))
    fig.update_layout(
        title=f"📉 ROC 曲線 — {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        width=600, height=500,
    )
    return fig


def plot_roc_curves_comparison(all_results):
    """繪製多模型 ROC 曲線比較"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, (name, res) in enumerate(all_results.items()):
        if res.get('fpr') is not None and res.get('tpr') is not None:
            fig.add_trace(go.Scatter(
                x=res['fpr'], y=res['tpr'],
                mode='lines',
                name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(width=2, color=colors[i % len(colors)]),
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines', name='隨機基線',
        line=dict(dash='dash', color='grey'),
    ))
    fig.update_layout(
        title="📉 多模型 ROC 曲線比較",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        width=700, height=550,
    )
    return fig


def plot_model_comparison_bar(comparison_df):
    """繪製多模型效能比較長條圖（F1, Precision, Recall, Accuracy）"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plot_df = comparison_df[['模型'] + metrics].copy()

    # 過濾掉有錯誤的模型
    for m in metrics:
        plot_df[m] = pd.to_numeric(plot_df[m], errors='coerce')
    plot_df = plot_df.dropna()

    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#FFA15A']
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=plot_df['模型'], y=plot_df[metric],
            name=metric,
            marker_color=colors[i],
            text=plot_df[metric].round(3),
            textposition='auto',
        ))

    fig.update_layout(
        title="📊 多模型效能比較",
        xaxis_title="模型",
        yaxis_title="分數",
        yaxis=dict(range=[0, 1.05]),
        barmode='group',
        template="plotly_white",
        width=900, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_kfold_results(cv_results_list):
    """繪製 K-Fold 交叉驗證結果箱型圖（支援 Accuracy 與 R²）"""
    data = []
    scoring_label = 'Score'
    for res in cv_results_list:
        scoring_label = res.get('scoring', 'accuracy').upper().replace('_', ' ')
        if scoring_label == 'ACCURACY':
            scoring_label = 'Accuracy'
        elif scoring_label == 'R2':
            scoring_label = 'R²'
        for score in res['scores']:
            data.append({'模型': res['model_name'], scoring_label: score})
    df_cv = pd.DataFrame(data)

    fig = px.box(
        df_cv, x='模型', y=scoring_label, color='模型',
        title=f"📦 K-Fold 交叉驗證結果比較 ({scoring_label})",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(width=800, height=500, showlegend=False)
    return fig


# ============================================================
# 第三部分：新功能 — 特徵重要性 & 迴歸評估
# ============================================================

def plot_feature_importance(feature_names, importances, model_name="Model", top_n=20):
    """
    繪製特徵重要性水平長條圖。
    支援樹模型 (feature_importances_) 與線性模型 (coef_)。
    正值為藍色，負值為紅色。
    """
    feature_names = list(feature_names)
    importances = np.array(importances)

    # 按絕對值排序，取 top_n
    n = min(top_n, len(feature_names))
    indices = np.argsort(np.abs(importances))[-n:]
    sorted_features = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    colors = ['#EF553B' if v < 0 else '#636EFA' for v in sorted_values]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in sorted_values],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"🔑 特徵重要性 — {model_name} (Top {n})",
        xaxis_title="重要性",
        yaxis_title="特徵",
        template="plotly_white",
        height=max(400, n * 28),
        margin=dict(l=200, r=80),
    )
    return fig


def plot_regression_actual_vs_predicted(y_test, y_pred, model_name="Model"):
    """
    繪製迴歸模型「實際值 vs 預測值」散點圖，含完美擬合線。
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        name='預測點',
        marker=dict(color='#636EFA', opacity=0.6, size=6),
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='完美擬合線',
        line=dict(color='red', dash='dash', width=2),
    ))
    fig.update_layout(
        title=f"📍 實際值 vs 預測值 — {model_name}",
        xaxis_title="實際值",
        yaxis_title="預測值",
        template="plotly_white",
        width=600, height=500,
    )
    return fig


def plot_regression_comparison_bar(comparison_df):
    """繪製多迴歸模型效能比較長條圖（R², RMSE, MAE）"""
    metrics = ['R²', 'RMSE', 'MAE']
    plot_df = comparison_df[['模型'] + metrics].copy()
    for m in metrics:
        plot_df[m] = pd.to_numeric(plot_df[m], errors='coerce')
    plot_df = plot_df.dropna()

    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96']
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=plot_df['模型'], y=plot_df[metric],
            name=metric,
            marker_color=colors[i],
            text=plot_df[metric].round(4),
            textposition='auto',
        ))

    fig.update_layout(
        title="📊 多迴歸模型效能比較",
        xaxis_title="模型",
        yaxis_title="指標值",
        barmode='group',
        template="plotly_white",
        width=900, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ============================================================
# 第四部分：風險建模視覺化
# ============================================================

def plot_ks_chart(ks_table, ks_value):
    """繪製 KS 曲線圖。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks_table["decile"], y=ks_table["cum_event_pct"],
        mode="lines+markers", name="累積事件 %",
        line=dict(color="#EF553B"),
    ))
    fig.add_trace(go.Scatter(
        x=ks_table["decile"], y=ks_table["cum_non_event_pct"],
        mode="lines+markers", name="累積非事件 %",
        line=dict(color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=ks_table["decile"], y=ks_table["ks"],
        mode="lines+markers", name="KS",
        line=dict(color="#00CC96", dash="dot"),
    ))
    fig.update_layout(
        title=f"KS Chart (KS = {ks_value:.4f})",
        xaxis_title="Decile", yaxis_title="Cumulative %",
        template="plotly_white", height=450,
    )
    return fig


def plot_lift_chart(lift_data):
    """繪製 Lift Chart。"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lift_data["decile"], y=lift_data["cumulative_lift"],
        name="Cumulative Lift",
        marker_color="#636EFA",
        text=lift_data["cumulative_lift"].round(2),
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Baseline (1.0)")
    fig.update_layout(
        title="Lift Chart",
        xaxis_title="Decile", yaxis_title="Cumulative Lift",
        template="plotly_white", height=400,
    )
    return fig


def plot_gain_chart(gain_data):
    """繪製 Gain (Cumulative Capture) Chart。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gain_data["percentile"], y=gain_data["cumulative_gain"],
        mode="lines+markers", name="Model",
        line=dict(color="#636EFA", width=2),
    ))
    # 隨機基線
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 1],
        mode="lines", name="Random",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        title="Gain Chart",
        xaxis_title="% Population", yaxis_title="% Events Captured",
        template="plotly_white", height=400,
    )
    return fig


def plot_woe_chart(woe_table, feature_name):
    """繪製 WOE 柱狀圖。"""
    woe_table = woe_table.copy()
    woe_table["bin_str"] = woe_table["bin"].astype(str)
    colors = ["#EF553B" if w < 0 else "#636EFA" for w in woe_table["woe"]]

    fig = go.Figure(go.Bar(
        x=woe_table["bin_str"], y=woe_table["woe"],
        marker_color=colors,
        text=woe_table["woe"].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"WOE — {feature_name}",
        xaxis_title="Bin", yaxis_title="WOE",
        template="plotly_white", height=400,
    )
    return fig


def plot_iv_ranking(iv_table):
    """繪製 IV 排名水平柱狀圖。"""
    iv_table = iv_table.sort_values("IV", ascending=True)

    def _color(power):
        mapping = {
            "無預測力": "#d3d3d3",
            "弱": "#FFA15A",
            "中等": "#636EFA",
            "強": "#00CC96",
            "極強 (需檢查)": "#EF553B",
        }
        return mapping.get(power, "#636EFA")

    colors = [_color(p) for p in iv_table["predictive_power"]]

    fig = go.Figure(go.Bar(
        x=iv_table["IV"], y=iv_table["feature"],
        orientation="h",
        marker_color=colors,
        text=iv_table["IV"].round(4),
        textposition="outside",
    ))
    # IV 門檻線
    for threshold, label in [(0.02, "0.02"), (0.1, "0.1"), (0.3, "0.3")]:
        fig.add_vline(x=threshold, line_dash="dot", line_color="gray",
                      annotation_text=label)

    fig.update_layout(
        title="IV Ranking",
        xaxis_title="Information Value",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, len(iv_table) * 28),
        margin=dict(l=200),
    )
    return fig


def plot_psi_comparison(psi_report):
    """繪製 PSI 柱狀圖。"""
    psi_report = psi_report.dropna(subset=["psi"]).sort_values("psi", ascending=True)

    def _color(stability):
        return {"穩定": "#00CC96", "輕微偏移": "#FFA15A", "顯著偏移": "#EF553B"}.get(stability, "#636EFA")

    colors = [_color(s) for s in psi_report["stability"]]

    fig = go.Figure(go.Bar(
        x=psi_report["psi"], y=psi_report["variable"],
        orientation="h",
        marker_color=colors,
        text=psi_report["psi"].round(4),
        textposition="outside",
    ))
    fig.add_vline(x=0.1, line_dash="dot", line_color="orange", annotation_text="0.1")
    fig.add_vline(x=0.25, line_dash="dot", line_color="red", annotation_text="0.25")

    fig.update_layout(
        title="PSI Report",
        xaxis_title="PSI", yaxis_title="Variable",
        template="plotly_white",
        height=max(400, len(psi_report) * 28),
        margin=dict(l=200),
    )
    return fig


def plot_distribution_shift(expected, actual, variable_name, bins=30):
    """繪製兩個分佈的疊加直方圖。"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=expected, name="基準期", opacity=0.6,
        marker_color="#636EFA", nbinsx=bins,
    ))
    fig.add_trace(go.Histogram(
        x=actual, name="監控期", opacity=0.6,
        marker_color="#EF553B", nbinsx=bins,
    ))
    fig.update_layout(
        title=f"Distribution Shift — {variable_name}",
        xaxis_title=variable_name, yaxis_title="Count",
        barmode="overlay",
        template="plotly_white", height=400,
    )
    return fig


if __name__ == '__main__':
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100) * 2 + 1,
        'C': np.random.choice(['X', 'Y', 'Z'], 100),
    })
    fig = plot_histogram(df, 'A')
    print("Histogram created OK")
    fig = plot_correlation_heatmap(df, ['A', 'B'])
    print("Correlation heatmap created OK")
