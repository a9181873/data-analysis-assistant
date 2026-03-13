"""Tab 5: 機器學習"""

import streamlit as st
import pandas as pd
import numpy as np
import config
from ml_models import (
    AVAILABLE_MODELS, BALANCE_STRATEGIES, get_balanced_models,
    get_regression_models, PARAM_GRIDS,
    prepare_data, train_single_model, train_regression_model,
    compare_models, compare_regression_models,
    kfold_cross_validation, apply_balancing, get_class_distribution,
    get_feature_importance, tune_hyperparameters, tune_with_optuna,
    IMBLEARN_AVAILABLE, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE, OPTUNA_AVAILABLE,
)
from visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_roc_curves_comparison,
    plot_model_comparison_bar, plot_kfold_results,
    plot_feature_importance, plot_regression_actual_vs_predicted,
    plot_regression_comparison_bar,
    plot_ks_chart, plot_lift_chart, plot_gain_chart,
)
from risk_metrics import ks_statistic, lift_chart_data, gain_chart_data
from model_export import export_model
from code_generator import generate_ml_pipeline_code, generate_r_pipeline_code

# SHAP (可選)
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def render(df: pd.DataFrame):
    st.subheader("機器學習")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()

    # 任務類型選擇
    task_type_label = st.radio(
        "選擇任務類型",
        ["分類 (Classification)", "迴歸 (Regression)"],
        horizontal=True,
        key="ml_task_type"
    )
    is_regression = "迴歸" in task_type_label
    task_type = config.ML_TASK_REGRESSION if is_regression else config.ML_TASK_CLASSIFICATION

    ml_mode = st.radio(
        "選擇模式",
        ["單模型訓練", "多模型比較", "K-Fold 交叉驗證",
         "超參數調整 (GridSearchCV)", "超參數調整 (Optuna)", "📖 演算法解說"],
        horizontal=True,
    )

    st.markdown("---")

    # --- 共用設定 ---
    st.write("**數據設定**")
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox(
            "目標變數 (Y)", all_cols,
            help="分類：選擇類別欄位；迴歸：選擇數值欄位"
        )
    with c2:
        available_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect("特徵變數 (X)", available_features,
                                      default=available_features)

    # 不平衡數據處理（只在分類任務顯示）
    if not is_regression:
        st.write("**不平衡數據處理**")
        c3, c4 = st.columns(2)
        with c3:
            balance_strategy = st.selectbox(
                "選擇處理策略",
                list(BALANCE_STRATEGIES.keys()),
                format_func=lambda x: BALANCE_STRATEGIES[x],
                help="當類別分佈不均時使用"
            )
        with c4:
            if balance_strategy in ("smote", "adasyn", "smote_tomek") and not IMBLEARN_AVAILABLE:
                st.warning("需安裝 `imbalanced-learn`: `pip install imbalanced-learn`")
    else:
        balance_strategy = "none"

    test_size = st.slider("測試集比例", 0.1, 0.4, config.DEFAULT_TEST_SIZE, 0.05)

    # 顯示目標變數分佈
    if target_col:
        with st.expander("目標變數分佈"):
            if is_regression and df[target_col].dtype in ['float64', 'float32', 'int64', 'int32']:
                st.bar_chart(df[target_col].describe().to_frame())
            else:
                class_counts = df[target_col].value_counts()
                st.bar_chart(class_counts)
                st.write(class_counts.to_frame("計數"))

    # 套件狀態提示
    status_info = []
    if XGBOOST_AVAILABLE:
        status_info.append("XGBoost")
    if LIGHTGBM_AVAILABLE:
        status_info.append("LightGBM")
    if CATBOOST_AVAILABLE:
        status_info.append("CatBoost")
    if OPTUNA_AVAILABLE:
        status_info.append("Optuna")
    if IMBLEARN_AVAILABLE:
        status_info.append("imbalanced-learn")
    if status_info:
        st.caption("已安裝: " + "  |  ".join(status_info))

    st.markdown("---")

    # 取得當前模型列表
    if is_regression:
        current_models = get_regression_models()
    else:
        current_models = get_balanced_models(balance_strategy)

    # --- 單模型訓練 ---
    if ml_mode == "單模型訓練":
        _render_single_model(df, current_models, target_col, feature_cols,
                             test_size, task_type, is_regression, balance_strategy,
                             raw_df=df)

    # --- 多模型比較 ---
    elif ml_mode == "多模型比較":
        _render_compare_models(df, current_models, target_col, feature_cols,
                               test_size, task_type, is_regression, balance_strategy)

    # --- K-Fold 交叉驗證 ---
    elif ml_mode == "K-Fold 交叉驗證":
        _render_kfold(df, current_models, target_col, feature_cols,
                      test_size, task_type)

    # --- 超參數調整 (GridSearchCV) ---
    elif ml_mode == "超參數調整 (GridSearchCV)":
        _render_grid_search(df, target_col, feature_cols, test_size, is_regression)

    # --- 超參數調整 (Optuna) ---
    elif ml_mode == "超參數調整 (Optuna)":
        _render_optuna(df, target_col, feature_cols, test_size, task_type, is_regression)

    # --- 演算法解說 ---
    elif ml_mode == "📖 演算法解說":
        _render_algorithm_guide(is_regression)


def _render_single_model(df, current_models, target_col, feature_cols,
                         test_size, task_type, is_regression, balance_strategy, raw_df=None):
    model_name = st.selectbox("選擇模型", list(current_models.keys()))

    if st.button("開始訓練", key="train_single"):
        if not feature_cols:
            st.error("請至少選擇一個特徵變數")
            return

        with st.spinner("模型訓練中..."):
            try:
                X_train_df, X_test_df, y_train, y_test, le, preprocessor = prepare_data(
                    df, target_col, feature_cols, test_size, task_type)
                
                # Get clean feature names if using ColumnTransformer
                feature_names_out = X_train_df.columns.tolist() if isinstance(X_train_df, pd.DataFrame) else feature_cols

                # 偵測原始資料格式（供 code generator 使用）
                file_fmt = "csv"

                if is_regression:
                    res = train_regression_model(
                        model_name, X_train_df, y_train, X_test_df, y_test)
                    _show_regression_results_full(res, model_name, feature_names_out,
                                                 X_train_df, X_test_df, preprocessor=preprocessor,
                                                 code_params=dict(target_col=target_col,
                                                     feature_cols=feature_cols, task_type=task_type,
                                                     model_name=model_name, balance_strategy=balance_strategy,
                                                     test_size=test_size, file_fmt=file_fmt))
                else:
                    X_train_bal, y_train_bal = apply_balancing(
                        X_train_df, y_train, balance_strategy)
                    if balance_strategy != "none":
                        st.info(f"平衡前: {len(y_train)} 筆 -> 平衡後: {len(y_train_bal)} 筆")

                    res = train_single_model(
                        model_name, X_train_bal, y_train_bal, X_test_df, y_test, le)
                    _show_classification_results(res, model_name, feature_names_out, le,
                                                X_train_bal, X_test_df, y_test, preprocessor=preprocessor,
                                                code_params=dict(target_col=target_col,
                                                    feature_cols=feature_cols, task_type=task_type,
                                                    model_name=model_name, balance_strategy=balance_strategy,
                                                    test_size=test_size, file_fmt=file_fmt))

            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")


def _render_compare_models(df, current_models, target_col, feature_cols,
                           test_size, task_type, is_regression, balance_strategy):
    selected_models = st.multiselect(
        "選擇要比較的模型",
        list(current_models.keys()),
        default=list(current_models.keys()),
    )

    if st.button("開始比較", key="compare_models"):
        if not feature_cols:
            st.error("請至少選擇一個特徵變數")
            return
        if not selected_models:
            st.error("請至少選擇一個模型")
            return

        with st.spinner("模型訓練與比較中..."):
            try:
                X_train_df, X_test_df, y_train, y_test, le, preprocessor = prepare_data(
                    df, target_col, feature_cols, test_size, task_type)
                
                # Get clean feature names if using ColumnTransformer
                feature_names_out = X_train_df.columns.tolist() if isinstance(X_train_df, pd.DataFrame) else feature_cols

                if is_regression:
                    comp_df, all_results = compare_regression_models(
                        X_train_df, y_train, X_test_df, y_test, selected_models)
                    st.success("迴歸模型比較完成!")
                    st.dataframe(comp_df, use_container_width=True)
                    fig_bar = plot_regression_comparison_bar(comp_df)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    X_train_bal, y_train_bal = apply_balancing(
                        X_train_df, y_train, balance_strategy)
                    comp_df, all_results = compare_models(
                        X_train_bal, y_train_bal, X_test_df, y_test,
                        selected_models, le)

                    st.success("分類模型比較完成!")
                    st.write("**模型效能比較表:**")
                    st.dataframe(comp_df, use_container_width=True)

                    fig_bar = plot_model_comparison_bar(comp_df)
                    st.plotly_chart(fig_bar, use_container_width=True)

                    has_roc = any(r.get('roc_auc') is not None
                                 for r in all_results.values())
                    if has_roc:
                        st.write("**ROC 曲線比較:**")
                        fig_roc = plot_roc_curves_comparison(all_results)
                        st.plotly_chart(fig_roc, use_container_width=True)

                    st.write("**各模型混淆矩陣:**")
                    cols_ui = st.columns(min(3, len(all_results)))
                    for i, (name, res) in enumerate(all_results.items()):
                        if 'confusion_matrix' in res:
                            with cols_ui[i % 3]:
                                class_names = le.classes_.tolist() if le else None
                                fig_cm = plot_confusion_matrix(
                                    res['confusion_matrix'], class_names)
                                fig_cm.update_layout(
                                    title=name.split(" (")[0], width=350, height=350)
                                st.plotly_chart(fig_cm)

                st.session_state.ml_results = all_results

            except Exception as e:
                st.error(f"比較失敗: {str(e)}")


def _render_kfold(df, current_models, target_col, feature_cols, test_size, task_type):
    selected_models = st.multiselect(
        "選擇模型",
        list(current_models.keys()),
        default=list(current_models.keys()),
        key="kfold_models",
    )
    n_folds = st.slider("K (折數)", 3, 10, config.DEFAULT_CV_FOLDS)

    if st.button("執行交叉驗證", key="run_kfold"):
        if not feature_cols:
            st.error("請至少選擇一個特徵變數")
            return
        if not selected_models:
            st.error("請至少選擇一個模型")
            return

        with st.spinner(f"{n_folds}-Fold 交叉驗證中..."):
            try:
                # 準備原始資料（不預先 scale，避免數據洩漏）
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
                from sklearn.pipeline import Pipeline

                model_df = df[feature_cols + [target_col]].dropna()
                X_all_raw = model_df[feature_cols].copy()
                y_all = model_df[target_col].copy()

                # 目標編碼
                le = None
                if task_type == config.ML_TASK_CLASSIFICATION:
                    if y_all.dtype == 'object' or str(y_all.dtype) == 'category':
                        le = LabelEncoder()
                        y_all = le.fit_transform(y_all.astype(str))
                    else:
                        y_all = y_all.astype(int).values

                # 建立 preprocessor（每 fold 內部重新 fit，防止洩漏）
                cat_cols = X_all_raw.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X_all_raw.select_dtypes(include=['number']).columns.tolist()
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), num_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
                    ],
                    remainder='passthrough',
                )

                cv_results = []
                for name in selected_models:
                    try:
                        res = kfold_cross_validation(
                            name, X_all_raw, y_all, n_folds, task_type,
                            preprocessor=preprocessor)
                        cv_results.append(res)
                    except Exception as e:
                        st.warning(f"{name}: {e}")

                if cv_results:
                    st.success("交叉驗證完成!")
                    scoring_label = cv_results[0].get('scoring', 'accuracy').upper().replace('_', ' ')
                    cv_table = pd.DataFrame([{
                        '模型': r['model_name'],
                        f'平均 {scoring_label}': f"{r['mean']:.4f}",
                        '標準差': f"{r['std']:.4f}",
                        '各折分數': ', '.join(f"{s:.3f}" for s in r['scores']),
                    } for r in cv_results])
                    st.dataframe(cv_table, use_container_width=True)

                    fig = plot_kfold_results(cv_results)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"交叉驗證失敗: {str(e)}")


def _render_grid_search(df, target_col, feature_cols, test_size, is_regression):
    tunable_models = [m for m in AVAILABLE_MODELS if m in PARAM_GRIDS]
    if not tunable_models:
        st.warning("沒有可調整的模型（需要在 PARAM_GRIDS 中定義搜索空間）")
        return

    tune_model = st.selectbox("選擇要調整的模型", tunable_models)
    cv_folds = st.slider("交叉驗證折數", 2, 5, config.DEFAULT_GRID_SEARCH_CV)

    st.info("GridSearchCV 可能需要較長時間，建議先用小資料集測試。")

    if is_regression:
        st.warning("超參數調整目前僅支援分類任務，請切換至分類模式。")
        return

    if st.button("開始超參數調整", key="run_tuning"):
        if not feature_cols:
            st.error("請至少選擇一個特徵變數")
            return

        with st.spinner(f"GridSearchCV 執行中 (模型: {tune_model})..."):
            try:
                X_train, X_test, y_train, y_test, le, scaler = prepare_data(
                    df, target_col, feature_cols, test_size,
                    config.ML_TASK_CLASSIFICATION)

                # 取得轉換後的特徵名稱
                feature_names_out = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else feature_cols

                tuning_res = tune_hyperparameters(
                    tune_model, X_train, y_train, cv=cv_folds)

                st.success("超參數調整完成!")

                st.write("**最佳超參數:**")
                st.json(tuning_res['best_params'])
                st.metric("最佳交叉驗證準確率",
                         f"{tuning_res['best_score']:.4f}")

                best_model = tuning_res['best_estimator']
                y_pred = best_model.predict(X_test)
                from sklearn.metrics import accuracy_score
                test_acc = accuracy_score(y_test, y_pred)
                st.metric("測試集準確率", f"{test_acc:.4f}")

                feat_names, importances = get_feature_importance(
                    best_model, feature_names_out)
                if feat_names is not None:
                    st.write("**最佳模型特徵重要性:**")
                    fig_fi = plot_feature_importance(
                        feat_names, importances, tune_model)
                    st.plotly_chart(fig_fi, use_container_width=True)

                with st.expander("完整 CV 結果"):
                    show_cols = ['params', 'mean_test_score', 'std_test_score',
                                 'rank_test_score']
                    cv_df = tuning_res['cv_results'][
                        [c for c in show_cols if c in tuning_res['cv_results'].columns]
                    ].sort_values('rank_test_score')
                    st.dataframe(cv_df, use_container_width=True)

            except Exception as e:
                st.error(f"超參數調整失敗: {str(e)}")


def _show_regression_results(res, model_name, feature_cols):
    st.success("迴歸模型訓練完成!")
    m1, m2, m3 = st.columns(3)
    m1.metric("R\u00b2", f"{res['r2']:.4f}")
    m2.metric("RMSE", f"{res['rmse']:.4f}")
    m3.metric("MAE", f"{res['mae']:.4f}")

    fig_avp = plot_regression_actual_vs_predicted(
        res['y_test'], res['y_pred'], model_name)
    st.plotly_chart(fig_avp, use_container_width=True)

    feat_names, importances = get_feature_importance(res['model'], feature_cols)
    if feat_names is not None:
        st.write("**特徵重要性:**")
        fig_fi = plot_feature_importance(feat_names, importances, model_name)
        st.plotly_chart(fig_fi, use_container_width=True)


def _show_classification_results(res, model_name, feature_cols, le,
                                 X_train=None, X_test=None, y_test=None, preprocessor=None,
                                 code_params=None):
    st.success("分類模型訓練完成!")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{res['accuracy']:.4f}")
    m2.metric("Precision", f"{res['precision']:.4f}")
    m3.metric("Recall", f"{res['recall']:.4f}")
    m4.metric("F1 Score", f"{res['f1']:.4f}")

    st.write("**分類報告:**")
    st.text(res['classification_report'])

    class_names = le.classes_.tolist() if le else None
    fig_cm = plot_confusion_matrix(res['confusion_matrix'], class_names)
    st.plotly_chart(fig_cm, use_container_width=True)

    if res['roc_auc'] is not None:
        fig_roc = plot_roc_curve(
            res['fpr'], res['tpr'], res['roc_auc'], model_name)
        st.plotly_chart(fig_roc, use_container_width=True)

    feat_names, importances = get_feature_importance(res['model'], feature_cols)
    if feat_names is not None:
        st.write("**特徵重要性:**")
        fig_fi = plot_feature_importance(feat_names, importances, model_name)
        st.plotly_chart(fig_fi, use_container_width=True)

    # --- 風控指標 (二分類 + predict_proba) ---
    if res['roc_auc'] is not None and hasattr(res['model'], 'predict_proba') and y_test is not None:
        y_prob = res['model'].predict_proba(X_test)[:, 1]
        with st.expander("風控指標 (KS / Lift / Gain)"):
            try:
                ks_val, ks_table = ks_statistic(y_test, y_prob)
                st.metric("KS Statistic", f"{ks_val:.4f}")
                fig_ks = plot_ks_chart(ks_table, ks_val)
                st.plotly_chart(fig_ks, use_container_width=True)

                lift_data = lift_chart_data(y_test, y_prob)
                fig_lift = plot_lift_chart(lift_data)
                st.plotly_chart(fig_lift, use_container_width=True)

                gain_data = gain_chart_data(y_test, y_prob)
                fig_gain = plot_gain_chart(gain_data)
                st.plotly_chart(fig_gain, use_container_width=True)
            except Exception as e:
                st.warning(f"風控指標計算失敗: {e}")

    # --- SHAP 解釋 ---
    if SHAP_AVAILABLE and X_test is not None:
        with st.expander("SHAP 特徵解釋"):
            try:
                model = res['model']
                X_explain = X_test[:100]
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train[:100] if X_train is not None else X_explain)

                shap_values = explainer(X_explain)

                fig_shap, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_explain,
                                  feature_names=feature_cols, show=False)
                st.pyplot(fig_shap)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP 計算失敗: {e}")

    # --- 模型匯出 ---
    _show_model_export(res, model_name, feature_cols, "classification", preprocessor=preprocessor)

    # --- 程式碼產生 ---
    if code_params:
        _show_code_generator(**code_params)


def _show_regression_results_full(res, model_name, feature_cols,
                                  X_train=None, X_test=None, preprocessor=None,
                                  code_params=None):
    """迴歸結果 + SHAP + 匯出。"""
    _show_regression_results(res, model_name, feature_cols)

    if SHAP_AVAILABLE and X_test is not None:
        with st.expander("SHAP 特徵解釋"):
            try:
                model = res['model']
                X_explain = X_test[:100]
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train[:100] if X_train is not None else X_explain)

                shap_values = explainer(X_explain)

                fig_shap, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_explain,
                                  feature_names=feature_cols, show=False)
                st.pyplot(fig_shap)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP 計算失敗: {e}")

    _show_model_export(res, model_name, feature_cols, "regression", preprocessor=preprocessor)

    # --- 程式碼產生 ---
    if code_params:
        _show_code_generator(**code_params)


def _show_model_export(res, model_name, feature_cols, task_type, preprocessor=None):
    """顯示模型下載按鈕。"""
    metrics_dict = {k: v for k, v in res.items()
                    if isinstance(v, (int, float, str)) and k != 'model'}
    try:
        buf = export_model(
            model=res['model'],
            scaler=preprocessor,
            label_encoder=None,
            feature_names=feature_cols,
            metrics=metrics_dict,
            model_name=model_name,
            task_type=task_type,
        )
        safe_name = model_name.split(" (")[0].replace(" ", "_")
        st.download_button(
            "下載模型 (.joblib)",
            data=buf,
            file_name=f"{safe_name}.joblib",
            mime="application/octet-stream",
        )
    except Exception as e:
        st.warning(f"模型匯出失敗: {e}")


def _render_optuna(df, target_col, feature_cols, test_size, task_type, is_regression):
    """Optuna 貝葉斯超參數優化面板。"""
    if not OPTUNA_AVAILABLE:
        st.error("需要安裝 Optuna 套件。請執行: `pip install optuna`")
        return

    from ml_models import OPTUNA_PARAM_SPACES
    tunable = [m for m in (AVAILABLE_MODELS if not is_regression else get_regression_models())
               if m in OPTUNA_PARAM_SPACES]
    if not tunable:
        st.warning("沒有可用 Optuna 調整的模型。")
        return

    tune_model = st.selectbox("選擇要調整的模型", tunable, key="optuna_model")
    c1, c2 = st.columns(2)
    with c1:
        n_trials = st.slider("搜索次數 (n_trials)", 10, 200, 50, key="optuna_trials")
    with c2:
        cv_folds = st.slider("交叉驗證折數", 2, 5, config.DEFAULT_GRID_SEARCH_CV, key="optuna_cv")

    st.info("Optuna 使用貝葉斯優化 (TPE)，比 GridSearchCV 更高效 — 不需窮舉所有組合。")

    if st.button("開始 Optuna 優化", key="run_optuna", type="primary"):
        if not feature_cols:
            st.error("請至少選擇一個特徵變數")
            return

        with st.spinner(f"Optuna 優化中 ({n_trials} trials, 模型: {tune_model})..."):
            try:
                X_train, X_test, y_train, y_test, le, preprocessor = prepare_data(
                    df, target_col, feature_cols, test_size, task_type)
                feature_names_out = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else feature_cols

                scoring = 'accuracy' if not is_regression else 'r2'
                result = tune_with_optuna(
                    tune_model, X_train, y_train,
                    n_trials=n_trials, cv=cv_folds,
                    scoring=scoring, task_type=task_type)

                st.success(f"Optuna 優化完成！({result['n_trials']} trials)")

                st.write("**最佳超參數:**")
                st.json(result['best_params'])

                m1, m2 = st.columns(2)
                m1.metric(f"最佳 CV {scoring.upper()}", f"{result['best_score']:.4f}")

                best_model = result['best_estimator']
                y_pred = best_model.predict(X_test)
                if is_regression:
                    from sklearn.metrics import r2_score
                    m2.metric("測試集 R²", f"{r2_score(y_test, y_pred):.4f}")
                else:
                    from sklearn.metrics import accuracy_score
                    m2.metric("測試集 Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

                feat_names, importances = get_feature_importance(best_model, feature_names_out)
                if feat_names is not None:
                    st.write("**最佳模型特徵重要性:**")
                    fig_fi = plot_feature_importance(feat_names, importances, tune_model)
                    st.plotly_chart(fig_fi, use_container_width=True)

                # 優化歷程圖
                with st.expander("Optuna 優化歷程"):
                    study = result['study']
                    trial_df = study.trials_dataframe()
                    if not trial_df.empty:
                        import plotly.express as px
                        fig_hist = px.line(
                            trial_df, x="number", y="value",
                            title="Trial 分數變化",
                            labels={"number": "Trial", "value": scoring.upper()},
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.write("**Top 10 Trials:**")
                        top10 = trial_df.nsmallest(10, 'number') if trial_df['value'].iloc[-1] < trial_df['value'].iloc[0] else trial_df.nlargest(10, 'value')
                        st.dataframe(top10[['number', 'value', 'state']].head(10),
                                     use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Optuna 優化失敗: {str(e)}")


def _render_algorithm_guide(is_regression):
    """演算法解說面板 — 每個演算法的原理、適用場景、優缺點。"""
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#519D9E 0%,#58C9B9 100%);
                    border-radius:10px;padding:14px 20px;margin-bottom:16px;">
            <span style="color:#fff;font-size:1.1rem;font-weight:700;">
                📖 機器學習演算法完整解說
            </span><br>
            <span style="color:#dff6f4;font-size:0.87rem;">
                了解每個演算法的原理、適用場景與限制，幫助你選擇最合適的模型。
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_regression:
        _ALGO_GUIDE = _REGRESSION_ALGO_GUIDE
    else:
        _ALGO_GUIDE = _CLASSIFICATION_ALGO_GUIDE

    for algo in _ALGO_GUIDE:
        with st.expander(f"{algo['icon']} {algo['name']}"):
            st.markdown(f"**一句話描述：** {algo['summary']}")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔬 原理**")
                st.markdown(algo['how_it_works'])

                st.markdown("**✅ 適合場景**")
                for s in algo['good_for']:
                    st.markdown(f"- {s}")

            with c2:
                st.markdown("**⚠️ 不適合 / 限制**")
                for s in algo['not_good_for']:
                    st.markdown(f"- {s}")

                st.markdown("**💡 實務建議**")
                st.markdown(algo['tips'])

            if algo.get('complexity'):
                st.caption(f"⏱️ 訓練速度：{algo['complexity']}")


# ── 分類演算法解說資料 ──────────────────────────────────────────────
_CLASSIFICATION_ALGO_GUIDE = [
    {
        "name": "Logistic Regression (邏輯迴歸)",
        "icon": "📐",
        "summary": "用 S 型函數 (sigmoid) 將線性組合映射為機率，是分類問題的「基準線模型」。",
        "how_it_works": (
            "對特徵做線性加權求和，再通過 sigmoid 函數轉換為 0~1 的機率值。"
            "透過最大化似然函數（MLE）學習權重，輸出可直接解釋為「正類機率」。"
        ),
        "good_for": [
            "二分類問題（信用評分、違約預測）",
            "需要模型可解釋性（係數 = 特徵影響方向與大小）",
            "特徵與目標呈線性關係",
            "作為 baseline 模型快速驗證",
        ],
        "not_good_for": [
            "特徵與目標之間有複雜非線性關係",
            "特徵之間有嚴重多元共線性（需先做 VIF 篩選）",
            "高維稀疏數據（需搭配正則化 L1/L2）",
        ],
        "tips": "風控建模的標準起點。先用 Logistic Regression 建立 baseline，再嘗試樹模型比較。搭配 WOE 轉換效果更好。",
        "complexity": "極快 — O(n × p)，適合大數據集",
    },
    {
        "name": "Decision Tree (決策樹)",
        "icon": "🌳",
        "summary": "像流程圖一樣，依照「最佳切分條件」逐層分割數據，直到每個葉節點足夠純淨。",
        "how_it_works": (
            "在每個節點選擇使 Gini 不純度或資訊熵下降最多的特徵與切分點，"
            "遞迴分割直到達到停止條件（最大深度、最小樣本數等）。"
        ),
        "good_for": [
            "需要視覺化解釋模型決策邏輯",
            "特徵包含數值與類別混合型",
            "數據有非線性關係",
            "快速探索特徵重要性",
        ],
        "not_good_for": [
            "容易過擬合（尤其深度不限制時）",
            "對數據微小變化敏感（不穩定）",
            "不擅長捕捉線性關係（階梯式近似）",
        ],
        "tips": "單棵決策樹通常只用於探索，實戰中建議使用 Random Forest 或 XGBoost 等集成方法。控制 max_depth 是防過擬合的關鍵。",
        "complexity": "快 — O(n × p × log(n))",
    },
    {
        "name": "Random Forest (隨機森林)",
        "icon": "🌲",
        "summary": "同時訓練大量隨機化的決策樹，用「多數投票」做最終預測，大幅降低過擬合風險。",
        "how_it_works": (
            "使用 Bagging（Bootstrap Aggregating）：每棵樹只看到隨機抽樣的數據子集和特徵子集，"
            "最終結果由所有樹投票決定。隨機性 + 聚合 = 穩定且強大。"
        ),
        "good_for": [
            "幾乎所有表格型數據問題（分類/迴歸）",
            "特徵數量多、不確定哪些重要",
            "不需要太多超參數調整就能有好結果",
            "需要特徵重要性排名",
        ],
        "not_good_for": [
            "超高維稀疏數據（如 NLP 文本特徵）",
            "需要即時推論的低延遲場景（模型體積大）",
            "需要係數級解釋性（黑箱模型）",
        ],
        "tips": "通常 100~300 棵樹就足夠。n_estimators 增加能穩定效能但不會過擬合。是「什麼都不確定時的安全選擇」。",
        "complexity": "中等 — O(n_trees × n × p × log(n))",
    },
    {
        "name": "SVM (支援向量機)",
        "icon": "📏",
        "summary": "在高維空間中找到「最大間距」的超平面來分隔不同類別。",
        "how_it_works": (
            "尋找能最大化兩類之間「邊界（margin）」的超平面。透過核函數（RBF, Linear）"
            "將數據映射到更高維空間，使原本不可分的數據變得可分。"
        ),
        "good_for": [
            "中小型數據集（< 10,000 筆）效果優異",
            "高維特徵空間（特徵數 > 樣本數）",
            "二分類問題",
            "文本分類、影像分類等",
        ],
        "not_good_for": [
            "大數據集（> 50,000 筆，訓練很慢）",
            "多分類問題（需拆成多個二分類）",
            "需要機率輸出（需額外計算，較慢）",
            "特徵需要標準化",
        ],
        "tips": "記得先做特徵標準化！RBF 核是預設好選擇。C 和 gamma 是最重要的超參數。大數據集建議改用 LinearSVC。",
        "complexity": "慢 — O(n² × p) 到 O(n³)，不適合大數據",
    },
    {
        "name": "KNN (K-近鄰)",
        "icon": "👥",
        "summary": "找到最接近的 K 個鄰居，用他們的多數類別作為預測結果 — 最直覺的演算法。",
        "how_it_works": (
            "不建立模型（懶惰學習），預測時計算新樣本與所有訓練樣本的距離，"
            "取最近的 K 個鄰居投票。距離通常用歐氏距離。"
        ),
        "good_for": [
            "小數據集的快速原型",
            "類別邊界不規則的問題",
            "推薦系統、異常偵測",
        ],
        "not_good_for": [
            "大數據集（預測時需計算所有距離，非常慢）",
            "高維數據（維度詛咒，距離失去意義）",
            "特徵尺度不一致時（必須標準化）",
            "類別不平衡嚴重時",
        ],
        "tips": "K 取奇數避免平票。必須做特徵標準化。通常只適合特徵 < 20 維的小數據集。",
        "complexity": "訓練瞬間，預測慢 — O(n × p) per query",
    },
    {
        "name": "Naive Bayes (樸素貝葉斯)",
        "icon": "📊",
        "summary": "基於貝氏定理，假設所有特徵之間互相獨立，用條件機率做分類。",
        "how_it_works": (
            "利用 P(類別|特徵) ∝ P(特徵|類別) × P(類別)，"
            "假設各特徵條件獨立（「樸素」假設），分別計算每個特徵的條件機率再相乘。"
        ),
        "good_for": [
            "文本分類（垃圾郵件偵測、情感分析）",
            "特徵真的接近獨立的情況",
            "訓練數據量少但維度高",
            "需要極快的訓練與預測速度",
        ],
        "not_good_for": [
            "特徵之間有強相關性（違反獨立假設）",
            "需要精確機率估計的場景",
            "數值特徵分佈非常態（GaussianNB 假設常態）",
        ],
        "tips": "在文本分類中效果出乎意料地好。對表格型結構化數據通常不如樹模型。可作為快速 baseline。",
        "complexity": "極快 — O(n × p)",
    },
    {
        "name": "LDA (線性判別分析)",
        "icon": "📐",
        "summary": "同時做降維和分類 — 找到最能區分各類別的投影方向。",
        "how_it_works": (
            "計算各類別的均值向量和共變異矩陣，找到使「類間距離最大、類內距離最小」的投影方向。"
            "本質上是在做有監督的降維 + 分類。"
        ),
        "good_for": [
            "特徵之間有多元共線性（LDA 天然處理）",
            "需要同時降維與分類",
            "類別數量少（2~5 類）",
            "特徵近似常態分佈且各類共變異矩陣相近",
        ],
        "not_good_for": [
            "類別數很多（降維後最多 C-1 維）",
            "特徵與目標有非線性關係",
            "各類的共變異矩陣差異很大",
            "樣本數少於特徵數（矩陣奇異，需正則化）",
        ],
        "tips": "在風控和生物統計中常用。如果 Logistic Regression 效果好，LDA 通常也不錯。可搭配 PCA 先降維。",
        "complexity": "極快 — O(n × p²)",
    },
    {
        "name": "XGBoost",
        "icon": "🚀",
        "summary": "Kaggle 競賽常勝軍。透過梯度提升逐步修正前一棵樹的錯誤，加入正則化防止過擬合。",
        "how_it_works": (
            "逐步建立決策樹，每棵新樹專注於修正前面所有樹的預測殘差。"
            "內建 L1/L2 正則化、缺失值處理、特徵子採樣，支援平行計算。"
        ),
        "good_for": [
            "結構化/表格型數據的首選模型",
            "Kaggle 競賽、風控建模",
            "特徵數量多且有交互效應",
            "需要高預測精度",
        ],
        "not_good_for": [
            "非結構化數據（圖片、文字，用深度學習）",
            "訓練數據極少（< 100 筆，容易過擬合）",
            "需要模型完全可解釋（可搭配 SHAP）",
        ],
        "tips": "三個最重要的超參數：learning_rate（通常 0.01~0.1）、max_depth（3~8）、n_estimators（100~1000，配合 early_stopping）。",
        "complexity": "中等 — 支援 GPU 加速，大數據集表現好",
    },
    {
        "name": "LightGBM",
        "icon": "⚡",
        "summary": "微軟開發的輕量級梯度提升框架，比 XGBoost 更快，尤其適合大數據和高維特徵。",
        "how_it_works": (
            "使用 Leaf-wise 生長策略（而非 Level-wise），優先分裂增益最大的葉子。"
            "支援 GOSS（梯度單邊採樣）和 EFB（互斥特徵綁定）加速訓練。"
        ),
        "good_for": [
            "大數據集（> 100,000 筆），訓練速度比 XGBoost 快 2-10 倍",
            "高維特徵（自動處理類別特徵）",
            "記憶體有限的環境",
            "需要快速迭代實驗",
        ],
        "not_good_for": [
            "小數據集（< 1,000 筆，Leaf-wise 容易過擬合）",
            "對過擬合非常敏感的場景（需仔細調參）",
        ],
        "tips": "小數據集用 XGBoost 更穩；大數據集 LightGBM 更快。num_leaves 是最重要的參數（預設 31，小數據建議降低）。",
        "complexity": "快 — 比 XGBoost 快 2-10x",
    },
    {
        "name": "CatBoost",
        "icon": "🐱",
        "summary": "Yandex 開發的梯度提升框架，對類別特徵有原生支援，無需手動 One-Hot Encoding。",
        "how_it_works": (
            "使用 Ordered Target Encoding 處理類別特徵（避免目標洩漏），"
            "搭配 Ordered Boosting 減少預測偏差。對類別特徵自動做最優的組合與編碼。"
        ),
        "good_for": [
            "類別特徵多的數據集（最大優勢）",
            "不想花時間做特徵工程",
            "需要處理缺失值（自動處理）",
            "想要開箱即用的好效果（預設參數就很好）",
        ],
        "not_good_for": [
            "純數值特徵的數據集（優勢不明顯）",
            "對訓練速度有極端要求（比 LightGBM 稍慢）",
            "模型體積敏感的部署場景",
        ],
        "tips": "類別特徵多時，CatBoost 通常比 XGBoost/LightGBM 更好。iterations（樹數量）和 depth（樹深度）是最重要的超參數。",
        "complexity": "中等 — 支援 GPU 加速",
    },
]


# ── 迴歸演算法解說資料 ──────────────────────────────────────────────
_REGRESSION_ALGO_GUIDE = [
    {
        "name": "Linear Regression (線性迴歸)",
        "icon": "📏",
        "summary": "找出特徵與目標之間的最佳線性關係，是迴歸問題最基本的模型。",
        "how_it_works": "使用最小平方法（OLS）最小化預測值與實際值的平方差總和，求出最佳的權重係數。",
        "good_for": ["特徵與目標呈線性關係", "需要可解釋的係數", "作為 baseline 模型"],
        "not_good_for": ["非線性關係", "特徵多元共線性嚴重", "存在離群值（對極端值敏感）"],
        "tips": "先檢查殘差是否隨機分佈。若有共線性，改用 Ridge 或 Lasso。",
        "complexity": "極快",
    },
    {
        "name": "Ridge Regression (嶺迴歸)",
        "icon": "🏔️",
        "summary": "加入 L2 正則化的線性迴歸，防止過擬合並處理多元共線性。",
        "how_it_works": "在 OLS 的損失函數中加入 α × Σ(w²) 懲罰項，限制權重不能太大，迫使模型更簡單。",
        "good_for": ["特徵之間有共線性", "特徵數接近或超過樣本數", "防止過擬合"],
        "not_good_for": ["需要自動特徵選擇（Ridge 不會將權重歸零）", "特徵稀疏性很重要"],
        "tips": "alpha 越大正則化越強。交叉驗證選最佳 alpha。",
        "complexity": "極快",
    },
    {
        "name": "Lasso Regression",
        "icon": "✂️",
        "summary": "加入 L1 正則化的線性迴歸，能自動將不重要的特徵權重歸零（特徵選擇）。",
        "how_it_works": "在損失函數中加入 α × Σ|w| 懲罰項，L1 範數會將部分權重壓到恰好為 0。",
        "good_for": ["自動特徵選擇（篩掉不重要的特徵）", "高維稀疏數據", "需要簡潔模型"],
        "not_good_for": ["高度相關的特徵群組（Lasso 只選其中一個）", "特徵數遠多於樣本數時不穩定"],
        "tips": "想保留所有特徵用 Ridge，想篩選特徵用 Lasso，不確定時用 ElasticNet。",
        "complexity": "極快",
    },
    {
        "name": "Decision Tree Regressor (決策樹迴歸)",
        "icon": "🌳",
        "summary": "用決策樹結構做迴歸預測，每個葉節點輸出該區域的平均值。",
        "how_it_works": "遞迴分割特徵空間，最小化每個區域內的 MSE，最終每個葉節點的預測值是該區域的樣本平均。",
        "good_for": ["非線性關係", "快速探索特徵交互效應", "不需標準化"],
        "not_good_for": ["容易過擬合", "不穩定（數據微小變化導致完全不同的樹）", "外推能力差"],
        "tips": "單棵決策樹不建議用於最終模型，建議用 Random Forest 或 Gradient Boosting。",
        "complexity": "快",
    },
    {
        "name": "Random Forest Regressor (隨機森林迴歸)",
        "icon": "🌲",
        "summary": "多棵隨機化決策樹的平均預測，穩定且不易過擬合。",
        "how_it_works": "與分類版相同的 Bagging 機制，最終預測值為所有樹的平均值（而非投票）。",
        "good_for": ["通用迴歸問題", "不確定數據特性時的安全選擇", "特徵重要性排名"],
        "not_good_for": ["外推預測（超出訓練範圍的值）", "需要精確的線性關係擬合"],
        "tips": "迴歸任務中 n_estimators=200~500 通常足夠。對外推問題（預測未見過的極端值）表現較差。",
        "complexity": "中等",
    },
    {
        "name": "Gradient Boosting Regressor (梯度提升迴歸)",
        "icon": "📈",
        "summary": "逐步建立弱學習器修正殘差，是 XGBoost/LightGBM 的基礎版本。",
        "how_it_works": "每棵新樹擬合前面所有樹的預測殘差，學習率控制每棵樹的貢獻程度。",
        "good_for": ["需要高精度預測", "特徵有交互效應", "中型數據集"],
        "not_good_for": ["訓練較慢（序列化訓練）", "需仔細調參防止過擬合"],
        "tips": "learning_rate 降低 + n_estimators 增加通常效果更好，但訓練更慢。",
        "complexity": "中等偏慢",
    },
    {
        "name": "XGBoost Regressor",
        "icon": "🚀",
        "summary": "XGBoost 的迴歸版本，支援正則化、缺失值處理、平行計算。",
        "how_it_works": "與分類版相同，但損失函數改為 MSE，輸出為連續值。",
        "good_for": ["結構化數據迴歸的首選", "需要高精度", "大數據集"],
        "not_good_for": ["超小數據集", "外推預測"],
        "tips": "同分類版建議：learning_rate=0.01~0.1, max_depth=3~8, 搭配 early_stopping。",
        "complexity": "中等",
    },
    {
        "name": "LightGBM Regressor",
        "icon": "⚡",
        "summary": "LightGBM 的迴歸版本，大數據集訓練速度最快。",
        "how_it_works": "與分類版相同的 Leaf-wise + GOSS + EFB 機制。",
        "good_for": ["大數據集（> 100K 筆）", "快速實驗迭代", "記憶體有限"],
        "not_good_for": ["小數據集容易過擬合", "需仔細調 num_leaves"],
        "tips": "num_leaves 控制模型複雜度，小數據建議 < 31。",
        "complexity": "快",
    },
    {
        "name": "CatBoost Regressor",
        "icon": "🐱",
        "summary": "CatBoost 的迴歸版本，對類別特徵有原生支援。",
        "how_it_works": "與分類版相同的 Ordered Target Encoding + Ordered Boosting。",
        "good_for": ["類別特徵多的迴歸問題", "開箱即用", "缺失值自動處理"],
        "not_good_for": ["純數值特徵（優勢不明顯）", "極端速度要求"],
        "tips": "類別特徵多時首選。iterations 和 depth 是最重要的超參數。",
        "complexity": "中等",
    },
]


def _show_code_generator(target_col, feature_cols, task_type, model_name,
                         balance_strategy, test_size, file_fmt="csv"):
    """在 UI 展示動態產生的 Jupyter-ready Python 與 R 程式碼。"""
    with st.expander("📝 檢視 / 下載建模程式碼 (Python / R)"):
        tab_py, tab_r = st.tabs(["🐍 Python", "🔵 R"])
        safe_name = model_name.split(" (")[0].replace(" ", "_").lower()
        
        with tab_py:
            try:
                code_str_py = generate_ml_pipeline_code(
                    target_col=target_col,
                    feature_cols=feature_cols,
                    task_type=task_type,
                    model_name=model_name,
                    balance_strategy=balance_strategy,
                    test_size=test_size,
                    file_format_hint=file_fmt,
                )
                st.code(code_str_py, language="python")
                
                st.download_button(
                    label="⬇️ 下載 train_model.py",
                    data=code_str_py.encode("utf-8"),
                    file_name=f"train_{safe_name}.py",
                    mime="text/x-python",
                    key=f"dl_code_py_{model_name}_{target_col}",
                )
            except Exception as e:
                st.warning(f"Python 程式碼產生失敗: {e}")

        with tab_r:
            try:
                code_str_r = generate_r_pipeline_code(
                    target_col=target_col,
                    feature_cols=feature_cols,
                    task_type=task_type,
                    model_name=model_name,
                    balance_strategy=balance_strategy,
                    test_size=test_size,
                    file_format_hint=file_fmt,
                )
                st.code(code_str_r, language="r")
                
                st.download_button(
                    label="⬇️ 下載 train_model.R",
                    data=code_str_r.encode("utf-8"),
                    file_name=f"train_{safe_name}.R",
                    mime="text/plain",
                    key=f"dl_code_r_{model_name}_{target_col}",
                )
            except Exception as e:
                st.warning(f"R 程式碼產生失敗: {e}")

