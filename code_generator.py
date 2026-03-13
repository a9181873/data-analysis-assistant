"""
code_generator.py
根據使用者在 UI 上的設定，動態產生完整的機器學習 Python 建模腳本。
產生的程式碼可直接複製貼上至 Jupyter Notebook 或 .py 執行。
"""

from typing import List


# ── 模型名稱 → import + 建構程式碼 的映射表 ─────────────────────────────
_MODEL_MAP = {
    "Logistic Regression (邏輯迴歸)": {
        "import": "from sklearn.linear_model import LogisticRegression",
        "init": "LogisticRegression(max_iter=1000, random_state=42)",
    },
    "Decision Tree (決策樹)": {
        "import": "from sklearn.tree import DecisionTreeClassifier",
        "init": "DecisionTreeClassifier(random_state=42)",
    },
    "Random Forest (隨機森林)": {
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "init": "RandomForestClassifier(n_estimators=100, random_state=42)",
    },
    "SVM (支援向量機)": {
        "import": "from sklearn.svm import SVC",
        "init": "SVC(probability=True, random_state=42)",
    },
    "KNN (K-近鄰)": {
        "import": "from sklearn.neighbors import KNeighborsClassifier",
        "init": "KNeighborsClassifier(n_neighbors=5)",
    },
    "Naive Bayes (樸素貝葉斯)": {
        "import": "from sklearn.naive_bayes import GaussianNB",
        "init": "GaussianNB()",
    },
    "XGBoost": {
        "import": "from xgboost import XGBClassifier",
        "init": "XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)",
    },
    "LightGBM": {
        "import": "from lightgbm import LGBMClassifier",
        "init": "LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)",
    },
    "CatBoost": {
        "import": "from catboost import CatBoostClassifier",
        "init": "CatBoostClassifier(iterations=100, random_state=42, verbose=0)",
    },
    "LDA (線性判別分析)": {
        "import": "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
        "init": "LinearDiscriminantAnalysis()",
    },
    # ── 迴歸 ──
    "Linear Regression (線性迴歸)": {
        "import": "from sklearn.linear_model import LinearRegression",
        "init": "LinearRegression()",
    },
    "Ridge Regression (嶺迴歸)": {
        "import": "from sklearn.linear_model import Ridge",
        "init": "Ridge(alpha=1.0)",
    },
    "Lasso Regression": {
        "import": "from sklearn.linear_model import Lasso",
        "init": "Lasso(alpha=1.0, max_iter=5000)",
    },
    "Decision Tree Regressor (決策樹迴歸)": {
        "import": "from sklearn.tree import DecisionTreeRegressor",
        "init": "DecisionTreeRegressor(random_state=42)",
    },
    "Random Forest Regressor (隨機森林迴歸)": {
        "import": "from sklearn.ensemble import RandomForestRegressor",
        "init": "RandomForestRegressor(n_estimators=100, random_state=42)",
    },
    "Gradient Boosting Regressor (梯度提升迴歸)": {
        "import": "from sklearn.ensemble import GradientBoostingRegressor",
        "init": "GradientBoostingRegressor(n_estimators=100, random_state=42)",
    },
    "XGBoost Regressor": {
        "import": "from xgboost import XGBRegressor",
        "init": "XGBRegressor(n_estimators=100, random_state=42, verbosity=0)",
    },
    "LightGBM Regressor": {
        "import": "from lightgbm import LGBMRegressor",
        "init": "LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)",
    },
    "CatBoost Regressor": {
        "import": "from catboost import CatBoostRegressor",
        "init": "CatBoostRegressor(iterations=100, random_state=42, verbose=0)",
    },
}

_BALANCE_IMPORT = {
    "smote": "from imblearn.over_sampling import SMOTE",
    "adasyn": "from imblearn.over_sampling import ADASYN",
    "undersample": "from imblearn.under_sampling import RandomUnderSampler",
    "smote_tomek": "from imblearn.combine import SMOTETomek",
}

_BALANCE_INIT = {
    "smote": "SMOTE(random_state=42)",
    "adasyn": "ADASYN(random_state=42)",
    "undersample": "RandomUnderSampler(random_state=42)",
    "smote_tomek": "SMOTETomek(random_state=42)",
}


def generate_ml_pipeline_code(
    target_col: str,
    feature_cols: List[str],
    task_type: str,          # "classification" or "regression"
    model_name: str,
    balance_strategy: str = "none",
    test_size: float = 0.2,
    file_format_hint: str = "csv",   # "csv", "excel", "sas7bdat"
) -> str:
    """
    根據使用者的建模設定，產生可直接在 Jupyter Notebook 執行的 Python 腳本。

    Returns:
        str: 完整 Python 程式碼字串。
    """
    model_info = _MODEL_MAP.get(model_name, None)
    if model_info is None:
        return f"# ⚠️ 無法識別模型名稱: {model_name}\n# 請手動撰寫模型初始化程式碼。"

    is_classification = (task_type == "classification")
    use_balancing = balance_strategy not in ("none", "class_weight")

    # ── 產生載入資料的提示 ──────────────────────────────────────────────
    if file_format_hint == "sas7bdat":
        load_snippet = (
            "df = pd.read_sas('your_data.sas7bdat')\n"
            "# 若欄位有 bytes 型態，可用以下程式碼解碼：\n"
            "# for col in df.select_dtypes(['object']).columns:\n"
            "#     df[col] = df[col].str.decode('utf-8', errors='ignore')"
        )
    elif file_format_hint == "excel":
        load_snippet = "df = pd.read_excel('your_data.xlsx')"
    else:
        load_snippet = "df = pd.read_csv('your_data.csv')"

    feat_repr = repr(feature_cols)
    lines = []

    # ── 1. 標題與說明 ────────────────────────────────────────────────────
    lines.append("# ============================================================")
    lines.append(f"# 自動生成的建模腳本")
    lines.append(f"# 任務類型  : {'分類 (Classification)' if is_classification else '迴歸 (Regression)'}")
    lines.append(f"# 模型      : {model_name}")
    lines.append(f"# 目標變數  : {target_col}")
    lines.append(f"# 特徵變數  : {feature_cols}")
    lines.append(f"# 不平衡策略: {balance_strategy}")
    lines.append(f"# 測試集比例: {test_size}")
    lines.append("# ============================================================")
    lines.append("")

    # ── 2. Import 區塊 ───────────────────────────────────────────────────
    lines.append("import pandas as pd")
    lines.append("import numpy as np")
    lines.append("import warnings")
    lines.append("warnings.filterwarnings('ignore')")
    lines.append("")
    lines.append("from sklearn.model_selection import train_test_split")
    lines.append("from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder")
    lines.append("from sklearn.compose import ColumnTransformer")
    if is_classification:
        lines.append("from sklearn.metrics import (")
        lines.append("    accuracy_score, precision_score, recall_score, f1_score,")
        lines.append("    classification_report, confusion_matrix, roc_auc_score")
        lines.append(")")
    else:
        lines.append("from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score")
    lines.append("")
    lines.append(model_info["import"])
    if use_balancing and balance_strategy in _BALANCE_IMPORT:
        lines.append(_BALANCE_IMPORT[balance_strategy])
    lines.append("")

    # ── 3. 載入資料 ──────────────────────────────────────────────────────
    lines.append("# ──────────────────────────────────────")
    lines.append("# 1. 載入資料")
    lines.append("# ──────────────────────────────────────")
    lines.append(load_snippet)
    lines.append("")

    # ── 4. 特徵與目標 ────────────────────────────────────────────────────
    lines.append("# ──────────────────────────────────────")
    lines.append("# 2. 設定特徵與目標變數")
    lines.append("# ──────────────────────────────────────")
    lines.append(f"TARGET = {repr(target_col)}")
    lines.append(f"FEATURES = {feat_repr}")
    lines.append("")
    lines.append("model_df = df[FEATURES + [TARGET]].dropna()")
    lines.append("X = model_df[FEATURES].copy()")
    lines.append("y = model_df[TARGET].copy()")
    lines.append("")

    # ── 5. 目標變數編碼 (分類) ────────────────────────────────────────────
    if is_classification:
        lines.append("# ──────────────────────────────────────")
        lines.append("# 3. 目標變數編碼（分類任務）")
        lines.append("# ──────────────────────────────────────")
        lines.append("le = LabelEncoder()")
        lines.append("if y.dtype == 'object' or str(y.dtype) == 'category':")
        lines.append("    y = le.fit_transform(y.astype(str))")
        lines.append("    print('目標類別對應:', dict(enumerate(le.classes_)))")
        lines.append("else:")
        lines.append("    y = y.astype(int)")
        lines.append("    le = None")
        lines.append("")

    # ── 6. 切割資料 ───────────────────────────────────────────────────────
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {'4' if is_classification else '3'}. 切割訓練集與測試集")
    lines.append("# ──────────────────────────────────────")
    if is_classification:
        lines.append(
            f"X_train, X_test, y_train, y_test = train_test_split("
            f"X, y, test_size={test_size}, random_state=42, stratify=y)"
        )
    else:
        lines.append(
            f"X_train, X_test, y_train, y_test = train_test_split("
            f"X, y, test_size={test_size}, random_state=42)"
        )
    lines.append("")

    # ── 7. ColumnTransformer 前處理 ───────────────────────────────────────
    step_num = 5 if is_classification else 4
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 特徵前處理（數值標準化 + 類別 One-Hot Encoding）")
    lines.append("# ──────────────────────────────────────")
    lines.append("cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()")
    lines.append("num_cols = X.select_dtypes(include=['number']).columns.tolist()")
    lines.append("")
    lines.append("preprocessor = ColumnTransformer(transformers=[")
    lines.append("    ('num', StandardScaler(), num_cols),")
    lines.append("    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),")
    lines.append("], remainder='passthrough')")
    lines.append("")
    lines.append("X_train = preprocessor.fit_transform(X_train)")
    lines.append("X_test  = preprocessor.transform(X_test)")
    lines.append("")

    # ── 8. 不平衡處理（可選）───────────────────────────────────────────────
    if use_balancing and balance_strategy in _BALANCE_INIT:
        step_num += 1
        lines.append("# ──────────────────────────────────────")
        lines.append(f"# {step_num}. 不平衡數據處理（{balance_strategy}）")
        lines.append("# ──────────────────────────────────────")
        lines.append(f"sampler = {_BALANCE_INIT[balance_strategy]}")
        lines.append("X_train, y_train = sampler.fit_resample(X_train, y_train)")
        lines.append(f"print(f'平衡後訓練集大小: {{len(y_train)}}')")
        lines.append("")

    # ── 9. class_weight 策略 ──────────────────────────────────────────────
    # 僅對支援 class_weight 的模型注入參數
    _SUPPORTS_CLASS_WEIGHT = {
        "Logistic Regression (邏輯迴歸)", "Decision Tree (決策樹)",
        "Random Forest (隨機森林)", "SVM (支援向量機)", "XGBoost", "LightGBM",
    }
    if balance_strategy == "class_weight" and is_classification and model_name in _SUPPORTS_CLASS_WEIGHT:
        lines.append("# 注意：class_weight='balanced' 已在下方模型初始化中設定")
        base_init = model_info["init"]
        if "class_weight" not in base_init:
            idx = base_init.rfind(")")
            prefix = base_init[:idx]
            if prefix.endswith("("):
                model_init = prefix + "class_weight='balanced')"
            else:
                model_init = prefix + ", class_weight='balanced')"
        else:
            model_init = base_init
    else:
        model_init = model_info["init"]

    # ── 10. 模型訓練 ───────────────────────────────────────────────────────
    step_num += 1
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 模型建立與訓練")
    lines.append("# ──────────────────────────────────────")
    lines.append(f"model = {model_init}")
    lines.append("model.fit(X_train, y_train)")
    lines.append("")

    # ── 11. 模型評估 ───────────────────────────────────────────────────────
    step_num += 1
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 模型評估")
    lines.append("# ──────────────────────────────────────")
    lines.append("y_pred = model.predict(X_test)")
    lines.append("")
    if is_classification:
        lines.append("target_names = le.classes_.tolist() if le is not None else None")
        lines.append("print('Accuracy :', accuracy_score(y_test, y_pred))")
        lines.append("print('F1 Score :', f1_score(y_test, y_pred, average='weighted', zero_division=0))")
        lines.append("print()")
        lines.append("print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))")
        lines.append("")
        lines.append("# ROC AUC（僅二分類）")
        lines.append("if len(set(y_test)) == 2 and hasattr(model, 'predict_proba'):")
        lines.append("    y_prob = model.predict_proba(X_test)[:, 1]")
        lines.append("    print('ROC AUC  :', roc_auc_score(y_test, y_prob))")
    else:
        lines.append("mse = mean_squared_error(y_test, y_pred)")
        lines.append("print('R²   :', r2_score(y_test, y_pred))")
        lines.append("print('RMSE :', mse ** 0.5)")
        lines.append("print('MAE  :', mean_absolute_error(y_test, y_pred))")
    lines.append("")

    # ── 12. 特徵重要性（可選）────────────────────────────────────────────
    step_num += 1
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 特徵重要性（樹模型限定）")
    lines.append("# ──────────────────────────────────────")
    lines.append("if hasattr(model, 'feature_importances_'):")
    lines.append("    feat_names = preprocessor.get_feature_names_out()")
    lines.append("    feat_names = [f.split('__', 1)[1] if '__' in f else f for f in feat_names]")
    lines.append("    importance_df = pd.DataFrame({")
    lines.append("        'feature': feat_names,")
    lines.append("        'importance': model.feature_importances_")
    lines.append("    }).sort_values('importance', ascending=False)")
    lines.append("    print(importance_df.head(20).to_string(index=False))")
    lines.append("")

    return "\n".join(lines)


# ── 模型名稱 → R code 的映射表 (Caret 方法) ─────────────────────────────
_R_MODEL_MAP = {
    "Logistic Regression (邏輯迴歸)": {
        "pkg": "stats",
        "method": "'glm', family = 'binomial'",
    },
    "Decision Tree (決策樹)": {
        "pkg": "rpart",
        "method": "'rpart'",
    },
    "Random Forest (隨機森林)": {
        "pkg": "randomForest",
        "method": "'rf'",
    },
    "SVM (支援向量機)": {
        "pkg": "kernlab",
        "method": "'svmRadial'",
    },
    "KNN (K-近鄰)": {
        "pkg": "kknn",
        "method": "'kknn'",
    },
    "Naive Bayes (樸素貝葉斯)": {
        "pkg": "naivebayes",
        "method": "'naive_bayes'",
    },
    "XGBoost": {
        "pkg": "xgboost",
        "method": "'xgbTree'",
    },
    "LightGBM": {
        "pkg": "lightgbm",
        "method": "'LightGBM'",
    },
    "CatBoost": {
        "pkg": "catboost",
        "method": "'catboost'",
    },
    "LDA (線性判別分析)": {
        "pkg": "MASS",
        "method": "'lda'",
    },
    "Linear Regression (線性迴歸)": {
        "pkg": "stats",
        "method": "'lm'",
    },
    "Ridge Regression (嶺迴歸)": {
        "pkg": "glmnet",
        "method": "'ridge'",
    },
    "Lasso Regression": {
        "pkg": "glmnet",
        "method": "'lasso'",
    },
    "Decision Tree Regressor (決策樹迴歸)": {
        "pkg": "rpart",
        "method": "'rpart'",
    },
    "Random Forest Regressor (隨機森林迴歸)": {
        "pkg": "randomForest",
        "method": "'rf'",
    },
    "Gradient Boosting Regressor (梯度提升迴歸)": {
        "pkg": "gbm",
        "method": "'gbm'",
    },
    "XGBoost Regressor": {
        "pkg": "xgboost",
        "method": "'xgbTree'",
    },
    "LightGBM Regressor": {
        "pkg": "lightgbm",
        "method": "'LightGBM'",
    },
    "CatBoost Regressor": {
        "pkg": "catboost",
        "method": "'catboost'",
    },
}

def generate_r_pipeline_code(
    target_col: str,
    feature_cols: List[str],
    task_type: str,          # "classification" or "regression"
    model_name: str,
    balance_strategy: str = "none",
    test_size: float = 0.2,
    file_format_hint: str = "csv",
) -> str:
    """
    根據使用者的建模設定，產生可直接運行的 R 腳本。

    Returns:
        str: 完整 R 程式碼字串。
    """
    model_info = _R_MODEL_MAP.get(model_name, None)
    if model_info is None:
        return f"# ⚠️ 無法識別模型名稱: {model_name}\n# 請手動撰寫模型初始化程式碼。"

    is_classification = (task_type == "classification")
    use_balancing = balance_strategy not in ("none", "class_weight")

    # ── 產生載入資料的提示 ──────────────────────────────────────────────
    if file_format_hint == "sas7bdat":
        load_snippet = "df <- haven::read_sas('your_data.sas7bdat')"
    elif file_format_hint == "excel":
        load_snippet = "df <- readxl::read_excel('your_data.xlsx')"
    else:
        load_snippet = "df <- read.csv('your_data.csv')"

    lines = []

    # ── 1. 標題與說明 ────────────────────────────────────────────────────
    lines.append("# ============================================================")
    lines.append(f"# 自動生成的 R 建模腳本")
    lines.append(f"# 任務類型  : {'分類 (Classification)' if is_classification else '迴歸 (Regression)'}")
    lines.append(f"# 模型      : {model_name}")
    lines.append(f"# 測試集比例: {test_size}")
    lines.append("# ============================================================")
    lines.append("")

    # ── 2. Library 區塊 ───────────────────────────────────────────────────
    lines.append("if (!require(caret)) install.packages('caret')")
    lines.append("if (!require(dplyr)) install.packages('dplyr')")
    if file_format_hint == "sas7bdat":
        lines.append("if (!require(haven)) install.packages('haven')")
    elif file_format_hint == "excel":
        lines.append("if (!require(readxl)) install.packages('readxl')")

    if model_info["pkg"] not in ("stats",):
        lines.append(f"if (!require({model_info['pkg']})) install.packages('{model_info['pkg']}')")

    if use_balancing:
        lines.append("if (!require(DMwR)) install.packages('DMwR') # For SMOTE")
        lines.append("if (!require(ROSE)) install.packages('ROSE')")
        
    lines.append("")
    lines.append("library(caret)")
    lines.append("library(dplyr)")
    if model_info["pkg"] not in ("stats",):
        lines.append(f"library({model_info['pkg']})")
    lines.append("")

    # ── 3. 載入資料 ──────────────────────────────────────────────────────
    lines.append("# ──────────────────────────────────────")
    lines.append("# 1. 載入資料")
    lines.append("# ──────────────────────────────────────")
    lines.append(load_snippet)
    lines.append("")

    # ── 4. 特徵與目標 ────────────────────────────────────────────────────
    lines.append("# ──────────────────────────────────────")
    lines.append("# 2. 設定特徵與清理缺失值")
    lines.append("# ──────────────────────────────────────")
    
    # R formula constructor
    formula_str = f"`{target_col}` ~ " + " + ".join([f"`{f}`" for f in feature_cols])
    
    lines.append(f"target_col <- '{target_col}'")
    
    f_str = 'c(' + ', '.join([f"'{f}'" for f in feature_cols]) + ')'
    lines.append(f"feature_cols <- {f_str}")
    lines.append("")
    
    lines.append("model_df <- df %>%")
    lines.append("  select(all_of(c(feature_cols, target_col))) %>%")
    lines.append("  na.omit()")
    lines.append("")

    # ── 5. 目標變數編碼 (分類) ────────────────────────────────────────────
    if is_classification:
        lines.append("# ──────────────────────────────────────")
        lines.append("# 3. 目標變數轉為 Factor（分類任務）")
        lines.append("# ──────────────────────────────────────")
        lines.append(f"model_df[[target_col]] <- as.factor(make.names(as.character(model_df[[target_col]])))")
        lines.append("")
    else:
        lines.append(f"model_df[[target_col]] <- as.numeric(model_df[[target_col]])")
        lines.append("")

    # ── 6. 切割資料 ───────────────────────────────────────────────────────
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {'4' if is_classification else '3'}. 切割訓練集與測試集")
    lines.append("# ──────────────────────────────────────")
    lines.append("set.seed(42)")
    lines.append(f"trainIndex <- createDataPartition(model_df[[target_col]], p = {1.0 - test_size}, list = FALSE)")
    lines.append("train_data <- model_df[trainIndex, ]")
    lines.append("test_data  <- model_df[-trainIndex, ]")
    lines.append("")

    # ── 7. caret 訓練設定 ───────────────────────────────────────
    step_num = 5 if is_classification else 4
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 模型訓練設定與前處理")
    lines.append("# ──────────────────────────────────────")
    
    if is_classification and use_balancing:
        samp_arg = ""
        if balance_strategy == "smote":
            samp_arg = ", sampling = 'smote'"
        elif balance_strategy == "undersample":
            samp_arg = ", sampling = 'down'"
        lines.append(f"train_control <- trainControl(method = 'cv', number = 5, classProbs = TRUE{samp_arg})")
    elif is_classification:
        lines.append("train_control <- trainControl(method = 'cv', number = 5, classProbs = TRUE)")
    else:
        lines.append("train_control <- trainControl(method = 'cv', number = 5)")
        
    lines.append("")
    
    # ── 8. 模型訓練 ───────────────────────────────────────────────────────
    step_num += 1
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 模型訓練 (caret)")
    lines.append("# ──────────────────────────────────────")
    lines.append("set.seed(42)")
    
    metric = "ROC" if is_classification else "RMSE"
    # Note: Using formula directly
    fit_call = (
        f"model_fit <- train(\n"
        f"  {formula_str},\n"
        f"  data = train_data,\n"
        f"  method = {model_info['method']},\n"
        f"  trControl = train_control,\n"
        f"  preProcess = c('center', 'scale')\n" 
        f")"
    )
    lines.append(fit_call)
    lines.append("print(model_fit)")
    lines.append("")

    # ── 9. 模型評估 ───────────────────────────────────────────────────────
    step_num += 1
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 模型評估")
    lines.append("# ──────────────────────────────────────")
    lines.append("pred <- predict(model_fit, newdata = test_data)")
    lines.append("")
    
    if is_classification:
        lines.append("cm <- confusionMatrix(pred, test_data[[target_col]])")
        lines.append("print(cm)")
        lines.append("")
        lines.append("# 預測機率與 ROC AUC")
        lines.append("if (!require(pROC)) install.packages('pROC')")
        lines.append("library(pROC)")
        lines.append("pred_prob <- predict(model_fit, newdata = test_data, type = 'prob')")
        lines.append("roc_obj <- roc(test_data[[target_col]], pred_prob[, 2])")
        lines.append("print(paste('ROC AUC:', round(auc(roc_obj), 4)))")
    else:
        lines.append("results <- postResample(pred = pred, obs = test_data[[target_col]])")
        lines.append("print(results)")
        
    lines.append("")

    # ── 10. 特徵重要性 (可選) ───────────
    step_num += 1
    lines.append("# ──────────────────────────────────────")
    lines.append(f"# {step_num}. 特徵重要性")
    lines.append("# ──────────────────────────────────────")
    lines.append("imp <- varImp(model_fit)")
    lines.append("print(imp)")
    lines.append("plot(imp, top = 20)")
    lines.append("")

    return "\n".join(lines)

