"""
機器學習模型模組
支援分類與迴歸任務，包含 XGBoost/LightGBM/CatBoost/LDA、
超參數調整 (GridSearchCV / Optuna)、特徵重要性提取。
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config

# --- 可選套件：不平衡數據處理 ---
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# --- 可選套件：XGBoost ---
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- 可選套件：LightGBM ---
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# --- 可選套件：CatBoost ---
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# --- 可選套件：Optuna ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# --- sklearn 內建：LDA ---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# --- 不平衡數據處理策略 ---
BALANCE_STRATEGIES = {
    "none": "不處理",
    "class_weight": "類別權重平衡 (class_weight='balanced')",
    "smote": "SMOTE 過採樣",
    "adasyn": "ADASYN 自適應過採樣",
    "undersample": "隨機欠採樣",
    "smote_tomek": "SMOTETomek 混合採樣",
}


def get_balanced_models(strategy="none"):
    """根據不平衡處理策略返回對應的分類模型清單。"""
    use_balanced = strategy == "class_weight"
    models = {
        "Logistic Regression (邏輯迴歸)": LogisticRegression(
            max_iter=1000,
            class_weight='balanced' if use_balanced else None,
            random_state=config.DEFAULT_RANDOM_STATE
        ),
        "Decision Tree (決策樹)": DecisionTreeClassifier(
            class_weight='balanced' if use_balanced else None,
            random_state=config.DEFAULT_RANDOM_STATE
        ),
        "Random Forest (隨機森林)": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced' if use_balanced else None,
            random_state=config.DEFAULT_RANDOM_STATE
        ),
        "SVM (支援向量機)": SVC(
            probability=True,
            class_weight='balanced' if use_balanced else None,
            random_state=config.DEFAULT_RANDOM_STATE
        ),
        "KNN (K-近鄰)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (樸素貝葉斯)": GaussianNB(),
        "LDA (線性判別分析)": LinearDiscriminantAnalysis(),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            eval_metric='logloss',
            random_state=config.DEFAULT_RANDOM_STATE,
            verbosity=0,
        )
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            verbose=-1,
        )
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            verbose=0,
        )
    return models


# --- 預設可用分類模型清單（延遲初始化）---
_AVAILABLE_MODELS = None

def _get_available_models():
    global _AVAILABLE_MODELS
    if _AVAILABLE_MODELS is None:
        _AVAILABLE_MODELS = get_balanced_models("none")
    return _AVAILABLE_MODELS

# 保持向後相容的屬性存取
class _LazyModels:
    """延遲初始化模型清單，避免 import 時即建立所有模型實例。"""
    def __getattr__(self, name):
        return getattr(_get_available_models(), name)
    def __getitem__(self, key):
        return _get_available_models()[key]
    def __contains__(self, key):
        return key in _get_available_models()
    def __iter__(self):
        return iter(_get_available_models())
    def keys(self):
        return _get_available_models().keys()
    def values(self):
        return _get_available_models().values()
    def items(self):
        return _get_available_models().items()
    def __len__(self):
        return len(_get_available_models())

AVAILABLE_MODELS = _LazyModels()


def get_regression_models():
    """返回可用的迴歸模型清單（含 XGBoost/LightGBM 若已安裝）。"""
    models = {
        "Linear Regression (線性迴歸)": LinearRegression(),
        "Ridge Regression (嶺迴歸)": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=5000),
        "Decision Tree Regressor (決策樹迴歸)": DecisionTreeRegressor(
            random_state=config.DEFAULT_RANDOM_STATE
        ),
        "Random Forest Regressor (隨機森林迴歸)": RandomForestRegressor(
            n_estimators=100, random_state=config.DEFAULT_RANDOM_STATE
        ),
        "Gradient Boosting Regressor (梯度提升迴歸)": GradientBoostingRegressor(
            n_estimators=100, random_state=config.DEFAULT_RANDOM_STATE
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            verbosity=0,
        )
    if LIGHTGBM_AVAILABLE:
        models["LightGBM Regressor"] = LGBMRegressor(
            n_estimators=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            verbose=-1,
        )
    if CATBOOST_AVAILABLE:
        models["CatBoost Regressor"] = CatBoostRegressor(
            iterations=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            verbose=0,
        )
    return models


# --- 超參數搜索空間 ---
PARAM_GRIDS = {
    "Logistic Regression (邏輯迴歸)": {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
    },
    "Decision Tree (決策樹)": {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy'],
    },
    "Random Forest (隨機森林)": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    },
    "SVM (支援向量機)": {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
    },
    "KNN (K-近鄰)": {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance'],
    },
    "XGBoost": {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
    },
    "LightGBM": {
        'n_estimators': [50, 100],
        'num_leaves': [31, 50],
        'learning_rate': [0.01, 0.1],
    },
    "CatBoost": {
        'iterations': [50, 100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.3],
    },
    "LDA (線性判別分析)": {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto'],
    },
}


def apply_balancing(X_train, y_train, strategy="none"):
    """
    對訓練數據應用不平衡處理。
    返回平衡後的 X_train, y_train。
    """
    if strategy in ("none", "class_weight"):
        return X_train, y_train

    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "需要安裝 imbalanced-learn 套件才能使用採樣策略。\n"
            "請執行: pip install imbalanced-learn"
        )

    if strategy == "smote":
        sampler = SMOTE(random_state=config.DEFAULT_RANDOM_STATE)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=config.DEFAULT_RANDOM_STATE)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=config.DEFAULT_RANDOM_STATE)
    elif strategy == "smote_tomek":
        sampler = SMOTETomek(random_state=config.DEFAULT_RANDOM_STATE)
    else:
        return X_train, y_train

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_class_distribution(y):
    """獲取類別分佈統計。"""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    dist = {}
    for cls, cnt in zip(unique, counts):
        dist[str(cls)] = {'count': int(cnt), 'ratio': round(cnt / total, 4)}
    return dist


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def prepare_data(df, target_column, feature_columns, test_size=None, task_type="classification"):
    """
    準備機器學習數據：類別特徵 One-Hot Encoding、數值特徵標準化、訓練/測試分割。
    解決資料洩漏：在切割資料後才 fit transform。
    task_type: "classification" 或 "regression"
    返回 X_train_transformed, X_test_transformed, y_train, y_test, label_encoder, preprocessor
    """
    if test_size is None:
        test_size = config.DEFAULT_TEST_SIZE

    model_df = df[feature_columns + [target_column]].dropna()
    if len(model_df) < 10:
        raise ValueError("有效數據量不足（需至少 10 筆），請檢查缺失值。")

    X = model_df[feature_columns].copy()
    y = model_df[target_column].copy()

    # 目標變數編碼 (僅分類任務)
    label_encoder = None
    if task_type == config.ML_TASK_CLASSIFICATION:
        if y.dtype == 'object' or str(y.dtype) == 'category':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))
        else:
            # 確保為整數
            y = y.astype(int)

    # 分類任務用 stratify，迴歸任務不用
    stratify_arg = y if task_type == config.ML_TASK_CLASSIFICATION else None
    
    # ❗️ 解決資料洩漏：先切割資料，再做特徵工程
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=config.DEFAULT_RANDOM_STATE,
        stratify=stratify_arg
    )

    # 區分類別特徵與數值特徵
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    # 建立前處理管線 (Pipeline)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )

    # 在 training_data 上 fit_transform，在 test_data 上 transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # 確保轉換後為 pandas DataFrame，保留欄位名稱 (對樹模型或後續呈現很重要)
    # 取得 Transformer 產生的新特徵名稱
    feature_names = preprocessor.get_feature_names_out()
    # 移除 'num__' 或 'cat__' 前綴
    clean_feature_names = [f.split('__', 1)[1] if '__' in f else f for f in feature_names]
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=clean_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=clean_feature_names, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, label_encoder, preprocessor


def train_single_model(model_name, X_train, y_train, X_test, y_test, label_encoder=None):
    """
    訓練單個分類模型並返回評估結果。
    返回 dict: model, accuracy, precision, recall, f1, confusion_matrix, classification_report, y_pred
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"不支援的模型: {model_name}")

    from sklearn.base import clone
    model = clone(AVAILABLE_MODELS[model_name])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    unique_classes = np.unique(y_test)
    average = 'binary' if len(unique_classes) == 2 else 'weighted'

    target_names = None
    if label_encoder is not None:
        target_names = label_encoder.classes_.tolist()

    results = {
        'model_name': model_name,
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_test, y_pred, average=average, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=target_names,
            zero_division=0
        ),
    }

    # ROC/AUC (僅二分類)
    if len(unique_classes) == 2 and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results['roc_auc'] = auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
    else:
        results['roc_auc'] = None
        results['fpr'] = None
        results['tpr'] = None

    return results


def train_regression_model(model_name, X_train, y_train, X_test, y_test):
    """
    訓練單個迴歸模型並返回評估結果。
    返回 dict: model, mse, rmse, mae, r2, y_pred, y_test
    """
    regression_models = get_regression_models()
    if model_name not in regression_models:
        raise ValueError(f"不支援的迴歸模型: {model_name}")

    from sklearn.base import clone
    model = clone(regression_models[model_name])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return {
        'model_name': model_name,
        'model': model,
        'y_pred': y_pred,
        'y_test': y_test,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }


def compare_regression_models(X_train, y_train, X_test, y_test, model_names=None):
    """
    比較多個迴歸模型的效能。
    返回 (DataFrame, all_results_dict)
    """
    regression_models = get_regression_models()
    if model_names is None:
        model_names = list(regression_models.keys())

    results_list = []
    all_results = {}
    for name in model_names:
        try:
            res = train_regression_model(name, X_train, y_train, X_test, y_test)
            results_list.append({
                '模型': name,
                'R²': round(res['r2'], 4),
                'RMSE': round(res['rmse'], 4),
                'MAE': round(res['mae'], 4),
                'MSE': round(res['mse'], 4),
            })
            all_results[name] = res
        except Exception as e:
            results_list.append({
                '模型': name,
                'R²': f'錯誤: {e}',
                'RMSE': '-', 'MAE': '-', 'MSE': '-',
            })

    return pd.DataFrame(results_list), all_results


def kfold_cross_validation(model_name, X, y, n_folds=None, task_type="classification",
                           balance_strategy="none", preprocessor=None):
    """
    執行 K-Fold 交叉驗證。
    解決採樣洩漏問題：使用 imblearn.pipeline 將採樣步驟與模型綁定。
    preprocessor: 若提供 ColumnTransformer，將包進 Pipeline 以避免數據洩漏。
    task_type: "classification" 用 StratifiedKFold + accuracy；"regression" 用 KFold + R²
    返回 dict: scores, mean, std
    """
    if n_folds is None:
        n_folds = config.DEFAULT_CV_FOLDS

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    if task_type == config.ML_TASK_CLASSIFICATION:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"不支援的分類模型: {model_name}")
        base_model = clone(AVAILABLE_MODELS[model_name])
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.DEFAULT_RANDOM_STATE)
        scoring = 'accuracy'

        # 建立附帶採樣策略的 Pipeline
        if balance_strategy not in ("none", "class_weight"):
            if not IMBLEARN_AVAILABLE:
                raise ImportError("需要安裝 imbalanced-learn 套件才能使用採樣策略進行交叉驗證。")

            from imblearn.pipeline import Pipeline as ImbPipeline

            if balance_strategy == "smote":
                sampler = SMOTE(random_state=config.DEFAULT_RANDOM_STATE)
            elif balance_strategy == "adasyn":
                sampler = ADASYN(random_state=config.DEFAULT_RANDOM_STATE)
            elif balance_strategy == "undersample":
                sampler = RandomUnderSampler(random_state=config.DEFAULT_RANDOM_STATE)
            elif balance_strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=config.DEFAULT_RANDOM_STATE)
            else:
                sampler = None

            if sampler is not None:
                steps = []
                if preprocessor is not None:
                    steps.append(('preprocessor', clone(preprocessor)))
                steps.append(('sampler', sampler))
                steps.append(('model', base_model))
                model = ImbPipeline(steps)
            else:
                if preprocessor is not None:
                    model = Pipeline([('preprocessor', clone(preprocessor)), ('model', base_model)])
                else:
                    model = base_model
        else:
            if preprocessor is not None:
                model = Pipeline([('preprocessor', clone(preprocessor)), ('model', base_model)])
            else:
                model = base_model

    else:
        regression_models = get_regression_models()
        if model_name not in regression_models:
            raise ValueError(f"不支援的迴歸模型: {model_name}")
        base_model = clone(regression_models[model_name])
        if preprocessor is not None:
            model = Pipeline([('preprocessor', clone(preprocessor)), ('model', base_model)])
        else:
            model = base_model
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=config.DEFAULT_RANDOM_STATE)
        scoring = 'r2'

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return {
        'model_name': model_name,
        'n_folds': n_folds,
        'scoring': scoring,
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
    }


def compare_models(X_train, y_train, X_test, y_test, model_names=None, label_encoder=None):
    """
    比較多個分類模型的效能。
    返回 DataFrame: model, accuracy, precision, recall, f1, roc_auc
    """
    if model_names is None:
        model_names = list(AVAILABLE_MODELS.keys())

    results_list = []
    all_results = {}
    for name in model_names:
        try:
            res = train_single_model(name, X_train, y_train, X_test, y_test, label_encoder)
            results_list.append({
                '模型': name,
                'Accuracy': round(res['accuracy'], 4),
                'Precision': round(res['precision'], 4),
                'Recall': round(res['recall'], 4),
                'F1 Score': round(res['f1'], 4),
                'ROC AUC': round(res['roc_auc'], 4) if res['roc_auc'] is not None else 'N/A',
            })
            all_results[name] = res
        except Exception as e:
            results_list.append({
                '模型': name,
                'Accuracy': f'錯誤: {e}',
                'Precision': '-', 'Recall': '-', 'F1 Score': '-', 'ROC AUC': '-',
            })

    comparison_df = pd.DataFrame(results_list)
    return comparison_df, all_results


def get_feature_importance(model, feature_names):
    """
    從訓練好的模型提取特徵重要性。
    支援：樹模型 (feature_importances_)、線性模型 (coef_)。
    返回 (feature_names_list, importances_array)，不支援時返回 (None, None)。
    """
    if hasattr(model, 'feature_importances_'):
        return list(feature_names), model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        return list(feature_names), coef
    return None, None


def tune_hyperparameters(model_name, X_train, y_train, cv=None, scoring='accuracy'):
    """
    使用 GridSearchCV 執行超參數調整。
    返回 dict: best_params, best_score, cv_results_df, best_estimator
    """
    if cv is None:
        cv = config.DEFAULT_GRID_SEARCH_CV

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"不支援的模型: {model_name}")

    # 模型 key 可能含中文括號，需對應到 PARAM_GRIDS
    # 嘗試完整名稱，找不到則嘗試英文前綴
    param_grid = None
    if model_name in PARAM_GRIDS:
        param_grid = PARAM_GRIDS[model_name]
    else:
        for key in PARAM_GRIDS:
            if model_name.startswith(key) or key.startswith(model_name.split(" ")[0]):
                param_grid = PARAM_GRIDS[key]
                break

    if param_grid is None:
        raise ValueError(f"模型 '{model_name}' 尚未定義超參數搜索空間。")

    from sklearn.base import clone
    base_model = clone(AVAILABLE_MODELS[model_name])

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    return {
        'model_name': model_name,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
        'best_estimator': grid_search.best_estimator_,
    }


# --- Optuna 超參數搜索空間 ---
OPTUNA_PARAM_SPACES = {
    "Logistic Regression (邏輯迴歸)": {
        'C': ('log_float', 0.001, 100.0),
        'solver': ('categorical', ['lbfgs', 'liblinear']),
    },
    "Decision Tree (決策樹)": {
        'max_depth': ('int', 2, 30),
        'min_samples_split': ('int', 2, 20),
        'criterion': ('categorical', ['gini', 'entropy']),
    },
    "Random Forest (隨機森林)": {
        'n_estimators': ('int', 50, 300),
        'max_depth': ('int', 3, 20),
        'min_samples_split': ('int', 2, 10),
    },
    "SVM (支援向量機)": {
        'C': ('log_float', 0.01, 100.0),
        'kernel': ('categorical', ['rbf', 'linear']),
        'gamma': ('categorical', ['scale', 'auto']),
    },
    "KNN (K-近鄰)": {
        'n_neighbors': ('int', 1, 21),
        'weights': ('categorical', ['uniform', 'distance']),
    },
    "XGBoost": {
        'n_estimators': ('int', 50, 300),
        'max_depth': ('int', 3, 10),
        'learning_rate': ('log_float', 0.005, 0.5),
        'subsample': ('float', 0.6, 1.0),
        'colsample_bytree': ('float', 0.6, 1.0),
    },
    "LightGBM": {
        'n_estimators': ('int', 50, 300),
        'num_leaves': ('int', 20, 80),
        'learning_rate': ('log_float', 0.005, 0.5),
        'subsample': ('float', 0.6, 1.0),
    },
    "CatBoost": {
        'iterations': ('int', 50, 300),
        'depth': ('int', 3, 10),
        'learning_rate': ('log_float', 0.005, 0.5),
    },
    "LDA (線性判別分析)": {
        'solver': ('categorical', ['svd', 'lsqr', 'eigen']),
    },
}


def tune_with_optuna(model_name, X_train, y_train, n_trials=50,
                     cv=None, scoring='accuracy', task_type="classification"):
    """
    使用 Optuna 貝葉斯超參數優化。
    比 GridSearchCV 更高效，尤其在高維搜索空間中。
    返回 dict: best_params, best_score, study, best_estimator
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("需要安裝 optuna 套件。請執行: pip install optuna")
    if cv is None:
        cv = config.DEFAULT_GRID_SEARCH_CV

    is_classification = (task_type == config.ML_TASK_CLASSIFICATION)

    # 取得搜索空間
    param_space = OPTUNA_PARAM_SPACES.get(model_name)
    if param_space is None:
        raise ValueError(f"模型 '{model_name}' 尚未定義 Optuna 搜索空間。")

    # 取得基礎模型
    if is_classification:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"不支援的分類模型: {model_name}")
    else:
        reg_models = get_regression_models()
        if model_name not in reg_models:
            raise ValueError(f"不支援的迴歸模型: {model_name}")

    from sklearn.base import clone

    def _suggest_params(trial):
        params = {}
        for name, spec in param_space.items():
            if spec[0] == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec[0] == 'float':
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif spec[0] == 'log_float':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif spec[0] == 'categorical':
                params[name] = trial.suggest_categorical(name, spec[1])
        return params

    def objective(trial):
        params = _suggest_params(trial)

        if is_classification:
            base = clone(AVAILABLE_MODELS[model_name])
        else:
            base = clone(get_regression_models()[model_name])

        # LDA shrinkage 只能在 lsqr/eigen solver 下使用
        if model_name == "LDA (線性判別分析)":
            if params.get('solver') == 'svd':
                params.pop('shrinkage', None)

        base.set_params(**params)

        if is_classification:
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True,
                                     random_state=config.DEFAULT_RANDOM_STATE)
        else:
            cv_obj = KFold(n_splits=cv, shuffle=True,
                           random_state=config.DEFAULT_RANDOM_STATE)

        scores = cross_val_score(base, X_train, y_train, cv=cv_obj, scoring=scoring)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # 用最佳參數重新訓練
    best_params = study.best_params
    if is_classification:
        best_model = clone(AVAILABLE_MODELS[model_name])
    else:
        best_model = clone(get_regression_models()[model_name])

    if model_name == "LDA (線性判別分析)" and best_params.get('solver') == 'svd':
        best_params.pop('shrinkage', None)

    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    return {
        'model_name': model_name,
        'best_params': best_params,
        'best_score': study.best_value,
        'study': study,
        'best_estimator': best_model,
        'n_trials': n_trials,
    }


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_diabetes
    import warnings
    warnings.filterwarnings('ignore')

    print("=== 分類任務測試 (Iris) ===")
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X_train, X_test, y_train, y_test, le, preprocessor = prepare_data(
        df, 'target', data.feature_names, task_type="classification"
    )
    res = train_single_model("Random Forest (隨機森林)", X_train, y_train, X_test, y_test)
    print(f"Random Forest Accuracy: {res['accuracy']:.4f}")

    if XGBOOST_AVAILABLE:
        res_xgb = train_single_model("XGBoost", X_train, y_train, X_test, y_test)
        print(f"XGBoost Accuracy: {res_xgb['accuracy']:.4f}")

    print("\n=== 迴歸任務測試 (Diabetes) ===")
    data2 = load_diabetes()
    df2 = pd.DataFrame(data2.data, columns=data2.feature_names)
    df2['target'] = data2.target

    X_train2, X_test2, y_train2, y_test2, le2, preprocessor2 = prepare_data(
        df2, 'target', data2.feature_names, task_type="regression"
    )
    res2 = train_regression_model("Random Forest Regressor (隨機森林迴歸)", X_train2, y_train2, X_test2, y_test2)
    print(f"Random Forest R2: {res2['r2']:.4f}, RMSE: {res2['rmse']:.4f}")

    print("\n=== 特徵重要性測試 ===")
    names, importances = get_feature_importance(res['model'], X_train.columns.tolist())
    if names:
        for n, imp in sorted(zip(names, importances), key=lambda x: -x[1]):
            print(f"  {n}: {imp:.4f}")
