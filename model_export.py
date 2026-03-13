"""
模型匯出模組 — 使用 joblib 打包模型 + 元數據。
"""

import io
import datetime
import sys
import joblib
import sklearn


def export_model(model, scaler, label_encoder, feature_names: list,
                 metrics: dict, model_name: str, task_type: str) -> io.BytesIO:
    """
    將訓練好的模型打包為可下載的 bytes buffer。

    包含: model, scaler, label_encoder, feature_names, metrics, metadata
    """
    package = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "metrics": metrics,
        "model_name": model_name,
        "task_type": task_type,
        "exported_at": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
    }

    buf = io.BytesIO()
    joblib.dump(package, buf)
    buf.seek(0)
    return buf


def load_model(file_bytes: bytes) -> dict:
    """載入模型包。返回包含所有元數據的 dict。"""
    return joblib.load(io.BytesIO(file_bytes))
