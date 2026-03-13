import os
import sys

# Ensure the project root is on the Python path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any
import tempfile
import pandas as pd

from data_loader import load_data

router = APIRouter()

# Simple dict to act as a session store for our dataframes
SESSION_STORE: Dict[str, pd.DataFrame] = {}

@router.post("/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Uploads a dataset and saves it to the session store."""
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        df = load_data(tmp_path)
        SESSION_STORE[session_id] = df

        preview = df.head(10).fillna("").to_dict(orient="records")
        columns = [{"field": col, "type": str(df[col].dtype)} for col in df.columns]

        return {
            "status": "success",
            "message": f"Successfully loaded {file.filename}",
            "rows": len(df),
            "columns_count": len(df.columns),
            "columns": columns,
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.get("/summary/{session_id}")
async def get_summary(session_id: str):
    """Gets descriptive statistics for the uploaded dataset."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload data.")

    df = SESSION_STORE[session_id]

    try:
        desc = df.describe(include='all').fillna("").to_dict()
        missing = df.isnull().sum().to_dict()
        types = df.dtypes.astype(str).to_dict()

        summary = {
            col: {
                "type": types[col],
                "missing": missing[col],
                "stats": desc.get(col, {})
            }
            for col in df.columns
        }

        return {"status": "success", "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
