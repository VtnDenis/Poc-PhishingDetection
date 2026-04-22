from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import re
import uuid
import threading
import os
from datetime import datetime

# Numeric feature columns used during training
NUMERIC_COLUMNS = ['sender_length', 'is_free_email']

class EmailIn(BaseModel):
    body: str
    sender: str = ""
    model: str = ""

app = FastAPI(title="PhishingDetector")

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("favicon.ico")

@app.get("/automl", include_in_schema=False)
def automl():
    return FileResponse("static/automl.html")

# Chemin des modèles
MODEL_PATH = "models/xgb_model.joblib"
VECT_PATH = "models/tfidf.joblib"

tfidf = joblib.load(VECT_PATH)
model = joblib.load(MODEL_PATH)


def feature_engineering(df):
    df_features = df.copy()
    
    if 'body' in df_features.columns:
        df_features['body'] = df_features['body'].fillna('')
    
    
    def extract_email(text):
        if pd.isna(text): return ""
        match = re.search(r'<([^>]+)>', str(text))
        if match: return match.group(1)
        return str(text)
        
    df_features['sender_clean'] = df_features['sender'].apply(extract_email)
    df_features['sender_domain'] = df_features['sender_clean'].apply(lambda x: x.split('@')[-1] if '@' in x else "unknown")
    df_features['sender_length'] = df_features['sender_clean'].apply(len)
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    df_features['is_free_email'] = df_features['sender_domain'].isin(free_domains).astype(int)
    
    df_features['body_length'] = df_features['body'].apply(len)
    df_features["urls"] = df_features['body'].str.contains(r"https?://|www\.", regex=True, na=False).astype(int)
    
    return df_features[["urls","sender_length","is_free_email","body_length","body"]]


@app.post("/predict")
def predict(payload: EmailIn):
    df_input = pd.DataFrame([{
        "body": payload.body,
        "sender": payload.sender,
    }])

    try:
        df_enriched = feature_engineering(df_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")

    # Use selected model if provided, otherwise default
    pred_model = model
    pred_tfidf = tfidf

    if payload.model:
        filepath = os.path.join(SAVED_MODELS_DIR, f"{payload.model}.joblib")
        if not os.path.isfile(filepath):
            raise HTTPException(status_code=404, detail="Model not found")
        try:
            loaded = joblib.load(filepath)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        if isinstance(loaded, dict):
            pred_model = loaded.get("model")
            if pred_model is None:
                raise HTTPException(status_code=400, detail="Saved model file is missing model object")
            # Use vectorizer from saved file if available, otherwise fall back to default
            pred_tfidf = loaded.get("vectorizer", tfidf)
        else:
            # Backward compatibility: raw model object
            pred_model = loaded

    X_num = df_enriched[NUMERIC_COLUMNS].values
    X_tfidf = pred_tfidf.transform(df_enriched['body'])
    X_final = hstack([X_tfidf, csr_matrix(X_num)])

    try:
        pred = pred_model.predict(X_final)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = {"prediction": int(pred[0])}
    if hasattr(pred_model, "predict_proba"):
        result["probability"] = pred_model.predict_proba(X_final).tolist()[0]
    return result


# In-memory job tracking for training jobs
training_jobs = {}

class TrainIn(BaseModel):
    algo: str
    params: dict
    tuning: dict = {}

class SaveModelIn(BaseModel):
    job_id: str

SAVED_MODELS_DIR = "saved_models"

def _validate_model_name(name: str) -> str:
    """Validate model name to prevent path traversal. Returns sanitized name or raises."""
    if not name or len(name) > 128:
        raise HTTPException(status_code=400, detail="Invalid model name")
    if ".." in name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid characters in model name")
    if not all(c.isalnum() or c in "_-" for c in name):
        raise HTTPException(status_code=400, detail="Model name must be alphanumeric, hyphen, or underscore")
    return name

def _ensure_saved_models_dir():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def _get_model_algo(model) -> str:
    """Try to infer algorithm name from a model object."""
    if model is None:
        return "unknown"
    cls_name = type(model).__name__
    module = type(model).__module__
    if "xgboost" in module:
        return "XGBoost"
    if "catboost" in module:
        return "CatBoost"
    if "sklearn" in module:
        return cls_name
    return cls_name

def run_training_job(job_id: str, payload: TrainIn):
    try:
        # Lazy imports to avoid loading heavy deps at startup
        from automl import (
            feature_engineering,
            train_and_log_model,
            run_grid_search,
            run_random_search,
            run_bayesian_search,
        )
        from sklearn.model_selection import train_test_split

        # Load and prepare data
        df = pd.read_csv("data/SpamAssasin.csv")
        df_enriched = feature_engineering(df)

        X = df_enriched[["urls", "sender_length", "is_free_email", "body_length", "body"]]
        y = df_enriched["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        algo = payload.algo
        training_jobs[job_id]["algo"] = algo

        tuning = payload.tuning or {}
        method = tuning.get("method")

        def update_progress(current_run, total_runs):
            training_jobs[job_id]["current_run"] = current_run
            training_jobs[job_id]["progress"] = int((current_run / total_runs) * 100) if total_runs else 0

        if method:
            # Hyperparameter tuning
            param_ranges = tuning.get("param_ranges", {})
            n_runs = tuning.get("n_runs", 5)
            training_jobs[job_id]["total_runs"] = n_runs
            training_jobs[job_id]["message"] = f"Running {method} ({n_runs} iterations)"

            search_fn = {
                "GridSearch": run_grid_search,
                "RandomSearch": run_random_search,
                "Bayesian": run_bayesian_search,
            }.get(method)

            if search_fn is None:
                raise ValueError(f"Unknown tuning method: {method}")

            result = search_fn(algo, param_ranges, n_runs, X_train, X_test, y_train, y_test, progress_callback=update_progress)

            # Extract best model from best_run
            best_run = result.get("best_run")
            if best_run:
                training_jobs[job_id]["model"] = best_run.get("model")
                training_jobs[job_id]["vectorizer"] = best_run.get("vectorizer")

            # Format runs for API response (strip model/vectorizer objects from JSON)
            runs = []
            for r in result.get("runs", []):
                runs.append({
                    "run_number": r["run_number"],
                    "params": r["params"],
                    "metrics": r["metrics"],
                    "score": r["metrics"].get("f1_score", 0),
                })

            best_run_serializable = None
            if best_run:
                best_run_serializable = {
                    "run_number": best_run["run_number"],
                    "params": best_run["params"],
                    "metrics": best_run["metrics"],
                    "score": best_run["metrics"].get("f1_score", 0),
                }

            training_jobs[job_id]["progress"] = 100
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["message"] = "Training completed successfully"
            training_jobs[job_id]["best_run"] = best_run_serializable
            training_jobs[job_id]["runs"] = runs

        else:
            # Single run with provided params
            training_jobs[job_id]["total_runs"] = 1
            training_jobs[job_id]["message"] = "Training single run"

            result = train_and_log_model(algo, payload.params, X_train, X_test, y_train, y_test)

            training_jobs[job_id]["model"] = result["model"]
            training_jobs[job_id]["vectorizer"] = result["vectorizer"]
            training_jobs[job_id]["current_run"] = 1
            training_jobs[job_id]["progress"] = 100

            run_entry = {
                "run_number": 1,
                "params": payload.params,
                "metrics": result["metrics"],
                "score": result["metrics"].get("f1_score", 0),
            }

            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["message"] = "Training completed successfully"
            training_jobs[job_id]["best_run"] = run_entry
            training_jobs[job_id]["runs"] = [run_entry]

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = str(e)

@app.post("/train")
def train(payload: TrainIn):
    job_id = uuid.uuid4().hex
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "current_run": 0,
        "total_runs": 0,
        "message": "Job started",
        "best_run": None,
        "runs": [],
    }

    thread = threading.Thread(target=run_training_job, args=(job_id, payload))
    thread.start()

    return {"job_id": job_id, "status": "started"}

@app.get("/train/{job_id}/status")
def get_train_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = training_jobs[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "current_run": job["current_run"],
        "total_runs": job["total_runs"],
        "message": job["message"],
    }

@app.get("/train/{job_id}/results")
def get_train_results(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = training_jobs[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "best_run": job["best_run"],
        "runs": job["runs"],
    }

@app.get("/models")
def list_models():
    _ensure_saved_models_dir()
    models = []
    try:
        for filename in os.listdir(SAVED_MODELS_DIR):
            if not filename.endswith(".joblib"):
                continue
            filepath = os.path.join(SAVED_MODELS_DIR, filename)
            name = filename[:-7]  # remove .joblib
            ctime = os.path.getctime(filepath)
            date_str = datetime.fromtimestamp(ctime).isoformat()
            algo = "unknown"
            try:
                loaded = joblib.load(filepath)
                if isinstance(loaded, dict) and "algo" in loaded:
                    algo = loaded["algo"]
                elif hasattr(loaded, "__class__"):
                    algo = _get_model_algo(loaded)
            except Exception:
                pass
            models.append({"name": name, "algo": algo, "date": date_str})
    except FileNotFoundError:
        pass
    return models

@app.get("/models/{name}/download")
def download_model(name: str):
    name = _validate_model_name(name)
    filepath = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(filepath, filename=f"{name}.joblib", media_type="application/octet-stream")

@app.post("/models/{name}/save")
def save_model(name: str, payload: SaveModelIn):
    name = _validate_model_name(name)
    if payload.job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = training_jobs[payload.job_id]
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed")
    model_obj = job.get("model")
    if model_obj is None:
        raise HTTPException(status_code=400, detail="No trained model available in this job")
    _ensure_saved_models_dir()
    filepath = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
    try:
        joblib.dump({
            "model": model_obj,
            "algo": job.get("algo", "unknown"),
            "vectorizer": job.get("vectorizer"),
        }, filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")
    return {"name": name, "status": "saved"}