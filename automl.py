import itertools
import warnings

from skopt import Optimizer
from skopt.space import Integer, Real

import pandas as pd
import numpy as np
import os
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix

import mlflow
import mlflow.xgboost
import mlflow.catboost
import shap

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

NUMERIC_COLUMNS = ['sender_length', 'is_free_email']


def enrichir_donnees_phishing(df):
    """Feature engineering matching api.py exactly."""
    df_features = df.copy()

    if 'body' in df_features.columns:
        df_features['body'] = df_features['body'].fillna('')

    def extract_email(text):
        if pd.isna(text):
            return ""
        match = re.search(r'<([^>]+)>', str(text))
        if match:
            return match.group(1)
        return str(text)

    df_features['sender_clean'] = df_features['sender'].apply(extract_email)
    df_features['sender_domain'] = df_features['sender_clean'].apply(
        lambda x: x.split('@')[-1] if '@' in x else "unknown"
    )
    df_features['sender_length'] = df_features['sender_clean'].apply(len)
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    df_features['is_free_email'] = df_features['sender_domain'].isin(free_domains).astype(int)

    df_features['body_length'] = df_features['body'].apply(len)
    df_features["urls"] = df_features['body'].str.contains(
        r"https?://|www\.", regex=True, na=False
    ).astype(int)

    cols_to_return = ["urls", "sender_length", "is_free_email", "body_length", "body"]
    if "label" in df_features.columns:
        cols_to_return.append("label")
    return df_features[cols_to_return]


def generate_run_name(algo_name, params):
    """Crée un nom de run lisible basé sur les hyperparamètres."""
    n = params.get('n_estimators', params.get('iterations', 100))
    lr = params.get('learning_rate', 0.1)

    base_name = f"{algo_name}_{n}n_{lr}lr"

    if 'reg_alpha' in params:
        base_name += f"_{params['reg_alpha']}L1"
    if 'l2_leaf_reg' in params:
        base_name += f"_{params['l2_leaf_reg']}L2"

    return base_name


def train_and_log_model(algo_name, params, X_train, X_test, y_train, y_test, generate_shap=True):
    """
    Fonction appelée par l'UI pour lancer un entraînement complet.
    algo_name : 'XGBoost' ou 'CatBoost'
    params : dictionnaire des hyperparamètres venant de l'UI
    X_train/X_test : DataFrames with columns ['urls', 'sender_length', 'is_free_email', 'body_length', 'body']
    y_train/y_test : labels
    Returns dict with keys: metrics, model, vectorizer
    """
    mlflow.set_experiment("Phishing_Detection")
    run_name = generate_run_name(algo_name, params)

    with mlflow.start_run(run_name=run_name):

        # TF-IDF on body text
        tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
        X_train_tfidf = tfidf.fit_transform(X_train['body'])
        X_test_tfidf = tfidf.transform(X_test['body'])

        # Numeric features
        X_train_num = X_train[NUMERIC_COLUMNS].values
        X_test_num = X_test[NUMERIC_COLUMNS].values

        # Stack: TF-IDF + numeric
        X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num)])
        X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_num)])

        # Model instantiation
        if algo_name == "XGBoost":
            model = XGBClassifier(**params, random_state=42)
        elif algo_name == "CatBoost":
            model = CatBoostClassifier(**params, random_seed=42, verbose=0)
        else:
            raise ValueError("Modèle non supporté. Choisissez 'XGBoost' ou 'CatBoost'.")

        # Training
        mlflow.log_params(params)
        model.fit(X_train_final, y_train)

        # Evaluation
        y_pred = model.predict(X_test_final)
        y_prob = model.predict_proba(X_test_final)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)

        # Save vectorizer artifact
        os.makedirs("models/artifacts", exist_ok=True)
        vect_path = "models/artifacts/tfidf_vectorizer.pkl"
        with open(vect_path, "wb") as f:
            pickle.dump(tfidf, f)
        mlflow.log_artifact(vect_path, artifact_path="preprocessing")

        # Log model
        if algo_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        elif algo_name == "CatBoost":
            mlflow.catboost.log_model(model, "model")

        # SHAP summary plot (once, on final best model)
        if generate_shap:
            X_test_sample = X_test_final[:300].toarray()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)

            feature_names = list(tfidf.get_feature_names_out()) + NUMERIC_COLUMNS
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
            shap_path = "models/artifacts/shap_summary.png"
            plt.savefig(shap_path, bbox_inches="tight")
            plt.close()

            mlflow.log_artifact(shap_path, artifact_path="audit_reports")

        return {
            "metrics": metrics,
            "model": model,
            "vectorizer": tfidf,
        }


# ---------------------------------------------------------------------------
# Hyperparameter tuning helpers
# ---------------------------------------------------------------------------

_INT_PARAMS = frozenset({
    "n_estimators", "iterations", "max_depth", "depth", "border_count",
})


def _is_int_param(name):
    """Return True if the parameter should be treated as an integer."""
    return name in _INT_PARAMS


def _build_param_value(r, as_int):
    """Convert a range dict {min, max, step} to a concrete value."""
    val = r["min"]
    if as_int:
        return int(round(val))
    return float(val)


def _param_range_values(r, as_int):
    """Yield all values in a range dict {min, max, step}."""
    lo, hi, step = r["min"], r["max"], r.get("step", 1)
    if as_int:
        lo, hi, step = int(round(lo)), int(round(hi)), int(round(step))
        val = lo
        while val <= hi:
            yield val
            val += step
    else:
        val = float(lo)
        hi = float(hi)
        step = float(step)
        while val <= hi + step * 1e-9:
            yield round(val, 10)
            val += step


def _generate_all_combinations(param_ranges):
    """Generate every combination from param_ranges. Returns list of dicts."""
    keys = sorted(param_ranges.keys())
    value_lists = []
    for k in keys:
        r = param_ranges[k]
        as_int = _is_int_param(k)
        value_lists.append(list(_param_range_values(r, as_int)))
    combos = []
    for values in itertools.product(*value_lists):
        combo = {}
        for k, v in zip(keys, values):
            combo[k] = int(v) if _is_int_param(k) else float(v)
        combos.append(combo)
    return combos


def _sample_random_params(param_ranges, n):
    """Sample n random parameter combinations from param_ranges."""
    import random
    combos = []
    keys = sorted(param_ranges.keys())
    for _ in range(n):
        combo = {}
        for k in keys:
            r = param_ranges[k]
            lo, hi = r["min"], r["max"]
            if _is_int_param(k):
                combo[k] = random.randint(int(round(lo)), int(round(hi)))
            else:
                combo[k] = round(random.uniform(float(lo), float(hi)), 6)
        combos.append(combo)
    return combos


def _score_run(run):
    """Primary sort key for best-run selection (higher is better)."""
    m = run.get("metrics", {})
    return m.get("f1_score", 0)


def generate_shap_summary(model, X_test, tfidf):
    """Generate SHAP summary plot for a trained model. X_test is the enriched DataFrame."""
    X_test_tfidf = tfidf.transform(X_test['body'])
    X_test_num = X_test[NUMERIC_COLUMNS].values
    X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_num)])
    X_test_sample = X_test_final[:300].toarray()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    feature_names = list(tfidf.get_feature_names_out()) + NUMERIC_COLUMNS
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    shap_path = "models/artifacts/shap_summary.png"
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(shap_path, artifact_path="audit_reports")


def _run_search(param_sets, algo_name, n_runs, X_train, X_test, y_train, y_test, progress_callback=None):
    """Core loop: train each param set and collect results."""
    runs = []
    for idx, params in enumerate(param_sets[:n_runs]):
        result = train_and_log_model(algo_name, params, X_train, X_test, y_train, y_test, generate_shap=False)
        run_entry = {
            "run_number": idx + 1,
            "params": params,
            "metrics": result["metrics"],
            "model": result["model"],
            "vectorizer": result["vectorizer"],
        }
        runs.append(run_entry)
        if progress_callback:
            progress_callback(idx + 1, n_runs)

    best = max(runs, key=_score_run) if runs else None
    return {"best_run": best, "runs": runs}


# ---------------------------------------------------------------------------
# Public tuning functions
# ---------------------------------------------------------------------------

def run_grid_search(algo_name, param_ranges, n_runs, X_train, X_test, y_train, y_test, progress_callback=None):
    """
    Exhaustive grid search over param_ranges, truncated to n_runs if necessary.
    param_ranges: dict of {param_name: {"min": x, "max": y, "step": z}}
    """
    all_combos = _generate_all_combinations(param_ranges)
    total = len(all_combos)
    if total > n_runs:
        warnings.warn(
            f"GridSearch: {total} combinations exceed n_runs={n_runs}. "
            f"Truncating to first {n_runs} runs."
        )
    result = _run_search(all_combos, algo_name, n_runs, X_train, X_test, y_train, y_test, progress_callback=progress_callback)
    if result["best_run"]:
        best = result["best_run"]
        generate_shap_summary(best["model"], X_test, best["vectorizer"])
    return result


def run_random_search(algo_name, param_ranges, n_runs, X_train, X_test, y_train, y_test, progress_callback=None):
    """Random sampling from param_ranges."""
    combos = _sample_random_params(param_ranges, n_runs)
    result = _run_search(combos, algo_name, n_runs, X_train, X_test, y_train, y_test, progress_callback=progress_callback)
    if result["best_run"]:
        best = result["best_run"]
        generate_shap_summary(best["model"], X_test, best["vectorizer"])
    return result


def run_bayesian_search(algo_name, param_ranges, n_runs, X_train, X_test, y_train, y_test, progress_callback=None):
    """
    Bayesian optimization using skopt.Optimizer.
    Each param in param_ranges maps to Integer or Real space.
    """
    keys = sorted(param_ranges.keys())
    dimensions = []
    for k in keys:
        r = param_ranges[k]
        if _is_int_param(k):
            dimensions.append(Integer(int(round(r["min"])), int(round(r["max"])), name=k))
        else:
            dimensions.append(Real(float(r["min"]), float(r["max"]), name=k))

    opt = Optimizer(dimensions=dimensions, random_state=42)
    runs = []

    for idx in range(n_runs):
        suggestion = opt.ask()
        params = {}
        for k, v in zip(keys, suggestion):
            params[k] = int(v) if _is_int_param(k) else float(v)

        result = train_and_log_model(algo_name, params, X_train, X_test, y_train, y_test, generate_shap=False)
        run_entry = {
            "run_number": idx + 1,
            "params": params,
            "metrics": result["metrics"],
            "model": result["model"],
            "vectorizer": result["vectorizer"],
        }
        runs.append(run_entry)

        # skopt minimizes, so negate f1_score
        score = result["metrics"].get("f1_score", 0)
        opt.tell(suggestion, -score)

        if progress_callback:
            progress_callback(idx + 1, n_runs)

    best = max(runs, key=_score_run) if runs else None
    result = {"best_run": best, "runs": runs}
    if result["best_run"]:
        best = result["best_run"]
        generate_shap_summary(best["model"], X_test, best["vectorizer"])
    return result
