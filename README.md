# PoC Phishing Detection

Proof-of-concept implementation for detecting phishing emails. Uses a TF-IDF vectorizer and a tree-based classifier (pretrained models are stored in `models/`). Minimal API is provided in `api.py` to serve predictions.

## Quick start

Requirements: Python 3.8+ and the packages in `requirements.txt`.

Create a local virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # PowerShell
pip install -r requirements.txt
```

Run the API (development):

```bash
uvicorn api:app --reload
```

The service exposes a POST /predict endpoint that accepts JSON: `{ "email": "<raw email text>" }`.

## Usage

Example using curl:

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"email":"Please update your account details"}'
```

Simple Python example:

```python
import requests

resp = requests.post("http://127.0.0.1:8000/predict", json={"email": "Sample message text"})
print(resp.json())
```

## Project structure

- `api.py` — FastAPI app, endpoint `/predict` (loads `models/tfidf.joblib` and `models/xgb_model.joblib`).
- `main.ipynb` — notebook for exploration and training.
- `data/Phishing_Email.csv` — source dataset used for the proof of concept.
- `models/` — serialized vectorizer and model files used by the API.

## Training / update models

Training code is in `main.ipynb`. To update the served models:

1. Retrain vectorizer and classifier in the notebook.
2. Export the fitted TF-IDF and trained model to `models/tfidf.joblib` and `models/xgb_model.joblib` (joblib.dump).
3. Restart the API.

## Notes

- The API returns `prediction` (int) and, if available, `probability` from `predict_proba`.
- This repository is a proof of concept: validate, test, and harden before any production use.

No license specified.

