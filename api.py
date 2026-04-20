from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

class EmailIn(BaseModel):
    email: str

app = FastAPI(title="PhishingDetector")

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

# Chemin des modèles
MODEL_PATH = "models/xgb_model.joblib"
VECT_PATH = "models/tfidf.joblib"

tfidf = joblib.load(VECT_PATH)
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(payload: EmailIn):
    text = payload.email
    X = tfidf.transform([text])
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = {"prediction": int(pred[0])}
    if hasattr(model, "predict_proba"):
        result["probability"] = model.predict_proba(X).tolist()[0]
    return result