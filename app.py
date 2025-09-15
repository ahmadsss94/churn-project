# app.py
from fastapi import FastAPI
import joblib
import pandas as pd

# create the FastAPI app at top level
app = FastAPI()

# load your trained artifact
art = joblib.load("churn_model.joblib")
pipe = art["model"]
thr = art.get("threshold", 0.5)
feat_order = art["feature_order"]  # list of feature column names used in training

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_proba")
def predict_proba(payload: dict):
    # payload must include ALL feature names in feat_order
    X = pd.DataFrame([payload])[feat_order]
    p = float(pipe.predict_proba(X)[:, 1][0])
    return {"churn_probability": p}

def _complete(payload: dict) -> dict:
    p = dict(payload)
    ns = float(p.get("n_sessions", 0) or 0)
    if "events_per_session" not in p and "event_count" in p:
        p["events_per_session"] = float(p["event_count"]) / (ns if ns else 1.0)
    if "songs_per_session" not in p and "n_songs" in p:
        p["songs_per_session"] = float(p["n_songs"]) / (ns if ns else 1.0)
    return p

@app.post("/predict")
def predict(payload: dict):
    payload = _complete(payload)
    X = pd.DataFrame([payload])[feat_order]
    p = float(pipe.predict_proba(X)[:,1][0])
    y = int(p >= thr)
    return {"churn_probability": p, "prediction": y, "threshold": thr}


# optional: let you run `python app.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
