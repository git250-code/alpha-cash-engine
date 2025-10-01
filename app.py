from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import io

# Load models
clf_model = joblib.load("alpha_cash_classifier_200k.joblib")
reg_model = joblib.load("alpha_cash_regressor_200k.joblib")

app = FastAPI(title="AlphaCash Engine API")  # <-- IMPORTANT: must be named `app`

class ManualInput(BaseModel):
    income: float
    expenses: float
    savings: float
    age: int
    risk_profile: str
    goals: str

@app.post("/predict/manual")
def predict_manual(data: ManualInput):
    df = pd.DataFrame([data.dict()])
    action = clf_model.predict(df)[0]
    expected_return = reg_model.predict(df)[0]
    return {"Suggested Action": action, "Expected Return": expected_return}

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    action = clf_model.predict(df)
    expected_return = reg_model.predict(df)
    df["Suggested Action"] = action
    df["Expected Return"] = expected_return
    return df.to_dict(orient="records")

@app.get("/")
def root():
    return {"message": "AlphaCash API is running. Use /predict/manual or /predict/csv"}
