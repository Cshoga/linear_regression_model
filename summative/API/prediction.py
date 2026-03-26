import json
import os
import io
import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Performance Predictor",
    description="Predicts a student's overall score (0–100) using a Random Forest model trained on student habits and background.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_artifacts():
    model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    with open(os.path.join(BASE_DIR, "feature_columns.json")) as f:
        cols = json.load(f)
    return model, scaler, cols

model, scaler, feature_columns = _load_artifacts()

class StudentInput(BaseModel):
    age: int = Field(..., ge=10, le=25, description="Student age (10–25)")
    study_hours: float = Field(..., ge=0.0, le=16.0, description="Daily study hours (0–16)")
    attendance_percentage: float = Field(..., ge=0.0, le=100.0, description="Attendance % (0–100)")
    math_score: float = Field(..., ge=0.0, le=100.0, description="Math score (0–100)")
    science_score: float = Field(..., ge=0.0, le=100.0, description="Science score (0–100)")
    english_score: float = Field(..., ge=0.0, le=100.0, description="English score (0–100)")

    gender_male: int = Field(..., ge=0, le=1)
    gender_other: int = Field(..., ge=0, le=1)
    school_type_public: int = Field(..., ge=0, le=1)

    parent_education_graduate: int = Field(..., ge=0, le=1)
    parent_education_high_school: int = Field(..., ge=0, le=1, alias="parent_education_high school")
    parent_education_no_formal: int = Field(..., ge=0, le=1, alias="parent_education_no formal")
    parent_education_phd: int = Field(..., ge=0, le=1)
    parent_education_post_graduate: int = Field(..., ge=0, le=1, alias="parent_education_post graduate")

    internet_access_yes: int = Field(..., ge=0, le=1)

    travel_time_30_60_min: int = Field(..., ge=0, le=1, alias="travel_time_30-60 min")
    travel_time_lt15_min: int = Field(..., ge=0, le=1, alias="travel_time_<15 min")
    travel_time_gt60_min: int = Field(..., ge=0, le=1, alias="travel_time_>60 min")

    extra_activities_yes: int = Field(..., ge=0, le=1)

    study_method_group_study: int = Field(..., ge=0, le=1, alias="study_method_group study")
    study_method_mixed: int = Field(..., ge=0, le=1)
    study_method_notes: int = Field(..., ge=0, le=1)
    study_method_online_videos: int = Field(..., ge=0, le=1, alias="study_method_online videos")
    study_method_textbook: int = Field(..., ge=0, le=1)

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_overall_score: float
    message: str


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Student Performance Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentInput):
    """
    Predicting a student's overall score.

    **Categorical baselines** :
    - gender = female
    - school_type = private
    - parent_education = diploma
    - travel_time = 15-30 min
    - study_method = coaching
    """
    input_dict = student.model_dump(by_alias=True)
    row = np.array([[input_dict.get(col, 0) for col in feature_columns]])
    row_scaled = scaler.transform(row)
    score = float(model.predict(row_scaled)[0])
    score = max(0.0, min(100.0, round(score, 2)))
    return PredictionResponse(predicted_overall_score=score, message="Prediction successful.")


@app.post("/retrain", tags=["Retraining"])
async def retrain(file: UploadFile = File(...)):
    """
    Saved artifacts are updated automatically.
    """
    global model, scaler, feature_columns

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if "overall_score" not in new_df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an 'overall_score' column.")

    drop_cols = [c for c in ["student_id", "final_grade"] if c in new_df.columns]
    new_df = new_df.drop(columns=drop_cols)
    new_df = pd.get_dummies(new_df, drop_first=True)

    X_new = new_df.drop("overall_score", axis=1)
    y_new = new_df["overall_score"]

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    X_tr, X_te, y_tr, y_te = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    new_scaler = StandardScaler()
    X_tr_sc = new_scaler.fit_transform(X_tr)
    X_te_sc  = new_scaler.transform(X_te)

    new_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    new_model.fit(X_tr_sc, y_tr)

    mse = mean_squared_error(y_te, new_model.predict(X_te_sc))
    r2  = r2_score(y_te, new_model.predict(X_te_sc))

    joblib.dump(new_model,  os.path.join(BASE_DIR, "best_model.pkl"))
    joblib.dump(new_scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    with open(os.path.join(BASE_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X_new.columns), f)

    model, scaler, feature_columns = _load_artifacts()

    return {"message": "Model retrained successfully.", "test_mse": round(mse, 4), "test_r2": round(r2, 4), "rows_used": len(new_df)}


if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)
