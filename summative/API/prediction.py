# ============================================================
# Student Performance Prediction API
# ============================================================
import json
import os
import io

from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLS_PATH  = os.path.join(BASE_DIR, "feature_columns.json")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(COLS_PATH) as f:
    FEATURE_COLUMNS: List[str] = json.load(f)

app = FastAPI(
    title="Student Counseling – Score Predictor",
    description=(
        "Predict a student's overall academic score using study habits, "
        "attendance, subject scores, and background factors. "
        "Designed to support counselors in identifying at-risk students."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "https://student-score-predictor.onrender.com",  
    ],
    allow_credentials=True,           
    allow_methods=["GET", "POST"],    
    allow_headers=["Content-Type", "Authorization"],  
)


class StudentInput(BaseModel):
    
    age: int = Field(
        ...,
        ge=10, le=25,
        description="Student age in years (10–25)",
        example=16
    )
    gender: str = Field(
        ...,
        description="Student gender: 'male', 'female', or 'other'",
        example="male"
    )
    school_type: str = Field(
        ...,
        description="Type of school: 'public' or 'private'",
        example="public"
    )
    parent_education: str = Field(
        ...,
        description=(
            "Highest parent education level: "
            "'high school', 'graduate', 'post graduate', 'phd', 'no formal'"
        ),
        example="graduate"
    )

    
    study_hours: float = Field(
        ...,
        ge=0.0, le=12.0,
        description="Average study hours per day (0–12)",
        example=4.5
    )
    attendance_percentage: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Attendance percentage (0–100)",
        example=87.5
    )
    internet_access: str = Field(
        ...,
        description="Does the student have internet access? 'yes' or 'no'",
        example="yes"
    )
    travel_time: str = Field(
        ...,
        description=(
            "Daily travel time to school: "
            "'<15 min', '15-30 min', '30-60 min', or '>60 min'"
        ),
        example="<15 min"
    )
    extra_activities: str = Field(
        ...,
        description="Participates in extra-curricular activities? 'yes' or 'no'",
        example="yes"
    )
    study_method: str = Field(
        ...,
        description=(
            "Preferred study method: "
            "'notes', 'textbook', 'online videos', 'group study', or 'mixed'"
        ),
        example="mixed"
    )

    
    math_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Math score (0–100)",
        example=72.0
    )
    science_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Science score (0–100)",
        example=68.0
    )
    english_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="English score (0–100)",
        example=75.0
    )



def encode_input(data: StudentInput) -> pd.DataFrame:
    """
    Replicate the same get_dummies() encoding used during training.
    We manually set the one-hot columns so the feature order matches.
    """
    gender           = data.gender.lower().strip()
    school_type      = data.school_type.lower().strip()
    parent_education = data.parent_education.lower().strip()
    internet_access  = data.internet_access.lower().strip()
    travel_time      = data.travel_time.strip()
    extra_activities = data.extra_activities.lower().strip()
    study_method     = data.study_method.lower().strip()

    
    row = {
        "age":                   data.age,
        "study_hours":           data.study_hours,
        "attendance_percentage": data.attendance_percentage,
        "math_score":            data.math_score,
        "science_score":         data.science_score,
        "english_score":         data.english_score,
    }

    
    row["gender_male"]  = 1 if gender == "male"  else 0
    row["gender_other"] = 1 if gender == "other" else 0

    
    row["school_type_public"] = 1 if school_type == "public" else 0

    
    for level in ["graduate", "high school", "no formal", "phd", "post graduate"]:
        col = f"parent_education_{level}"
        row[col] = 1 if parent_education == level else 0

    
    row["internet_access_yes"] = 1 if internet_access == "yes" else 0

    
    for t in ["30-60 min", "<15 min", ">60 min"]:
        col = f"travel_time_{t}"
        row[col] = 1 if travel_time == t else 0

    
    row["extra_activities_yes"] = 1 if extra_activities == "yes" else 0

    
    for method in ["group study", "mixed", "notes", "online videos", "textbook"]:
        col = f"study_method_{method}"
        row[col] = 1 if study_method == method else 0

    
    df_row = pd.DataFrame([row])[FEATURE_COLUMNS]
    return df_row



@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Student Counseling API is running 🎓",
        "docs": "/docs",
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": "RandomForestRegressor"}


@app.post("/predict", tags=["Prediction"])
def predict(student: StudentInput):
    """
    **Predict a student's overall academic score.**

    Supply the student's demographic info, study habits, and subject scores.
    Returns the predicted overall score (0–100) and a counseling flag.
    """
    try:
        
        df_input = encode_input(student)

        scaled = scaler.transform(df_input)

        prediction = float(model.predict(scaled)[0])
        prediction = round(min(max(prediction, 0.0), 100.0), 2)  # clamp 0–100

        if prediction < 50:
            flag = "High risk: urgent counseling recommended"
        elif prediction < 65:
            flag = "At risk: counseling suggested"
        else:
            flag = "On track"

        return {
            "predicted_overall_score": prediction,
            "counseling_flag": flag,
            "input_received": student.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/retrain", tags=["Model Update"])
async def retrain(file: UploadFile = File(...)):
    
    global model, scaler   

    try:
        contents = await file.read()
        df_new = pd.read_csv(io.BytesIO(contents))

        
        required_cols = [
            "age", "gender", "school_type", "parent_education",
            "study_hours", "attendance_percentage", "internet_access",
            "travel_time", "extra_activities", "study_method",
            "math_score", "science_score", "english_score", "overall_score"
        ]
        missing = [c for c in required_cols if c not in df_new.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"CSV is missing columns: {missing}"
            )

        
        if "student_id" in df_new.columns:
            df_new = df_new.drop("student_id", axis=1)
        if "final_grade" in df_new.columns:
            df_new = df_new.drop("final_grade", axis=1)

        df_new = pd.get_dummies(df_new, drop_first=True)

        for col in FEATURE_COLUMNS:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new = df_new[FEATURE_COLUMNS + ["overall_score"]]

        X_new = df_new[FEATURE_COLUMNS]
        y_new = df_new["overall_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )

        new_scaler = StandardScaler()
        X_train_sc = new_scaler.fit_transform(X_train)
        X_test_sc  = new_scaler.transform(X_test)

        new_model = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        new_model.fit(X_train_sc, y_train)

        mse = mean_squared_error(y_test, new_model.predict(X_test_sc))
        r2  = r2_score(y_test, new_model.predict(X_test_sc))

        
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        model  = new_model    # update global
        scaler = new_scaler   # update global

        return {
            "message": "Model retrained and updated successfully ✅",
            "rows_used": len(df_new),
            "test_mse": round(mse, 4),
            "test_r2":  round(r2,  4),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)
