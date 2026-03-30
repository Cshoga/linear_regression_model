# Student Counseling – Score Predictor

## Mission
Many students fall behind without early warning. This app predicts a student's overall academic score from study habits, attendance, subject marks, and background factors — giving counselors data-driven insight to intervene before it is too late.

**Dataset:** [Student Performance Dataset – Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/student-performance-dataset?resource=download) | 25,000 rows × 16 columns covering demographics, study habits, and academic scores.

---

## Project Structure

```
linear_regression_model/
└── summative/
    ├── linear_regression/
    │   └── multivariate.ipynb          
    ├── API/
    │   ├── prediction.py               
    │   ├── requirements.txt
    │   ├── python-version         
    │   ├── train.py                 
    │   └── feature_columns.json        
    └── FlutterApp/
        └── student_score_prediction_app/  
```

---

## Live API

**Swagger UI (click to open):**
```
https://linear-regression-model-0959.onrender.com
```

**Predict endpoint:**
```
POST https://linear-regression-model-0959.onrender.com/predict
```

**Retrain endpoint:**
```
POST https://linear-regression-model-0959.onrender.com/retrain
```

---

## Running the API Locally

```bash
cd summative/API
pip install -r requirements.txt
uvicorn prediction:app --reload --port 8000
# Open http://localhost:8000/docs
```

---

## Running the Flutter App

### Prerequisites
- Flutter SDK ≥ 3.0  ([install guide](https://flutter.dev/docs/get-started/install))
- Android Studio or VS Code with Flutter extension
- An Android device or emulator (API 21+)

### Steps

```bash
cd summative/FlutterApp/student_score_prediction_app

# 1. Install dependencies
flutter pub get

# 2. Run on connected device / emulator
flutter run

# Or build an APK:
flutter build apk --release
# APK will be at build/app/outputs/flutter-apk/app-release.apk
```

---

## Video Demo

[YouTube Demo Link – coming soon]

---

## Model Performance

| Model               | Test MSE | Test R²  |
|---------------------|----------|----------|
| SGD Linear Reg.     | ~22      | ~0.94    |
| Decision Tree       | ~8       | ~0.97    |
| **Random Forest**  | **~10**  | **~0.97** |

Random Forest was selected as the best model (highest R², robust to overfitting).

---

## API Input Variables

| Field | Type | Range / Values |
|-------|------|----------------|
| age | int | 10–25 |
| gender | string | male / female / other |
| school_type | string | public / private |
| parent_education | string | high school / graduate / post graduate / phd / no formal |
| study_hours | float | 0–12 |
| attendance_percentage | float | 0–100 |
| internet_access | string | yes / no |
| travel_time | string | <15 min / 15-30 min / 30-60 min / >60 min |
| extra_activities | string | yes / no |
| study_method | string | notes / textbook / online videos / group study / mixed |
| math_score | float | 0–100 |
| science_score | float | 0–100 |
| english_score | float | 0–100 |
