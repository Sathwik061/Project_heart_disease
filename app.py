import gradio as gr
import joblib
import pandas as pd

# Load exported MLflow model (dumped locally)
# Load both models
lr_model = joblib.load("model/logistic.pkl")
rf_model = joblib.load("model/random_forest.pkl")

# Feature columns (MUST match training exactly)
FEATURE_COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "serum_cholesterol",
    "fasting_blood_sugar",
    "resting_electrocardiographic_results",
    "maximum_heart_rate_achieved",
    "exercise_induced_angina",
    "st_depression_exercise",
    "st_slope",
    "number_of_major_vessels",
    "thalassemia",
]

def predict(
    age,
    sex,
    chest_pain_type,
    resting_blood_pressure,
    serum_cholesterol,
    fasting_blood_sugar,
    resting_electrocardiographic_results,
    maximum_heart_rate_achieved,
    exercise_induced_angina,
    st_depression_exercise,
    st_slope,
    number_of_major_vessels,
    thalassemia,
):
    # Create input DataFrame
    df=pd.DataFrame( [[
        age,
        sex,
        chest_pain_type,
        resting_blood_pressure,
        serum_cholesterol,
        fasting_blood_sugar,
        resting_electrocardiographic_results,
        maximum_heart_rate_achieved,
        exercise_induced_angina,
        st_depression_exercise,
        st_slope,
        number_of_major_vessels,
        thalassemia,
    ]],columns=FEATURE_COLUMNS)

     # --- Logistic Regression ---
    lr_prob = lr_model.predict_proba(df)[0][1]   # probability of disease
    lr_pred = 1 if lr_prob > 0.5 else 0

    # --- Random Forest ---
    rf_prob = rf_model.predict_proba(df)[0][1]
    rf_pred = 1 if rf_prob > 0.5 else 0

    # Average risk (simple ensemble for display)
    avg_risk = (lr_prob + rf_prob) / 2
    return f"""
Risk Probability: {avg_risk*100:.2f} %
Logistic Regression: {"You are having Heart Disease" if lr_pred == 1 else "You don't have Heart Disease"}
Random Forest: {"You are having Heart Disease" if rf_pred == 1 else "You don't have Heart Disease"}
"""
# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex (1 = Male, 0 = Female)"),
        gr.Number(label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Serum Cholesterol"),
        gr.Number(label="Fasting Blood Sugar"),
        gr.Number(label="Resting ECG Results"),
        gr.Number(label="Maximum Heart Rate Achieved"),
        gr.Number(label="Exercise Induced Angina"),
        gr.Number(label="ST Depression (Exercise)"),
        gr.Number(label="ST Slope"),
        gr.Number(label="Number of Major Vessels"),
        gr.Number(label="Thalassemia"),
    ],
    outputs="text",
    title="Heart Disease Prediction App",
    description="MLflow + Gradio deployment using UCI Heart Disease Dataset",
)

interface.launch()
