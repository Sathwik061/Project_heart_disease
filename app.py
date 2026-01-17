import gradio as gr
import mlflow.pyfunc
import pandas as pd

# Load exported MLflow model (dumped locally)
model = mlflow.pyfunc.load_model("model/model")

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
    data = [[
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
    ]]

    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

    # Model prediction
    prediction = model.predict(df)[0]
    if prediction == 1:
        return "Yes you have heart disease"
    else:
        return "You are healthy no heart disease predicted"
    

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
