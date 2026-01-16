import mlflow.pyfunc
import pandas as pd

MODEL_URI = "runs:/c550db5c08ed49039e80c19d2e75a50c/model"


def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)


def predict():
    model = load_model()

    df = pd.read_csv("heart_disease/heart_cleaned1.csv")

    X = df.drop("heart_disease_target", axis=1)

    sample = X.iloc[:1]

    prediction = model.predict(sample)
    return prediction


if __name__ == "__main__":
    pred = predict()
    print("Heart Disease Prediction:", pred)
