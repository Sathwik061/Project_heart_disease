import mlflow
import mlflow.pyfunc
import pandas as pd
import os


def test_deployment():

    # Resolve project root
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Path to heart disease dataset
    data_path = os.path.join(
        project_root,
        "heart_disease",
        "heart_cleaned1.csv"
    )

    # Connect to MLflow
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("Default")

    assert experiment is not None, "MLflow experiment not found"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    assert len(runs) > 0, "No MLflow runs found"

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.read_csv(data_path)

    X = df.drop("heart_disease_target", axis=1)
    sample = X.iloc[:1]

    prediction = model.predict(sample)

    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
