import mlflow
import numpy as np
import pytest
from src.train import train_model


@pytest.mark.parametrize("model_type", ["logistic", "random_forest"])
def test_prediction(model_type):

    accuracy, roc_auc, run_id = train_model(model_type)

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Heart disease dataset has 13 features
    sample = np.random.rand(1, 13)

    prediction = model.predict(sample)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]

