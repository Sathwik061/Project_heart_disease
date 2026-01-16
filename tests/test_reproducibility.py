import pytest
from src.train import train_model


@pytest.mark.parametrize("model_type", ["logistic", "random_forest"])
def test_reproducibility(model_type):
   

    acc1, roc1, _ = train_model(model_type)
    acc2, roc2, _ = train_model(model_type)

    assert acc1 == pytest.approx(acc2, rel=1e-6)
    assert roc1 == pytest.approx(roc2, rel=1e-6)
