from train import train_model
baseline_accuracy=0.80

def evaluate(model_type="logistic"):
    accuracy, run_id = train_model(model_type)

    print(f"Model Type: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {run_id}")

    if accuracy < baseline_accuracy:
        raise ValueError(
            f"{model_type} model performance below baseline ({baseline_accuracy})"
        )

    print("Performance meets baseline threshold")
    return accuracy, run_id


if __name__ == "__main__":   
    evaluate("logistic")

    evaluate("random_forest")
