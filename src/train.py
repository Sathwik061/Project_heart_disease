import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


data_path=r"D:\MLOPS\DSA\PROJECT_HEART\heart_disease\heart_cleaned1.csv"
def load_data(path):
    return pd.read_csv(path)

def train_model(model_type='logistic'):
    df=load_data(data_path)
    X=df.drop("heart_disease_target",axis=1)
    y=df["heart_disease_target"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    if model_type=="logistic":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type=="random_forest":
         model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    else: raise ValueError("Invalid model type") 



    pipeline=Pipeline([("scaler",StandardScaler()),("model",model)])
    with mlflow.start_run(run_name=model_type) as run:
        pipeline.fit(X_train,y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(pipeline,"model")

        run_id=run.info.run_id

        
    return accuracy, roc_auc, run_id
if __name__ == "__main__":
    acc_lr, roc_lr, run_lr = train_model("logistic")
    print("Logistic Regression Accuracy:", acc_lr)
    print("Logistic Regression ROC-AUC:", roc_lr)

    acc_rf, roc_rf, run_rf = train_model("random_forest")
    print("Random Forest Accuracy:", acc_rf)
    print("Random Forest ROC-AUC:", roc_rf)

