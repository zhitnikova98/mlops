import json
import pickle
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Вычисление метрик
    num_rows = len(df)
    metrics = {
        "accuracy": float(accuracy),
        "num_rows": int(num_rows)
    }

    # Запись метрик в JSON файл
    os.makedirs("metrics", exist_ok=True)  # Создать директорию, если нет
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Логирование метрик в MLflow
    mlflow.log_metrics(metrics)

    # Логирование JSON файла как артефакта
    mlflow.log_artifact("metrics/metrics.json")

    print(f"Metrics saved: accuracy={accuracy:.4f}, num_rows={num_rows}")
    print(f"Metrics logged to MLflow and saved to metrics/metrics.json")


# Запуск MLflow run
if __name__ == "__main__":
    with mlflow.start_run():
        evaluate_model()
