from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
import json
from datetime import datetime

if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'iris_model.pkl')
    print("The model training was successful")

    # --- Functionality 1: Model Evaluation Metrics ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # --- Functionality 2: Feature Importance Logging ---
    feature_names = iris.feature_names
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n--- Feature Importances (ranked) ---")
    for rank, idx in enumerate(indices):
        print(f"  {rank + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # --- Functionality 3: Save Training Summary to JSON ---
    summary = {
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "test_accuracy": round(float(accuracy), 4),
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_importances": {
            feature_names[i]: round(float(importances[i]), 4)
            for i in range(len(feature_names))
        }
    }
    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining summary saved to training_summary.json")