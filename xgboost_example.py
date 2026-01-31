import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def main():
    # 1. Load Dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")

    # 2. Split Data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # 3. Initialize XGBoost Classifier
    print("\nInitializing XGBoost Classifier...")
    # use_label_encoder=False is deprecated in newer versions but good to be explicit if using old versions.
    # eval_metric='mlogloss' removes a warning for multiclass classification.
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    # 4. Train the Model
    print("Training the model...")
    model.fit(X_train, y_train)

    # 5. Make Predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # 6. Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Optional: Show some predictions vs actual
    print("\nSample Predictions (First 5):")
    results = pd.DataFrame({
        'Actual': y_test[:5],
        'Predicted': y_pred[:5]
    })
    # Map back to names for readability
    results['Actual_Name'] = [target_names[i] for i in results['Actual']]
    results['Predicted_Name'] = [target_names[i] for i in results['Predicted']]
    print(results[['Actual_Name', 'Predicted_Name']])

if __name__ == "__main__":
    main()
