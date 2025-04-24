"""
Trains the machine learning model.
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# import joblib # No longer needed
import mlflow # Added
import mlflow.sklearn # Added
from data_loader import load_data # Import the function from data_loader.py

# Define the directory to save the MLflow model
# Note: mlflow saves a DIRECTORY, not just a single file.
MODEL_SAVE_PATH = os.path.join('trained_model', 'model_mlflow')
DATA_PATH = '_data/preem.csv'
TARGET_COLUMN = 'target'

def train_model(X, y, save_path):
    """Trains a RandomForestRegressor model and saves it using MLflow format."""
    if X is None or y is None:
        print("Error: Features (X) or target (y) data is missing. Cannot train model.")
        return None

    print(f"Training RandomForestRegressor model...")
    # Initialize the model
    rf_params = {"n_estimators": 100, "random_state": 42, "oob_score": True}
    model = RandomForestRegressor(**rf_params)

    try:
        # Train the model
        model.fit(X, y)
        oob_score = model.oob_score_
        print(f"Model trained successfully. OOB Score: {oob_score:.4f}")

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # --- Save using MLflow --- #
        print(f"Saving model in MLflow format to: {save_path}")
        mlflow.sklearn.save_model(
            sk_model=model,
            path=save_path,
            # Optionally add signature, input example, etc.
            # signature=...,
            # input_example=X.iloc[:5]
        )
        print(f"Model saved successfully.")
        # ------------------------- #

        # --- Optional: Log to MLflow Tracking (if server configured) --- #
        # with mlflow.start_run() as run:
        #     print(f"MLflow Run ID: {run.info.run_id}")
        #     mlflow.log_params(rf_params)
        #     mlflow.log_metric("oob_score", oob_score)
        #     mlflow.sklearn.log_model(model, "model") # Log artifact within run
        #     print("Logged run to MLflow Tracking Server.")
        # --- End Optional Logging --- #

        return model
    except Exception as e:
        print(f"Error during model training or saving: {e}")
        return None

# Main execution block
if __name__ == '__main__':
    print("--- Model Training Script (MLflow) ---")
    # Load data
    X, y, _ = load_data(DATA_PATH, target_col=TARGET_COLUMN)

    # Train and save the model
    if X is not None and y is not None:
        trained_model = train_model(X, y, MODEL_SAVE_PATH)
        if trained_model:
            print("Model training and saving complete.")
        else:
            print("Model training/saving failed.")
    else:
        print("Data loading failed, skipping model training.")
    print("-------------------------------------") 