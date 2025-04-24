"""
Generates model explanations using DALEX.
"""
import dalex as dx
import joblib
import os
from data_loader import load_data # Import the function from data_loader.py

# Define paths (relative to the project root)
MODEL_PATH = 'trained_model/model.joblib'
DATA_PATH = '_data/preem.csv'
TARGET_COLUMN = 'target'

def create_explainer(model_path=MODEL_PATH, data_path=DATA_PATH, target_col=TARGET_COLUMN):
    """Loads the model and data, then creates a DALEX explainer object."""
    # Load the model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Train the model first by running model_trainer.py")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load data
    X, y, _ = load_data(data_path, target_col=target_col)
    if X is None or y is None:
        print("Error: Failed to load data. Cannot create explainer.")
        return None

    # Create DALEX explainer
    try:
        explainer = dx.Explainer(
            model=model,
            data=X,
            y=y,
            label="RandomForestRegressor on Preem Data", # Descriptive label for the explainer
            verbose=False # Set to True for more dalex output
        )
        print("DALEX explainer created successfully.")
        return explainer
    except Exception as e:
        print(f"Error creating DALEX explainer: {e}")
        return None

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("--- Explainer Creation Script ---")
    explainer_obj = create_explainer()

    if explainer_obj:
        print("Explainer created.")
        # Example: Print model performance
        try:
            print("\nModel Performance:")
            model_perf = explainer_obj.model_performance()
            print(model_perf)
            # Example: Print variable importance
            print("\nVariable Importance (calculated using permutation):")
            var_imp = explainer_obj.model_parts()
            print(var_imp)

            # You could plot directly here for testing, e.g.:
            # model_perf.plot(show=True)
            # var_imp.plot(show=True)

        except Exception as e:
            print(f"Error calculating/printing explanations: {e}")
    else:
        print("Explainer creation failed.")
    print("-----------------------------") 