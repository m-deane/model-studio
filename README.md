# Model Studio (Python Implementation)

This project provides a Python implementation of an interactive model explanation dashboard, similar to the R `modelStudio` package.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data:**
    Place your dataset (e.g., `preem.csv`) in the `_data/` directory.

3.  **(Optional) Train Model:**
    Run the `model_trainer.py` script if you need to train a new model:
    ```bash
    python model_trainer.py
    ```
    *Note: A pre-trained model might be provided or generated automatically by the app in future steps.*

4.  **Run the Dashboard:**
    ```bash
    python app.py
    ```
    The dashboard will be available at http://127.0.0.1:8050/ by default.

## Project Structure

*   `_data/`: Contains the dataset(s).
*   `assets/`: Static files for the Dash app (e.g., CSS).
*   `trained_model/`: Stores the saved machine learning model.
*   `data_loader.py`: Handles data loading and preprocessing.
*   `model_trainer.py`: Trains and saves the model.
*   `explainer.py`: Creates the DALEX explainer and generates explanations.
*   `app.py`: The main Dash application file.
*   `requirements.txt`: Python package dependencies.
*   `README.md`: This file. 