# Model Studio - Time Series Dashboard

This project provides an interactive dashboard built with Dash and Plotly for exploring and understanding time series models using DALEX explainers.

## Features

*   Load time series data and a corresponding model.
*   Visualize model explanations using:
    *   Partial Dependence Plots (PDP)
    *   Accumulated Local Effects (ALE) Plots
*   Display fitted vs. actual time series values.
*   Interactive controls for selecting features and data points.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd model-studio
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

Ensure your data file (`data/dummy_data.csv` by default) and your model (`data/model.pkl` by default) are in the `data/` directory.

Then, run the Dash application:

```bash
python app.py
```

The dashboard will be available at http://127.0.0.1:8050/ by default.

## Data

The application expects:
*   A CSV file containing the time series data, including features used by the model and the target variable.
*   A date/time column that can be parsed by pandas.
*   A pickled model file (`.pkl`) compatible with the DALEX explainer.

Modify the `DATA_FILE_PATH`, `MODEL_FILE_PATH`, and `DATE_COLUMN` variables in `app.py` if your file names or date column differ. 