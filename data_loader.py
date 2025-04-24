"""
Loads and preprocesses the dataset.
"""
import pandas as pd
import re

# --- Define rename map at module level for broader access --- #
RENAME_MAP = {
    'mean_med_diesel_crack_input1_trade_month_lag2': 'diesel_crack_m_lag2',
    'mean_nwe_hsfo_crack_trade_month_lag1': 'hsfo_nwe_crack_m_lag1',
    'mean_nwe_lsfo_crack_trade_month': 'lsfo_nwe_crack_m',
    'mean_nwe_ulsfo_crack_trade_month_lag3': 'ulsfo_nwe_crack_m_lag3',
    'mean_sing_gasoline_vs_vlsfo_trade_month': 'gas_vs_vlsfo_sing_m',
    'mean_sing_vlsfo_crack_trade_month_lag3': 'vlsfo_crack_sing_m_lag3',
    'new_sweet_sr_margin': 'sweet_sour_margin',
    # Add other mappings if needed
}

def clean_col_names(df):
    """Cleans column names by replacing special characters and spaces, then renames."""
    new_cols = []
    for col in df.columns:
        # Replace spaces and special characters with underscores
        new_col = re.sub(r'[^0-9a-zA-Z_]+', '_', col)
        # Remove trailing underscores
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols

    # Apply renaming using the module-level map
    df = df.rename(columns=RENAME_MAP)
    print("Renamed columns for brevity:", df.columns.tolist())

    return df

def load_data(filepath, target_col='target'):
    """
    Loads data from a CSV file, parses dates, cleans column names,
    and separates features and target.

    Args:
        filepath (str): Path to the CSV file.
        target_col (str): Name of the target column.

    Returns:
        tuple: A tuple containing (X, y, df), where X is the feature DataFrame
               (potentially excluding date/target), y is the target Series,
               and df is the DataFrame with cleaned columns and parsed dates.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None

    original_date_col = None
    if 'date' in df.columns:
        original_date_col = 'date' # Store original name before cleaning
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(f"Warning: Could not parse 'date' column: {e}")

    # Clean column names (includes renaming via RENAME_MAP)
    df = clean_col_names(df)

    # Find the cleaned+renamed target column using module-level RENAME_MAP
    target_col_cleaned = RENAME_MAP.get(target_col, target_col)
    if target_col_cleaned not in df.columns:
        print(f"Error: Target column '{target_col_cleaned}' (cleaned/renamed) not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return None, None, None

    # Find the cleaned+renamed date column name using module-level RENAME_MAP
    cleaned_date_col = None
    if original_date_col:
        cleaned_date_col_potential = re.sub(r'[^0-9a-zA-Z_]+', '_', original_date_col).strip('_')
        cleaned_date_col_renamed = RENAME_MAP.get(cleaned_date_col_potential, cleaned_date_col_potential)
        if cleaned_date_col_renamed in df.columns:
            cleaned_date_col = cleaned_date_col_renamed
            print(f"Identified date column as: {cleaned_date_col}")

    # Separate features (X) and target (y)
    cols_to_drop_for_X = [target_col_cleaned]
    if cleaned_date_col and cleaned_date_col != target_col_cleaned:
        cols_to_drop_for_X.append(cleaned_date_col)

    X = df.drop(columns=cols_to_drop_for_X, errors='ignore')
    y = df[target_col_cleaned]

    print(f"Data loaded. Features (X) shape: {X.shape}, Target (y) shape: {y.shape}, DF shape: {df.shape}")
    # Return df with cleaned+renamed columns, X with only features, y as target
    return X, y, df

# Example usage (optional, for testing)
if __name__ == '__main__':
    data_path = '_data/preem.csv'
    X, y, df_loaded = load_data(data_path)
    if X is not None:
        print("\nFeatures (X):")
        print(X.head())
        print("\nTarget (y):")
        print(y.head())
        print("\nCleaned DataFrame Info:")
        print(df_loaded.info()) 