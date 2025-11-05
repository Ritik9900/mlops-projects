# train.py (Modified to accept a data path as an argument)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os
import argparse # Import the argparse library

# --- Define constants for file paths ---
MODEL_DIR = 'artifacts'
MODEL_NAME = 'model.joblib'
METRICS_FILE = 'metrics.txt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
# The default data path is no longer a constant, but a default for our argument.

def train_and_evaluate(data_path): # The function now accepts the data path as an argument
    """
    This function encapsulates the entire model training and evaluation process.
    It reads data from a specified CSV, trains a model, evaluates it,
    saves the artifacts, and returns the accuracy.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading training data from {data_path}...")
    try:
        # Load the data from the path provided to the function
        training_df = pd.read_csv(data_path)
        print(f"Found {len(training_df)} records.")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: {data_path} not found. "
            "Please ensure the data file exists at this path."
        )

    print("\n--- Training Data Sample ---")
    print(training_df.head())
    print("----------------------------\n")

    print("Splitting data...")
    # --- IMPORTANT FIX for poisoned data ---
    # The poisoned data might have random strings in the 'species' column if it was also affected.
    # We must handle potential errors during mapping.
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    # Create the encoded column, coercing errors to NaN (Not a Number)
    training_df['species_encoded'] = training_df['species'].map(species_map)
    
    # Drop rows where the species was not one of the three expected values
    # This cleans up any rows where the target variable itself might have been poisoned
    training_df.dropna(subset=['species_encoded'], inplace=True)
    training_df['species_encoded'] = training_df['species_encoded'].astype(int)
    # ------------------------------------

    X = training_df.drop(columns=['iris_id', 'event_timestamp', 'species', 'species_encoded'])
    y = training_df['species_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    print("Data split complete.")

    print("Training Decision Tree model...")
    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    print("Model training complete.")

    print("Evaluating model...")
    prediction = mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f'The accuracy of the Decision Tree is: {accuracy:.3f}')

    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
    print("Metrics saved.")

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(mod_dt, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

    return accuracy


if __name__ == "__main__":
    # --- Add command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Train an Iris model on specified data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/iris_prepared.csv", # The default value if no path is provided
        help="Path to the training data CSV file."
    )
    args = parser.parse_args()
    
    print(f"Running train.py as a script with data from: {args.data_path}")
    train_and_evaluate(args.data_path) # Pass the path to the function
    print("\ntrain.py script finished.")