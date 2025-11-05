import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
import argparse

# --- Evidently imports ---
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset


# Define constants for file paths
MODEL_DIR = 'artifacts'
MODEL_NAME = 'model.joblib'
METRICS_FILE = 'metrics.txt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

def train_and_evaluate(data_path):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading training data from {data_path}...")
    training_df = pd.read_csv(data_path)

    print("Splitting data...")
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    training_df['target'] = training_df['species'].map(species_map)

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_col = 'target'

    # Drop rows with missing target values
    training_df.dropna(subset=[target_col], inplace=True)
    training_df[target_col] = training_df[target_col].astype(int)

    # Split into train (reference) and test (current)
    train_df, test_df = train_test_split(
        training_df, test_size=0.4, stratify=training_df[target_col], random_state=42
    )

    print("Training Decision Tree model...")
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(train_df[feature_cols], train_df[target_col])
    print("Model training complete.")

    # Add predictions for Evidently report
    train_df['prediction'] = model.predict(train_df[feature_cols])
    test_df['prediction'] = model.predict(test_df[feature_cols])

    # --- Generate combined Evidently report ---
    print("\nGenerating Evidently AI full report with performance, drift, and quality...")

    full_report = Report(metrics=[
        ClassificationPreset(),  # Model performance metrics + confusion matrix
        DataDriftPreset(),       # Dataset drift detection and visualizations
        DataQualityPreset()      # Data quality metrics
    ])

    full_report.run(
        reference_data=train_df,
        current_data=test_df,
        column_mapping=None
    )

    # --- Generate Evidently test suite ---
    print("Running Evidently tests...")

    test_suite = TestSuite(tests=[
        DataDriftTestPreset(),   # Pass/fail drift tests
        DataQualityTestPreset()  # Pass/fail data quality tests
    ])

    test_suite.run(
        reference_data=train_df,
        current_data=test_df,
        column_mapping=None
    )

    # Save both reports into one HTML file
    merged_report_path = 'evidently_full_dashboard.html'
    with open(merged_report_path, 'w', encoding='utf-8') as f:
        f.write("<h1>Evidently AI Report Dashboard</h1>")
        f.write("<h2>Performance, Drift & Data Quality</h2>")
        f.write(full_report.get_html())
        f.write("<hr>")
        f.write("<h2>Pass/Fail Test Results</h2>")
        f.write(test_suite.get_html())

    print(f"Full dashboard saved to: {merged_report_path}")
    print("Open this file in your browser to see the interactive results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Iris model and generate an Evidently report.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/iris_with_location.csv",
        help="Path to the training data CSV file."
    )
    args = parser.parse_args()

    train_and_evaluate(args.data_path)
    print("\ntrain.py script finished.")
