# MLOps Project: Model Fairness and Explainability Analysis

This project extends a core MLOps pipeline to include critical aspects of Responsible AI: **fairness analysis** and **model explainability**.

The primary goal is to demonstrate how to simulate and analyze potential model bias by introducing a sensitive attribute. We use **Evidently AI**, a powerful open-source library, to generate comprehensive, interactive dashboards that evaluate model performance, data drift, and fairness across different subgroups.

## Key Features & Enhancements

*   **Sensitive Attribute Simulation**: A new script (`add_sensitive_feature.py`) is introduced to add a random binary `location` column to the Iris dataset. This serves as a proxy for a real-world sensitive attribute (e.g., gender, race, location) to facilitate fairness analysis.
*   **Automated Reporting with Evidently AI**: The training pipeline (`train.py`) is integrated with Evidently AI to automatically generate a detailed HTML report. This single dashboard provides insights into:
    *   **Model Performance**: Standard classification metrics (Accuracy, Precision, Recall, F1-Score) and a confusion matrix.
    *   **Fairness Analysis**: Performance metrics are broken down by the sensitive `location` attribute, allowing for a direct comparison of how the model performs for different subgroups.
    *   **Data Drift**: Statistical tests and visualizations to check if the distribution of the test data has shifted significantly from the training data.
    *   **Data Quality**: An overview of the dataset's health, including feature statistics and correlations.
*   **Model Explainability (Conceptual)**: This project addresses model explainability by interpreting what a SHAP (SHapley Additive exPlanations) plot would reveal about the model's decision-making process.

---

## Project Workflow

1.  **Introduce Sensitive Feature (`add_sensitive_feature.py`)**:
    *   This script is run first to generate the `data/iris_with_location.csv` dataset.
    *   It adds a `location` column with random `0`s and `1`s to the prepared Iris data.

2.  **Train and Analyze (`train.py`)**:
    *   This is the main script for this analysis. It takes the dataset with the sensitive feature as input.
    *   It splits the data into training and testing sets.
    *   It trains a standard `DecisionTreeClassifier` model.
    *   It then uses **Evidently AI** to generate a comprehensive `evidently_full_dashboard.html` file, comparing the training (reference) and testing (current) data.

---

