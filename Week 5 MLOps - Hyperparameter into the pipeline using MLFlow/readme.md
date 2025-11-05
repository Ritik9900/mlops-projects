# MLOps Project: CI/CD with Hyperparameter Tuning and Experiment Tracking

This project demonstrates a robust MLOps pipeline for a machine learning model built on the Iris dataset. It showcases a complete workflow from data preparation and feature management with **Feast** to automated testing and model optimization using **GitHub Actions**, **GridSearchCV**, and **MLflow**.

The core goal of this setup is to illustrate how modern MLOps practices can automate not just code validation but also the process of model improvement and experiment management, leading to more reliable and reproducible machine learning systems.

## Key Features

*   **Feature Store**: Uses **Feast** to manage feature definitions as code, bridging offline and online data stores.
*   **Automated CI/CD**: A **GitHub Actions** workflow automatically tests code changes, validates data, and runs the training pipeline on every pull request.
*   **Automated Hyperparameter Tuning**: The training pipeline uses **`GridSearchCV`** from Scikit-learn to automatically search for the best model hyperparameters, moving beyond static model configurations.
*   **Experiment Tracking**: Every training run is logged as an "experiment" using **MLflow**. This captures the model's parameters, performance metrics, and the model artifact itself, creating a complete audit trail.
*   **Automated Reporting**: Test results are posted directly to GitHub Pull Requests via **CML (Continuous Machine Learning)** for immediate feedback.

---

## Core Enhancement: The `train.py` Script

The `train.py` script has been significantly enhanced to serve as the heart of our model optimization and tracking process. It now performs four key functions in a single, automated run:

1.  **Feature Retrieval**: Connects to the **Feast Feature Store** to build a point-in-time correct training dataset.
2.  **Hyperparameter Tuning**:
    *   Defines a `param_grid` containing a range of hyperparameters to test for the `DecisionTreeClassifier` (e.g., `max_depth`, `min_samples_split`).
    *   Uses `GridSearchCV` to systematically train and evaluate the model with every combination of these hyperparameters, identifying the best-performing configuration.
3.  **MLflow Experiment Tracking**:
    *   Initializes an **MLflow experiment**.
    *   Within an `mlflow.start_run()` context, it automatically logs:
        *   **Parameters**: The best hyperparameters found by the grid search (`grid_search.best_params_`).
        *   **Metrics**: The accuracy of the best model on the test set.
        *   **Artifacts**: The final, trained `best_model` object is saved directly to the MLflow run using `mlflow.sklearn.log_model()`. This ensures the model is versioned and can be retrieved later.
4.  **Artifact Persistence**: Saves the best model as a `.joblib` file and the final metrics to a `.txt` file for easy access outside of MLflow.

---

## How to Run and View Experiments

This project now includes a powerful local workflow for experimentation.

### 1. Running a Training and Tuning Experiment

To run the entire pipeline—including feature retrieval, hyperparameter search, and MLflow logging—simply execute the training script from your terminal.

```bash
# Make sure your virtual environment is activated
python train.py