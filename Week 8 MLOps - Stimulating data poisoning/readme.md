# MLOps Project: CI/CD Pipeline for Iris Model Deployment and Security Analysis

This project demonstrates a complete, end-to-end MLOps workflow. It begins with data preparation and model training, automates testing and deployment through a CI/CD pipeline, and concludes with a security analysis on the impact of data poisoning attacks.

The pipeline integrates several key MLOps tools: **Feast** for feature management, **MLflow** for experiment tracking, **Docker** for containerization, **Kubernetes** for deployment, and **GitHub Actions** for CI/CD automation.

## Key Features & Components

*   **Feature Store (Feast)**: Manages feature definitions (`iris_features.py`) as code, providing a consistent interface for training.
*   **Experiment Tracking (MLflow)**: Automates hyperparameter tuning and logs all experiment parameters, metrics, and model artifacts for reproducibility.
*   **Containerized API (Docker & FastAPI)**: The trained model is served via a high-performance REST API (`api.py`) packaged into a portable Docker container (`Dockerfile`).
*   **Automated Deployment (Kubernetes)**: A Kubernetes manifest (`deployment.yaml`) defines how the API is deployed, scaled, and exposed to the internet.
*   **CI/CD Pipeline (GitHub Actions)**:
    *   **Continuous Integration (`ci-workflow.yml`)**: On every Pull Request, this workflow automatically runs data validation and model training tests to ensure code quality. Test results are posted as a comment using CML.
    *   **Continuous Deployment (`cd.yml`)**: On a merge to the `main` branch, this workflow automatically builds the Docker image, pushes it to Docker Hub, and deploys the new version to a GKE (Google Kubernetes Engine) cluster.
*   **Security Analysis (Data Poisoning)**: Includes a utility (`poison_data.py`) to simulate data poisoning attacks and evaluate their impact on model performance.

---

## Project Workflow

1.  **Data Preparation (`prepare_data.py`)**: Cleans the raw `iris.csv`, adds required IDs and timestamps, and saves `iris_prepared.csv`.
2.  **Model Training (`train.py`)**: Retrieves features from Feast, performs hyperparameter tuning with `GridSearchCV`, and logs the entire experiment to MLflow.
3.  **CI Pipeline (on Pull Request)**:
    *   `ci-workflow.yml` is triggered.
    *   Runs tests from `tests/test_data_validation.py`.
    *   Posts a CML report on the PR.
4.  **CD Pipeline (on Merge to `main`)**:
    *   `cd.yml` is triggered.
    *   Builds a Docker image using `Dockerfile`.
    *   Pushes the image to Docker Hub.
    *   Deploys the image to Kubernetes using `deployment.yaml`.
    *   Posts a deployment confirmation comment on the merge commit.

---

## Data Poisoning Analysis

This project includes a specific module to analyze the model's vulnerability to data poisoning attacks, a critical aspect of ML security.

*   **`poison_data.py`**
    *   **Utility**: This script simulates a data poisoning attack. It loads the clean Iris dataset and replaces the feature values in a random subset of rows (at 5%, 10%, or 50% levels) with random noise.

### Validation Outcomes

Training the model on these poisoned datasets reveals a clear degradation in performance:
*   **Baseline (0% Poisoning):** High accuracy (~97%), indicating the model learns effective patterns from clean data.
*   **5%-10% Poisoning:** A significant drop in accuracy. The model's decision boundaries are skewed by the noisy data, reducing its ability to generalize.
*   **50% Poisoning:** Catastrophic performance, with accuracy dropping to near-random chance (~33%). The signal in the data is overwhelmed by noise, making it impossible for the model to learn.

### Mitigating Poisoning Attacks

A layered defense is required to protect against such attacks:
1.  **Input Validation:** Implement strict rule-based filtering (e.g., values must be within a realistic range) and statistical outlier detection (e.g., using Z-scores or Isolation Forests) to reject malicious data before it enters the training set.
2.  **Robust Modeling:** Use ensemble models like Random Forests. Their voting mechanism is inherently more resistant to the influence of a small percentage of outliers.
3.  **MLOps Governance:** Maintain versioned, immutable "golden" datasets for training. If an attack is detected, the model can be quickly rolled back and retrained on a known-good data snapshot.

---

## How to Run

### 1. Local Training
```bash
# Prepare the initial dataset
python prepare_data.py

# Run the training, tuning, and MLflow tracking
python train.py

# View the experiment results
mlflow ui