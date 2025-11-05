# MLOps: CI/CD Pipeline for an Iris Feature Store Project

This project demonstrates a complete CI/CD (Continuous Integration/Continuous Deployment) pipeline for a machine learning project using Feast as a feature store and GitHub Actions for automation.

The core goal of this setup is to ensure code quality, data integrity, and reproducible deployments. The pipeline automatically runs validation tests on every code change, providing immediate feedback and preventing errors from being merged into the main codebase.

## Project Structure

            .
            ├── .github/
            │   └── workflows/
            │       └── ci-workflow.yml       # --> The CI/CD automation workflow
            ├── artifacts/                      # --> Stores trained model artifacts
            ├── data/
            │   ├── iris.csv                  # --> Raw input data
            │   └── iris_prepared.csv         
            ├── feature_repo/
            │   ├── data/
            │   │   ├── online_store.db       # --> (Local) Online feature store
            │   │   └── registry.db           # --> Feast's catalog of features
            │   ├── feature_store.yaml        # --> Feast configuration
            │   └── iris_features.py          # --> Feature definitions 
            ├── tests/
            │   └── test_data_validation.py   # --> Automated data quality tests
            ├── .gitignore                      # --> Specifies files for Git to ignore
            ├── prepare_data.py                 # --> Script to process raw data
            ├── requirements.txt                # --> List of Python dependencies
            └── train.py                        # --> Model training 
            script

## Automation (.github/workflows/)

- ci-workflow.yml
    - Utility: This is the heart of the automation. It defines the CI/CD pipeline that runs on every Pull Request.
    - Process:
        1. Triggers on any pull request to the main branch
        2. Sets up a clean Ubuntu environment with a specific Python version (e.g., 3.10).
        3. Installs all project dependencies from requirements.txt and a separate set of CI tools (pytest, cml).
        4. Runs the prepare_data.py script to generate the processed data file.
        5. Executes the pytest command to run all automated tests found in the tests/ directory.
        6. Uses CML (Continuous Machine Learning) to take the test output and post it as a comment on the Pull Request, providing immediate, visible feedback.

## Testing (tests/)

- test_data_validation.py
    - Utility: An automated test suite written with pytest. Its purpose is to guarantee the quality and integrity of the data before it's used for training.
    - Checks Performed:
        1. Schema Validation: Ensures all required columns are present.
        2. Null Value Check: Fails if any null values are found in the data.
        3. Uniqueness Check: Verifies that every iris_id is unique.
    - Role in CI/CD: This script is automatically run by the ci-workflow.yml. If any of these checks fail, the entire pipeline fails, preventing bad data from moving forward.