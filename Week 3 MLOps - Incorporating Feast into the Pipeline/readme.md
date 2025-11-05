# Feast with BigQuery Demo

This project demonstrates how [Feast](https://feast.dev/) acts as a feature store, bridging a large offline data source (Google BigQuery) with a fast online store (SQLite) for real-time access. It showcases the concept of 'features as code' for defining and managing feature logic in a version-controlled, reusable way.

## Core Concepts

*   **Offline Store**: A data warehouse (BigQuery) that serves as the single source of truth for all historical feature data.
*   **Online Store**: A low-latency database (SQLite) that stores the latest feature values for fast lookups, typically used in production model inference.
*   **Features as Code**: Feature definitions (`iris_features.py`) are written in Python, allowing them to be version-controlled, reviewed, and managed like any other application code.
*   **Feature Registry**: A catalog (`registry.db`) that stores the registered definitions of all available features in the feature store.

## Prerequisites

1.  **Google Cloud SDK**: The `gcloud` and `bq` command-line tools must be installed and configured.
2.  **Google Cloud Project**: An active GCP project with the BigQuery API enabled.
3.  **Authentication**: You must be authenticated with GCP. Run `gcloud auth application-default login`.
4.  **Python 3.8+** and `pip`.
5.  **Project Dependencies**: Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Execution Steps

Follow these steps to run the complete demonstration.

### 1. Configure GCP Project

Set the `GOOGLE_CLOUD_PROJECT` environment variable to your GCP Project ID.

```bash
# Replace with your actual Google Cloud Project ID
export GOOGLE_CLOUD_PROJECT="velvety-rookery-461404-b5"

# Verify the variable is set
echo $GOOGLE_CLOUD_PROJECT
```

### 2. Prepare and Load Data into BigQuery

These commands prepare the local Iris CSV data by adding the required iris_id and event_timestamp columns, create a dataset in BigQuery, and upload the data to serve as our offline store.

```bash
# Add required ID and timestamp columns to the raw data
python prepare_data.py

# Create a new dataset in BigQuery
bq mk --dataset $GOOGLE_CLOUD_PROJECT:iris_feast_dataset

# Upload the prepared CSV to a new BigQuery table
bq load \
    --autodetect \
    --source_format=CSV \
    "$GOOGLE_CLOUD_PROJECT:iris_feast_dataset.iris_data" \
    ./data/iris_prepared.csv
```

### 3. Deploy Feature Definitions

Navigate into the feature_repo and apply the feature definitions. This command reads the Python files (iris_features.py) and registers the entities and feature views in the Feast registry (data/registry.db).

```bash
cd feature_repo
feast apply
```

### 4. Materialize Features to the Online Store

This command loads the latest feature values from the BigQuery offline store into the SQLite online store (data/online_store.db). This makes the features available for low-latency lookups.

```bash
# The timestamp tells Feast to look for features that are new since the last run
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### 5. Train a Model Using the Feature Store

Run the training script. This script demonstrates the power of Feast's get_historical_features method, which abstracts away the complexity of joining data from BigQuery, providing a simple, point-in-time correct training dataset.

```bash
Python train.py
```
