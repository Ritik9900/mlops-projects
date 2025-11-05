# Fine-Tuning and Evaluating a Gemini Model for Iris Classification

## Project Overview

This project demonstrates a complete MLOps pipeline for fine-tuning a generative model (Google Gemini) for a classic machine learning task: Iris flower classification. The goal is to replace a traditional classification model with a cost-efficient, fine-tuned Large Language Model (LLM) and rigorously evaluate its performance against the base model.

The pipeline covers data preparation, supervised fine-tuning on Vertex AI, model deployment to a real-time endpoint, and comparative performance evaluation using custom Python scripts.

## Features

-   **Data Preparation**: Scripts to convert tabular CSV data into conversational JSONL format suitable for LLM fine-tuning.
-   **Vertex AI Fine-Tuning**: Leverages Google Cloud's Vertex AI for supervised fine-tuning of a Gemini Pro model.
-   **Vertex AI Endpoint Deployment**: Deploys the fine-tuned model to a scalable, low-latency endpoint for real-time inference.
-   **Comparative Evaluation**: A powerful evaluation script (`evaluate_llm.py`) that compares the performance of the fine-tuned model against the base Gemini model on a test set.
-   **Rich Metrics**: Generates a full classification report (precision, recall, F1-score) and a visual confusion matrix for both models to validate performance improvements.
-   **Containerization**: Includes a `Dockerfile` for containerizing the application or API components.

## Project Architecture

The project follows a standard MLOps workflow orchestrated between a local environment (or Google Cloud Shell) and Google Cloud Platform services:

1.  **Local/Cloud Shell**: Scripts are run to prepare the data.
2.  **Google Cloud Storage (GCS)**: The prepared training data (`.jsonl` format) is uploaded to a GCS bucket to be used as a data source for the fine-tuning job.
3.  **Vertex AI Fine-Tuning**: A supervised fine-tuning job is launched from the Google Cloud Console, using a base Gemini model and the training data from the GCS bucket.
4.  **Vertex AI Model Registry & Endpoints**: The resulting fine-tuned model is automatically registered. From the registry, it is deployed to a Vertex AI Endpoint to serve online predictions.
5.  **Evaluation**: The `evaluate_llm.py` script is run from Cloud Shell. It sends inference requests for each item in the test set to both the base model and the newly deployed fine-tuned endpoint, then compares the results and generates a performance report.

## Setup and Installation (in Google Cloud Shell)

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Authenticate with Google Cloud:**
    ```bash
    gcloud auth login
    gcloud config set project YOUR_GCP_PROJECT_ID
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *A typical `requirements.txt` for this project would contain:*
    ```
    google-cloud-aiplatform
    scikit-learn
    pandas
    seaborn
    matplotlib
    ```

## Usage Workflow

Follow these steps to run the full pipeline from data preparation to evaluation.

### 1. Prepare Data for Fine-Tuning

Run the script to convert the `iris_prepared.csv` file into the required conversational format.

```bash
python prepare_llm_data.py