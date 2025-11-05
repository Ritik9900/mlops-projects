# MLOps: CI/CD for Iris Classifier on GCP

This project demonstrates a complete MLOps workflow for training, containerizing, and deploying a scikit-learn model that classifies Iris flower species. The entire process is automated using a Continuous Integration/Continuous Deployment (CI/CD) pipeline with GitHub Actions, Docker, and Google Kubernetes Engine (GKE).

## Project Overview

The goal of this project is to build a robust, automated system that takes a machine learning model from training to a live, scalable production endpoint.

The key components are:
* **Model Training:** A simple Decision Tree Classifier is trained on the Iris dataset.
* **API:** A FastAPI server provides a RESTful API to serve predictions from the trained model.
* **Containerization:** The FastAPI application and the trained model are packaged into a Docker container for portability and scalability.
* **CI/CD Automation:** A GitHub Actions workflow automates the entire process of building the Docker image and deploying it to a GKE cluster on every push to the `main` branch.
* **Infrastructure as Code:** The Kubernetes deployment and service are defined declaratively in a `deployment.yaml` file, ensuring consistent and repeatable infrastructure setup.

## Technology Stack

* **Cloud Platform:** Google Cloud Platform (GCP)
* **Container Orchestration:** Google Kubernetes Engine (GKE) - Autopilot Mode
* **CI/CD:** GitHub Actions
* **Container Registry:** Docker Hub
* **ML Framework:** Scikit-learn
* **API Framework:** FastAPI
* **Containerization:** Docker

## File Structure

.├── .github/workflows/│   └── ci-workflow.yml      # GitHub Actions workflow definition├── artifacts/│   └── model.joblib         # The trained and serialized scikit-learn model├── data/│   ├── iris.csv             # Raw dataset│   └── iris_prepared.csv    # Processed data with timestamps├── feature_repo/            # (Legacy) Feast feature store definition├── tests/                   # (Legacy) Pytest tests├── api.py                   # FastAPI application to serve the model├── deployment.yaml          # Kubernetes Deployment and Service manifest├── Dockerfile               # Instructions to build the application container├── prepare_data.py          # Script to pre-process the raw data├── requirements.txt         # Python dependencies└── train.py                 # Script to train the model and save the artifact
## How It Works: The CI/CD Pipeline

The `.github/workflows/ci-workflow.yml` file defines the automated pipeline, which consists of the following major steps:

1.  **Checkout Code:** The workflow begins by checking out the latest version of the code from the repository.
2.  **Build and Push Docker Image:**
    * It logs into Docker Hub using credentials stored in GitHub Secrets.
    * It builds a Docker image using the `Dockerfile`. This image contains the FastAPI application, the trained `model.joblib`, and all necessary Python libraries.
    * The newly built image is pushed to Docker Hub, tagged with the unique Git commit SHA for versioning.
3.  **Authenticate with GCP:**
    * The workflow authenticates with Google Cloud using a service account key stored as a GitHub Secret.
4.  **Deploy to GKE:**
    * It connects to the `iris-cluster` on GKE.
    * It dynamically replaces the `IMAGE_PLACEHOLDER` in `deployment.yaml` with the new Docker image tag.
    * It applies the updated `deployment.yaml` to the cluster, which triggers a rolling update to the new version of the application.
5.  **Post-Deployment Notification:**
    * Using CML (Continuous Machine Learning), it posts a success comment on the triggering commit to confirm that the deployment was successful.

## How to Use the Deployed API

The API is exposed to the internet via a Kubernetes LoadBalancer Service.

### Endpoint

* **URL:** `http://<YOUR_EXTERNAL_IP>/predict`
* **Method:** `POST`

### Request Body

The API expects a JSON object with the four features of the Iris flower.

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Example curl RequestReplace 34.56.169.109 with the actual external IP address of your service.curl -X POST "[http://34.56.169.109/predict](http://34.56.169.109/predict)" \
-H "Content-Type: application/json" \
-d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
Successful ResponseA successful request will return the predicted class index and the corresponding species name.{
  "prediction": 0,
  "predicted_species": "setosa"
}
Local Development and TrainingTo train a new model locally (e.g., in Google Cloud Shell):Install Dependencies:pip install -r requirements.txt
Run the Training Script:This will create a new artifacts/model.joblib file.python train.py
Commit and Push:Commit the new model and the updated train.py script to trigger a new deployment.git add .
git add --force artifacts/model.joblib
git commit -m "feat: Retrain model with new parameters"
git push
