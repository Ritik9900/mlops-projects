# How to Load Test the API with Locust

This guide explains how to use Locust, a powerful and easy-to-use load testing tool, to simulate high traffic against our deployed Iris Prediction API. This allows us to understand the application's performance limits and identify potential bottlenecks.Step 1: The locustfile.pyThe locustfile.py is the heart of our load test. It's a Python script that defines the behavior of our "virtual users."from locust import HttpUser, task, between
import random

class IrisApiUser(HttpUser):

    # Each virtual user will wait 1 to 2 seconds between requests
    wait_time = between(1, 2)

    @task
    def predict_endpoint(self):
        # Define a sample payload with random data for the prediction
        payload = {
          "sepal_length": round(random.uniform(4.0, 8.0), 1),
          "sepal_width":  round(random.uniform(2.0, 4.5), 1),
          "petal_length": round(random.uniform(1.0, 7.0), 1),
          "petal_width":  round(random.uniform(0.1, 2.5), 1)
        }
        
        # Send a POST request to the /predict endpoint
        self.client.post("/predict", json=payload)
IrisApiUser: This class represents a single user.wait_time: This tells each user to pause for a random duration between 1 and 2 seconds after completing a task, which simulates more realistic user behavior.@task: This decorator marks the predict_endpoint function as a task that each user will repeatedly execute.self.client.post(...): This is the action the user performs—sending a POST request with a randomized, valid JSON payload to our /predict endpoint.Step 2: Running the Load TestYou can run the load test from your local machine or from the Google Cloud Shell.Install Locust:If you haven't already, install the Locust library using pip.
pip install locust
Start the Locust Server:Run the following command in your terminal. You must replace <YOUR_EXTERNAL_IP> with the actual external IP address of your GKE service.locust -f locustfile.py --host=http://<YOUR_EXTERNAL_IP>
Open the Locust Web UI:After running the command, open a web browser and navigate to http://localhost:8089.Step 3: Configuring and Starting the TestYou will see the Locust web interface, where you can configure the load test.Number of users: Enter the total number of concurrent users you want to simulate (e.g., 100).Spawn rate: This is the number of users to start per second (e.g., 10).Start swarming: Click this button to begin the test.Locust will start "swarming" your API, and you can watch the performance statistics in real-time on the "Statistics" tab.Step 4: Monitoring Application Logs During the TestWhile the Locust test is running, it's crucial to watch the logs from your application pods to see how they are behaving under stress. This is where you will find any application-level errors or tracebacks.Navigate to Logs Explorer:In the Google Cloud Console, go to Logging -> Logs Explorer.Build the Query:Use the query builder to filter for the logs from your specific deployment:Resource Type: Kubernetes ContainerCluster Name: iris-clusterDeployment Name: iris-api-deploymentYour final query will look like this:resource.type="k8s_container"
resource.labels.cluster_name="iris-cluster"
resource.labels.deployment_name="iris-api-deployment"
Run Query and Observe:Run the query. You will see a live, combined log stream from all your running pods. This is the best way to catch errors and see the detailed output from your api.py script as requests are being processed.