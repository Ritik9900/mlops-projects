import argparse
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.endpoint import Endpoint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_llm_response(response_text, expected_labels):
    """
    Parses the LLM's text response to extract the predicted class.
    This adds a basic layer of schema validation.
    """
    response_lower = response_text.lower().strip()
    for label in expected_labels:
        if label.lower() in response_lower:
            return label
    return "unknown" # Return a default if no label is found

def evaluate_and_compare(project_id, location, base_model_name, tuned_model_endpoint_id, test_file):
    """
    Evaluate base and tuned model responses against test data,
    providing detailed classification metrics.
    """
    # Init Vertex AI
    vertexai.init(project=project_id, location=location)

    print(f"Using base model: {base_model_name}")
    print(f"Using tuned model endpoint ID: {tuned_model_endpoint_id}")

    # Load test data
    with open(test_file, "r") as f:
        test_data = [json.loads(line) for line in f]

    prompts = [item["contents"][0]["parts"][0]["text"] for item in test_data]
    ground_truth = [item["contents"][1]["parts"][0]["text"] for item in test_data]
    
    # Define the possible labels for schema validation
    expected_labels = sorted(list(set(ground_truth)))

    # Initialize models
    base_model = GenerativeModel(base_model_name)
    tuned_model_endpoint = Endpoint(tuned_model_endpoint_id)

    base_predictions = []
    tuned_predictions = []

    print("\n--- Generating Predictions ---")
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}...")
        
        # --- Base Model Prediction ---
        base_response = base_model.generate_content(prompt).text
        base_pred_label = parse_llm_response(base_response, expected_labels)
        base_predictions.append(base_pred_label)

        # --- Tuned Model Prediction ---
        # The endpoint expects a JSON payload with an "instances" key
        prediction_result = tuned_model_endpoint.predict(instances=[{"content": prompt}])
        # The response structure is prediction.predictions[0]['content']
        tuned_response = prediction_result.predictions[0]['content']
        tuned_pred_label = parse_llm_response(tuned_response, expected_labels)
        tuned_predictions.append(tuned_pred_label)

        # Optional: Print individual results for debugging
        # print(f"  Prompt: {prompt}")
        # print(f"  Expected: {ground_truth[i]}")
        # print(f"  Base Raw: {base_response.strip()} -> Parsed: {base_pred_label}")
        # print(f"  Tuned Raw: {tuned_response.strip()} -> Parsed: {tuned_pred_label}\n")


    print("\n\n--- Evaluation Results ---\n")

    # --- Base Model Evaluation ---
    print("="*20)
    print("  Base Model Report")
    print("="*20)
    base_accuracy = accuracy_score(ground_truth, base_predictions)
    print(f"Accuracy: {base_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(ground_truth, base_predictions, labels=expected_labels))
    
    # --- Tuned Model Evaluation ---
    print("\n" + "="*20)
    print("  Fine-Tuned Model Report")
    print("="*20)
    tuned_accuracy = accuracy_score(ground_truth, tuned_predictions)
    print(f"Accuracy: {tuned_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(ground_truth, tuned_predictions, labels=expected_labels))
    
    # --- Confusion Matrix Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Comparison')

    # Base Model Confusion Matrix
    cm_base = confusion_matrix(ground_truth, base_predictions, labels=expected_labels)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=expected_labels, yticklabels=expected_labels, ax=axes[0])
    axes[0].set_title('Base Model Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Tuned Model Confusion Matrix
    cm_tuned = confusion_matrix(ground_truth, tuned_predictions, labels=expected_labels)
    sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=expected_labels, yticklabels=expected_labels, ax=axes[1])
    axes[1].set_title('Fine-Tuned Model Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("evaluation_comparison.png")
    print("\nSaved confusion matrix comparison to evaluation_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare a base Gemini model with a fine-tuned version.")
    parser.add_argument("--project-id", required=True, help="Your Google Cloud project ID.")
    parser.add-argument("--location", default="us-central1", help="The GCP region for your project.")
    parser.add_argument("--base-model-name", default="gemini-1.0-pro-002", help="The base model name for comparison (e.g., 'gemini-1.0-pro-002').")
    parser.add_argument("--tuned-model-endpoint-id", required=True, help="The numeric ID of the deployed Vertex AI Endpoint for your tuned model.")
    parser.add_argument("--test-file", default="data/iris_conversational_test.jsonl", help="Path to the test JSONL file.")
    args = parser.parse_args()

    evaluate_and_compare(
        args.project_id, args.location,
        args.base_model_name, args.tuned_model_endpoint_id,
        args.test_file
    )