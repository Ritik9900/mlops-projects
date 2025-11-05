import pandas as pd
import feast
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os


MODEL_DIR = 'artifacts'
MODEL_NAME = 'best_model.joblib' 
METRICS_FILE = 'metrics.txt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
FEATURE_REPO_PATH = 'feature_repo/'
MLFLOW_EXPERIMENT_NAME = "iris_decision_tree_tuning" 

def train_and_evaluate():
    """
    This function now includes hyperparameter tuning with GridSearchCV
    and experiment tracking with MLflow.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Connecting to Feast Feature Store...")
    fs = feast.FeatureStore(repo_path=FEATURE_REPO_PATH)
    
    print("Loading entity list...")
    entity_df = pd.read_csv("data/iris_prepared.csv")[['iris_id', 'event_timestamp']]
    entity_df['event_timestamp'] = pd.to_datetime(entity_df['event_timestamp'])

    print("Retrieving historical features...")
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "iris_features:sepal_length", "iris_features:sepal_width",
            "iris_features:petal_length", "iris_features:petal_width",
            "iris_target:species"
        ],
    ).to_df()

    X = training_df.drop(columns=['iris_id', 'event_timestamp', 'species'])
    y = training_df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    
    
    print(f"Setting up MLflow experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    
    print("Setting up hyperparameter grid search...")
    
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    print("Running grid search...")
    grid_search.fit(X_train, y_train)

    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    
    with mlflow.start_run():
        print("Logging experiment results to MLflow...")

        mlflow.log_params(grid_search.best_params_)

        prediction = best_model.predict(X_test)
        accuracy = metrics.accuracy_score(prediction, y_test)
        
        mlflow.log_metric("accuracy", accuracy)
        print(f"Best model accuracy: {accuracy:.3f}")

        mlflow.sklearn.log_model(best_model, "model")

        mlflow.set_tag("model_type", "DecisionTreeClassifier")

    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")

    print(f"Saving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)

    print("Training, tuning, and tracking complete.")
    return accuracy

if __name__ == "__main__":
    train_and_evaluate()