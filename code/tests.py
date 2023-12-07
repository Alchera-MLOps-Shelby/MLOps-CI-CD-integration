# tests.py
import mlflow
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_mlflow_server_running():
    """
    Test if the MLFlow server is running and accessible.
    """
    client = mlflow.tracking.MlflowClient()
    assert client is not None

def test_model_training():
    """
    Test the model training functionality.
    """
    # Load a sample dataset (Iris)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Train a simple RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Log the model to MLFlow (in a real scenario, this would be done in your training script)
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")

    # Add assertions based on your specific model training requirements
    assert accuracy_score(y_test, predictions) > 0.8  # Modify based on your model's expected performance

def test_hyperparameter_tuning():
    """
    Test the hyperparameter tuning functionality.
    """
    # Load a sample dataset (Iris)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Define a simple hyperparameter tuning process (random search for RandomForestClassifier)
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier()
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42)
    search.fit(X_train, y_train)

    # Log the best model to MLFlow (in a real scenario, this would be done in your tuning script)
    with mlflow.start_run():
        mlflow.sklearn.log_model(search.best_estimator_, "best_model")
        mlflow.log_params(search.best_params_)

    # Add assertions based on your specific hyperparameter tuning requirements
    assert search.best_score_ > 0.8  # Modify based on your tuning goals

if __name__ == "__main__":
    # Run the tests when the script is executed directly
    pytest.main(["-v", "tests.py"])

