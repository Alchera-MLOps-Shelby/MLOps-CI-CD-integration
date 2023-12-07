# tests.py
import mlflow
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, params):
    """
    Train a RandomForestClassifier with specified hyperparameters.
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def test_model():
    """
    Test the trained model using hyperparameters logged in MLFlow.
    """
    # Load a sample dataset (Iris)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Load the best hyperparameters from MLFlow
    with mlflow.start_run():
        run = mlflow.search_runs(order_by=["accuracy"], filter_string="tags.mlflow.runName = 'Hyperparameter Tuning'").iloc[0]
        best_params = run.params

    # Train a model with the best hyperparameters
    best_model = train_model(X_train, y_train, best_params)

    # Make predictions
    predictions = best_model.predict(X_test)

    # Add assertions based on your specific requirements
    assert accuracy_score(y_test, predictions) > 0.8  # Modify based on your model's expected performance

if __name__ == "__main__":
    # Run the tests when the script is executed directly
    pytest.main(["-v", "tests.py"])
