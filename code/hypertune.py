# hypertune.py
import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def hyperparameter_tuning():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier()

    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42)
    search.fit(X_train, y_train)

    # Log the best hyperparameters to MLFlow
    with mlflow.start_run():
        mlflow.log_params(search.best_params_)

if __name__ == "__main__":
    hyperparameter_tuning()
