import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Get parameters for current model
            param_grid = params.get(model_name, {})

            # If parameters exist → apply GridSearch
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
                gs.fit(x_train, y_train)

                best_model = gs.best_estimator_
            else:
                # No params (like Linear Regression)
                best_model = model
                best_model.fit(x_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(f"{model_name} → Train: {train_score}, Test: {test_score}")

            report[model_name] = test_score

            # IMPORTANT: update model with best version
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)