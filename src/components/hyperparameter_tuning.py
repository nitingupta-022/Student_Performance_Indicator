import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class HyperParameterTuningConfig:
    trained_model_file_path = os.path.join("artifacts", "tuned_model.pkl")


class HyperParameterTuning:
    """
    This class is responsible for tuning multiple regression models and saving
    the best performing estimator.
    """

    def __init__(self):
        self.hyperparameter_tuning_config = HyperParameterTuningConfig()

    def get_models_and_params(self):
        models = {
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(random_state=42, objective="reg:squarederror"),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_seed=42),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        }

        params = {
            "Decision Tree": {
                "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "splitter": ["best", "random"],
                "max_depth": [None, 3, 5, 8, 12],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "max_depth": [2, 3, 4, 5],
            },
            "Linear Regression": {},
            "K-Neighbors Regressor": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "XGBRegressor": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "max_depth": [3, 4, 5, 6, 8],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            },
            "CatBoosting Regressor": {
                "depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [30, 50, 100, 200],
            },
            "AdaBoost Regressor": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

        return models, params

    def tune_model(self, model, param_grid, X_train, y_train, X_test, y_test):
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring="r2",
            cv=3,
            verbose=0,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        test_score = r2_score(y_test, predictions)

        return best_model, test_score, search.best_params_

    def initiate_hyperparameter_tuning(self, train_array, test_array):
        """
        Tune all supported models and return the best estimator, score, and params.
        """

        logging.info("Hyperparameter tuning initiated....")

        try:
            logging.info("Splitting train & test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models, params = self.get_models_and_params()

            best_model = None
            best_model_name = None
            best_model_score = float("-inf")
            best_model_params = None

            for model_name, model in models.items():
                logging.info(f"Tuning {model_name}.")

                param_grid = params.get(model_name, {})

                if not param_grid:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    model_score = r2_score(y_test, predictions)
                    model_params = {}
                else:
                    model, model_score, model_params = self.tune_model(
                        model=model,
                        param_grid=param_grid,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )

                logging.info(f"{model_name} achieved R2 score: {model_score}")

                if model_score > best_model_score:
                    best_model = model
                    best_model_name = model_name
                    best_model_score = model_score
                    best_model_params = model_params

            if best_model is None:
                raise CustomException("No model was tuned successfully.", sys)

            logging.info(
                f"Best model found: {best_model_name} with R2 score {best_model_score}"
            )
            logging.info("Saving tuned model into the pickle file.")

            save_object(
                file_path=self.hyperparameter_tuning_config.trained_model_file_path,
                obj=best_model,
            )

            return {
                "best_model_name": best_model_name,
                "best_model": best_model,
                "best_score": best_model_score,
                "best_params": best_model_params,
                "model_path": self.hyperparameter_tuning_config.trained_model_file_path,
            }

        except Exception as e:
            raise CustomException(e, sys)