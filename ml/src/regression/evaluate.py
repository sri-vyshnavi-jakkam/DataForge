import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class RegressionEvaluator:
    """
    Evaluates regression models for DataForge.
    """

    def __init__(self, trained_models):
        self.trained_models = trained_models
        self.results = {}
        self.failed_models = {}

    def evaluate(self, X_test, y_test):
        """
        Evaluate all successfully trained regression models.

        If one model fails during evaluation, the remaining
        models continue to be evaluated.
        """

        self.results = {}
        self.failed_models = {}

        for name, model in self.trained_models.items():

            print(f"Evaluating {name}...")

            try:
                y_pred = model.predict(
                    X_test
                )

                mae = mean_absolute_error(
                    y_test,
                    y_pred
                )

                mse = mean_squared_error(
                    y_test,
                    y_pred
                )

                rmse = np.sqrt(mse)

                r2 = r2_score(
                    y_test,
                    y_pred
                )

                self.results[name] = {
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "r2_score": r2
                }

                print(
                    f"{name} evaluation completed."
                )

            except Exception as error:
                self.failed_models[name] = str(error)

                print(
                    f"{name} evaluation failed: "
                    f"{error}"
                )

        if not self.results:
            raise RuntimeError(
                "All regression models failed "
                "during evaluation."
            )

        print(
            f"\nSuccessfully evaluated models: "
            f"{len(self.results)}"
        )

        if self.failed_models:
            print(
                f"Failed evaluations: "
                f"{len(self.failed_models)}"
            )

        return self.results

    def get_results_dataframe(self):
        """
        Convert evaluation results into a DataFrame
        sorted by R² score.
        """

        if not self.results:
            return pd.DataFrame()

        results_df = pd.DataFrame(
            self.results
        ).T

        results_df = results_df.sort_values(
            by="r2_score",
            ascending=False
        )

        return results_df