import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluator:
    """
    Generic model evaluation class for DataForge.
    """

    def __init__(self, trained_models):
        self.trained_models = trained_models
        self.results = {}
        self.failed_models = {}

    def evaluate(self, X_test, y_test):
        """
        Evaluate all successfully trained classification models.

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

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                self.results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
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
                "All classification models failed "
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
        sorted by F1-score.
        """

        if not self.results:
            return pd.DataFrame()

        results_df = pd.DataFrame(
            self.results
        ).T

        results_df = results_df.sort_values(
            by="f1_score",
            ascending=False
        )

        return results_df