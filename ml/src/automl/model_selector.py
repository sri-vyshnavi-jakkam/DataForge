import os
import joblib


class ModelSelector:
    """
    Selects and saves the best model for DataForge.
    """

    def __init__(
        self,
        metric="f1_score",
        higher_is_better=True
    ):
        self.metric = metric
        self.higher_is_better = higher_is_better

        self.best_model_name = None
        self.best_score = None
        self.best_model = None

    def select_best_model(
        self,
        evaluation_results,
        trained_models=None
    ):
        """
        Select the best model based on evaluation results.
        """

        if not evaluation_results:
            raise ValueError(
                "No evaluation results available."
            )

        first_result = next(
            iter(evaluation_results.values())
        )

        if self.metric not in first_result:
            raise ValueError(
                f"Metric '{self.metric}' is not available."
            )

        if self.higher_is_better:
            self.best_model_name = max(
                evaluation_results,
                key=lambda name:
                evaluation_results[name][self.metric]
            )
        else:
            self.best_model_name = min(
                evaluation_results,
                key=lambda name:
                evaluation_results[name][self.metric]
            )

        self.best_score = (
            evaluation_results[
                self.best_model_name
            ][self.metric]
        )

        if trained_models is not None:

            if self.best_model_name not in trained_models:
                raise ValueError(
                    "Selected best model was not found "
                    "in trained models."
                )

            self.best_model = (
                trained_models[
                    self.best_model_name
                ]
            )

        print(
            f"Best model: "
            f"{self.best_model_name}"
        )

        print(
            f"{self.metric}: "
            f"{self.best_score:.4f}"
        )

        return (
            self.best_model_name,
            self.best_score
        )

    def save_best_model(self, path="models/best_model.pkl"):
        """
        Save the selected best model.
        """

        if self.best_model is None:
            raise RuntimeError(
                "No best model is available to save."
            )

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        joblib.dump(
            self.best_model,
            path
        )

        print(
            f"Best model saved: {path}"
        )

        return path