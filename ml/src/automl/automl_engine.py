from src.classification.models import ClassificationModels
from src.classification.train import ModelTrainer
from src.classification.evaluate import ModelEvaluator

from src.regression.models import RegressionModels
from src.regression.train import RegressionTrainer
from src.regression.evaluate import RegressionEvaluator


class AutoMLEngine:
    """
    Coordinates model training and evaluation for DataForge.
    """

    def __init__(self):
        self.classification_models = ClassificationModels()
        self.regression_models = RegressionModels()

        self.trainer = None
        self.evaluator = None

        self.trained_models = {}
        self.evaluation_results = {}

    def run(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        problem_type
    ):
        """
        Train and evaluate candidate models using
        already-preprocessed training and testing data.
        """

        print("\nDataForge AutoML")

        if problem_type == "Classification":

            models = (
                self.classification_models.get_models()
            )

            print(
                f"Number of candidate models: "
                f"{len(models)}"
            )

            self.trainer = ModelTrainer(
                models
            )

            self.trained_models = (
                self.trainer.train(
                    X_train,
                    y_train
                )
            )

            self.evaluator = ModelEvaluator(
                self.trained_models
            )

            self.evaluation_results = (
                self.evaluator.evaluate(
                    X_test,
                    y_test
                )
            )

        elif problem_type == "Regression":

            models = (
                self.regression_models.get_models()
            )

            print(
                f"Number of candidate models: "
                f"{len(models)}"
            )

            self.trainer = RegressionTrainer(
                models
            )

            self.trained_models = (
                self.trainer.train(
                    X_train,
                    y_train
                )
            )

            self.evaluator = RegressionEvaluator(
                self.trained_models
            )

            self.evaluation_results = (
                self.evaluator.evaluate(
                    X_test,
                    y_test
                )
            )

        else:
            raise ValueError(
                f"Unsupported problem type: "
                f"{problem_type}"
            )

        return {
            "trained_models": self.trained_models,
            "evaluation_results": self.evaluation_results,
            "failed_training_models": (
                self.trainer.failed_models
            ),
            "failed_evaluation_models": (
                self.evaluator.failed_models
            )
        }