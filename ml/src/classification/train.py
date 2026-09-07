import os
import joblib


class ModelTrainer:
    """
    Generic model training class for DataForge.
    """

    def __init__(self, models):
        self.models = models
        self.trained_models = {}
        self.failed_models = {}

    def train(self, X_train, y_train):
        """
        Train all provided classification models.

        If one model fails, the remaining models
        continue training.
        """

        self.trained_models = {}
        self.failed_models = {}

        for name, model in self.models.items():

            print(f"Training {name}...")

            try:
                model.fit(
                    X_train,
                    y_train
                )

                self.trained_models[name] = model

                print(
                    f"{name} training completed."
                )

            except Exception as error:
                self.failed_models[name] = str(error)

                print(
                    f"{name} training failed: "
                    f"{error}"
                )

        if not self.trained_models:
            raise RuntimeError(
                "All classification models failed "
                "during training."
            )

        print(
            f"\nSuccessful models: "
            f"{len(self.trained_models)}"
        )

        if self.failed_models:
            print(
                f"Failed models: "
                f"{len(self.failed_models)}"
            )

        return self.trained_models

    def save_models(self, path="models"):
        """
        Save successfully trained models.
        """

        os.makedirs(
            path,
            exist_ok=True
        )

        for name, model in self.trained_models.items():

            file_name = (
                name.lower()
                .replace(" ", "_")
                + ".pkl"
            )

            file_path = os.path.join(
                path,
                file_name
            )

            joblib.dump(
                model,
                file_path
            )

            print(
                f"Saved: {file_path}"
            )