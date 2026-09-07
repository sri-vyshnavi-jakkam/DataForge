import os
import joblib


class ModelPersistence:
    """
    Saves and loads the complete trained DataForge model package.
    """

    def save(
        self,
        model,
        imputer,
        encoder,
        scaler,
        problem_info,
        feature_columns,
        path="models/dataforge_model.pkl"
    ):
        """
        Save the model and preprocessing information
        required for future predictions.
        """

        model_package = {
            "model": model,
            "imputer": imputer,
            "encoder": encoder,
            "scaler": scaler,
            "problem_info": problem_info,
            "feature_columns": feature_columns
        }

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        joblib.dump(
            model_package,
            path
        )

        print(
            f"Complete model package saved: {path}"
        )

        return path

    def load(self, path="models/dataforge_model.pkl"):
        """
        Load a previously saved DataForge model package.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Saved model package not found: {path}"
            )

        model_package = joblib.load(path)

        required_items = [
            "model",
            "imputer",
            "encoder",
            "scaler",
            "problem_info",
            "feature_columns"
        ]

        missing_items = [
            item
            for item in required_items
            if item not in model_package
        ]

        if missing_items:
            raise ValueError(
                "Saved model package is incomplete. "
                f"Missing: {missing_items}"
            )

        print(
            f"Complete model package loaded: {path}"
        )

        return model_package