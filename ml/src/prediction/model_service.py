from src.automl.model_persistence import ModelPersistence
from src.prediction.predictor import Predictor


class ModelService:
    """
    Provides a simple interface for loading and using
    the saved DataForge model.
    """

    def __init__(
        self,
        model_path="models/dataforge_model.pkl"
    ):
        self.model_path = model_path
        self.predictor = None
        self.model_package = None

    def load_model(self):
        """
        Load the saved DataForge model package.
        """

        persistence = ModelPersistence()

        self.model_package = persistence.load(
            self.model_path
        )

        self.predictor = Predictor(
            model=self.model_package["model"],
            imputer=self.model_package["imputer"],
            encoder=self.model_package["encoder"],
            scaler=self.model_package["scaler"],
            feature_columns=self.model_package[
                "feature_columns"
            ]
        )

        return self.predictor

    def predict(self, data):
        """
        Generate predictions using the saved DataForge model.
        """

        if self.predictor is None:
            self.load_model()

        return self.predictor.predict(
            data
        )