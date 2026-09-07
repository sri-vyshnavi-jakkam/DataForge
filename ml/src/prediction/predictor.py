import re
import pandas as pd


class Predictor:
    """
    Handles predictions using the best model selected by DataForge.
    """

    def __init__(
        self,
        model,
        imputer,
        encoder,
        scaler,
        feature_columns
    ):
        self.model = model
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler
        self.feature_columns = list(feature_columns)

        if not self.feature_columns:
            raise ValueError(
                "Training feature schema is empty."
            )

    def clean_column_names(self, df):
        """
        Apply the same column-name normalization used
        by DataForge during training.
        """

        df = df.copy()

        cleaned_columns = []

        for column in df.columns:
            column = str(column).strip().lower()

            column = re.sub(
                r"[^a-zA-Z0-9]+",
                "_",
                column
            )

            column = re.sub(
                r"_+",
                "_",
                column
            )

            column = column.strip("_")

            if not column:
                column = "column"

            cleaned_columns.append(column)

        unique_columns = []
        column_counts = {}

        for column in cleaned_columns:

            if column not in column_counts:
                column_counts[column] = 0
                unique_columns.append(column)

            else:
                column_counts[column] += 1

                new_name = (
                    f"{column}_{column_counts[column]}"
                )

                while new_name in column_counts:
                    column_counts[column] += 1
                    new_name = (
                        f"{column}_{column_counts[column]}"
                    )

                column_counts[new_name] = 0
                unique_columns.append(new_name)

        df.columns = unique_columns

        return df

    def validate_input(self, data):
        """
        Validate prediction input against the exact
        feature schema stored during training.
        """

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required feature columns: "
                f"{missing_columns}"
            )

        extra_columns = [
            column
            for column in data.columns
            if column not in self.feature_columns
        ]

        if extra_columns:
            print(
                "Ignoring unexpected columns: "
                f"{extra_columns}"
            )

            data = data.drop(
                columns=extra_columns
            )

        return data[
            self.feature_columns
        ]

    def predict(self, data):
        """
        Generate predictions for new data.
        """

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        elif not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Input data must be a dictionary "
                "or pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Prediction input cannot be empty."
            )

        data = self.clean_column_names(
            data
        )

        data = self.validate_input(
            data
        )

        data = self.imputer.transform(
            data
        )

        data = self.encoder.transform(
            data
        )

        data = self.scaler.transform(
            data
        )

        predictions = self.model.predict(
            data
        )

        return predictions