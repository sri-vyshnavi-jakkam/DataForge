import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataScaler:
    """
    Handles numerical feature scaling for DataForge.
    """

    def __init__(self):
        self.scaler = None
        self.numerical_columns = []

    def detect_numerical_columns(self, df):
        """
        Detect numerical columns.
        """

        return df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    def fit(self, df, columns=None):
        """
        Fit the scaler using training data only.
        """

        if columns is None:
            self.numerical_columns = (
                self.detect_numerical_columns(df)
            )
        else:
            self.numerical_columns = list(columns)

        if not self.numerical_columns:
            self.scaler = None

            print(
                "No numerical columns found."
            )

            return

        missing_columns = [
            column
            for column in self.numerical_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Numerical columns not found: "
                f"{missing_columns}"
            )

        print(
            "Numerical columns to scale: "
            f"{self.numerical_columns}"
        )

        self.scaler = StandardScaler()

        self.scaler.fit(
            df[self.numerical_columns]
        )

    def transform(self, df):
        """
        Scale numerical columns using the scaler
        fitted on training data.
        """

        df = df.copy()

        if not self.numerical_columns:
            return df

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        missing_columns = [
            column
            for column in self.numerical_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Numerical columns not found: "
                f"{missing_columns}"
            )

        df[self.numerical_columns] = (
            self.scaler.transform(
                df[self.numerical_columns]
            )
        )

        return df

    def fit_transform(self, df, columns=None):
        """
        Fit the scaler and transform the data.
        """

        self.fit(
            df,
            columns
        )

        return self.transform(df)