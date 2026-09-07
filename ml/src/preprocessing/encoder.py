import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataEncoder:
    """
    Handles categorical feature encoding for DataForge.
    """

    def __init__(self):
        self.encoder = None
        self.categorical_columns = []
        self.encoded_columns = []

    def detect_categorical_columns(self, df):
        """
        Detect categorical and boolean columns.
        """

        return df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def fit(self, df):
        """
        Fit the encoder using training data only.
        """

        self.categorical_columns = (
            self.detect_categorical_columns(df)
        )

        if not self.categorical_columns:
            self.encoder = None
            self.encoded_columns = []

            print(
                "No categorical columns found."
            )

            return

        print(
            "Categorical columns detected: "
            f"{self.categorical_columns}"
        )

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        self.encoder.fit(
            df[self.categorical_columns]
        )

        self.encoded_columns = (
            self.encoder.get_feature_names_out(
                self.categorical_columns
            ).tolist()
        )

    def transform(self, df):
        """
        Transform categorical columns using
        the encoder fitted on training data.
        """

        df = df.copy()

        if not self.categorical_columns:
            return df

        if self.encoder is None:
            raise RuntimeError(
                "Encoder has not been fitted."
            )

        missing_columns = [
            column
            for column in self.categorical_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing categorical columns: "
                f"{missing_columns}"
            )

        encoded_data = self.encoder.transform(
            df[self.categorical_columns]
        )

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.encoded_columns,
            index=df.index
        )

        df = df.drop(
            columns=self.categorical_columns
        )

        df = pd.concat(
            [df, encoded_df],
            axis=1
        )

        return df

    def fit_transform(self, df):
        """
        Fit the encoder and transform the data.
        """

        self.fit(df)

        return self.transform(df)