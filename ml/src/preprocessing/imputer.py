import pandas as pd
from sklearn.impute import SimpleImputer


class DataImputer:
    """
    Handles missing values for DataForge.

    The imputer is fitted only on training data and
    then used to transform both training and test data.
    """

    def __init__(self):
        self.numerical_imputer = None
        self.categorical_imputer = None

        self.numerical_columns = []
        self.categorical_columns = []

    def detect_columns(self, df):
        """
        Detect numerical and categorical columns.
        """

        self.numerical_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        self.categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def fit(self, df):
        """
        Fit imputers using training data only.
        """

        self.detect_columns(df)

        if self.numerical_columns:
            self.numerical_imputer = SimpleImputer(
                strategy="median"
            )

            self.numerical_imputer.fit(
                df[self.numerical_columns]
            )

        if self.categorical_columns:
            self.categorical_imputer = SimpleImputer(
                strategy="most_frequent"
            )

            self.categorical_imputer.fit(
                df[self.categorical_columns]
            )

        print("\nDataForge Missing Value Handling")

        if self.numerical_columns:
            print(
                "Numerical columns: "
                f"{self.numerical_columns}"
            )

        if self.categorical_columns:
            print(
                "Categorical columns: "
                f"{self.categorical_columns}"
            )

    def transform(self, df):
        """
        Apply fitted imputers to new data.
        """

        df = df.copy()

        if self.numerical_imputer is not None:
            df[self.numerical_columns] = (
                self.numerical_imputer.transform(
                    df[self.numerical_columns]
                )
            )

        if self.categorical_imputer is not None:
            df[self.categorical_columns] = (
                self.categorical_imputer.transform(
                    df[self.categorical_columns]
                )
            )

        return df

    def fit_transform(self, df):
        """
        Fit the imputers and transform the training data.
        """

        self.fit(df)

        return self.transform(df)