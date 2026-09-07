import pandas as pd


class DataStatistics:
    """
    Generates basic statistical insights for DataForge.
    """

    def get_basic_statistics(self, df):
        if df.empty:
            raise ValueError("Dataset is empty.")

        statistics = df.describe(
            include="all"
        ).transpose()

        return statistics

    def get_missing_values(self, df):
        missing_values = df.isnull().sum()

        missing_values = missing_values[
            missing_values > 0
        ].sort_values(
            ascending=False
        )

        return missing_values

    def get_dataset_info(self, df):
        numerical_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isnull().sum().sum())
        }

    def analyze(self, df):
        print("\nDataForge Data Insights")

        info = self.get_dataset_info(df)

        print(f"Rows: {info['rows']}")
        print(f"Columns: {info['columns']}")
        print(
            f"Numerical columns: "
            f"{len(info['numerical_columns'])}"
        )
        print(
            f"Categorical columns: "
            f"{len(info['categorical_columns'])}"
        )
        print(
            f"Duplicate rows: "
            f"{info['duplicate_rows']}"
        )
        print(
            f"Missing values: "
            f"{info['missing_values']}"
        )

        return {
            "dataset_info": info,
            "statistics": self.get_basic_statistics(df),
            "missing_values": self.get_missing_values(df)
        }