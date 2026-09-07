import re
import pandas as pd


class DataCleaner:
    """
    Generic data cleaning class for DataForge.

    Handles structural cleaning before model training.
    """

    def load_data(self, file_path):
        """
        Load a CSV file into a pandas DataFrame.
        """

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )
        except Exception as error:
            raise ValueError(
                f"Unable to read CSV file: {error}"
            )

        if df.empty:
            raise ValueError(
                "The uploaded dataset is empty."
            )

        return df

    def clean_column_names(self, df):
        """
        Standardize column names.

        Examples:
            'First Name' -> 'first_name'
            ' Age ' -> 'age'
            'Salary($)' -> 'salary'
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

        # Ensure column names remain unique
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

    def remove_empty_columns(self, df):
        """
        Remove columns containing only missing values.
        """

        df = df.copy()

        empty_columns = df.columns[
            df.isnull().all()
        ].tolist()

        if empty_columns:
            print(
                f"Empty columns removed: "
                f"{empty_columns}"
            )

        df = df.dropna(
            axis=1,
            how="all"
        )

        return df

    def remove_duplicates(self, df):
        """
        Remove completely duplicated rows.
        """

        df = df.copy()

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        print(
            f"Duplicates removed: {removed}"
        )

        return df

    def validate_dataset(self, df):
        """
        Validate the cleaned dataset.
        """

        if df.empty:
            raise ValueError(
                "Dataset is empty after cleaning."
            )

        if df.shape[1] < 2:
            raise ValueError(
                "Dataset must contain at least "
                "one feature column and one target column."
            )

        if any(
            column == ""
            for column in df.columns
        ):
            raise ValueError(
                "Dataset contains an invalid column name."
            )

        return df

    def clean(self, file_path):
        """
        Execute structural data cleaning.

        Missing values are intentionally not imputed here.
        Imputation must happen after train/test splitting
        to prevent data leakage.
        """

        df = self.load_data(file_path)

        print("\nDataForge Data Cleaning")
        print(
            f"Original shape: {df.shape}"
        )

        # 1. Standardize column names
        df = self.clean_column_names(df)

        # 2. Remove completely empty columns
        df = self.remove_empty_columns(df)

        # 3. Remove duplicate rows
        df = self.remove_duplicates(df)

        # 4. Validate dataset
        df = self.validate_dataset(df)

        print(
            f"Cleaned shape: {df.shape}"
        )

        remaining_missing = int(
            df.isnull().sum().sum()
        )

        print(
            f"Missing values remaining: "
            f"{remaining_missing}"
        )

        print(
            "Missing values will be handled "
            "after train/test splitting."
        )

        print()

        return df