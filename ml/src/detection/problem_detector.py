import pandas as pd


class ProblemDetector:
    """
    Detects the ML problem type for DataForge.
    """

    def __init__(self):
        self.target_column = None
        self.problem_type = None

    def suggest_target_columns(self, df):
        target_keywords = [
            "target",
            "label",
            "class",
            "outcome",
            "result",
            "response",
            "prediction"
        ]

        suggestions = []

        for column in df.columns:
            column_lower = column.lower()

            if column_lower in target_keywords:
                suggestions.append(column)
                continue

            words = column_lower.replace("-", "_").split("_")

            if any(keyword in words for keyword in target_keywords):
                suggestions.append(column)

        return suggestions

    def find_target_column(self, df, target_column):
        if target_column is None:
            return None

        if target_column in df.columns:
            return target_column

        target_column_lower = target_column.strip().lower()

        for column in df.columns:
            if column.lower() == target_column_lower:
                return column

        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    def detect_problem_type(self, df, target_column):
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset."
            )

        target = df[target_column].dropna()

        if target.empty:
            raise ValueError(
                f"Target column '{target_column}' contains no valid values."
            )

        unique_values = target.nunique()
        data_type = target.dtype

        if (
            pd.api.types.is_object_dtype(data_type)
            or pd.api.types.is_categorical_dtype(data_type)
            or pd.api.types.is_bool_dtype(data_type)
        ):
            problem_type = "Classification"

        elif pd.api.types.is_numeric_dtype(data_type):
            if unique_values == 2:
                problem_type = "Classification"

            elif (
                pd.api.types.is_integer_dtype(data_type)
                and unique_values <= 10
            ):
                problem_type = "Classification"

            else:
                problem_type = "Regression"

        else:
            raise ValueError(
                f"Unable to determine problem type for target column "
                f"'{target_column}'."
            )

        self.target_column = target_column
        self.problem_type = problem_type

        return problem_type

    def analyze(self, df, target_column=None):
        print("\nDataForge Problem Detection")
        print(f"Dataset shape: {df.shape}")

        suggestions = self.suggest_target_columns(df)

        print(f"Suggested target columns: {suggestions}")

        if target_column is None:
            if suggestions:
                target_column = suggestions[0]
                print(
                    f"Automatically selected target: {target_column}"
                )
            else:
                raise ValueError(
                    "Could not identify a possible target column. "
                    "Please specify the target column."
                )

        else:
            target_column = self.find_target_column(
                df,
                target_column
            )

            print(
                f"User-selected target: {target_column}"
            )

        problem_type = self.detect_problem_type(
            df,
            target_column
        )

        target = df[target_column]

        print(f"Target data type: {target.dtype}")
        print(f"Number of unique target values: {target.nunique()}")
        print(f"Detected problem type: {problem_type}")

        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "target_dtype": str(target.dtype),
            "unique_values": int(target.nunique()),
            "suggested_targets": suggestions
        }