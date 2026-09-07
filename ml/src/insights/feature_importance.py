import pandas as pd


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for DataForge models.
    """

    def get_feature_importance(self, model, feature_names):
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        elif hasattr(model, "coef_"):
            coefficients = model.coef_

            if coefficients.ndim == 1:
                importance = abs(coefficients)

            elif coefficients.ndim == 2:
                importance = abs(coefficients).mean(axis=0)

        if importance is None:
            return pd.DataFrame()

        if len(importance) != len(feature_names):
            raise ValueError(
                "Number of feature importances does not match "
                "number of feature names."
            )

        result = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })

        result = result.sort_values(
            by="importance",
            ascending=False
        ).reset_index(drop=True)

        return result

    def analyze(self, model, feature_names):
        print("\nDataForge Feature Importance Analysis")

        result = self.get_feature_importance(
            model,
            feature_names
        )

        if result.empty:
            print(
                "Feature importance is not available "
                "for this model."
            )

            return result

        print("Feature importance calculated successfully.")

        print("\nFeature Importance:")
        print(result)

        return result