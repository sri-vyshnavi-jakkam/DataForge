import pandas as pd


class CorrelationAnalyzer:
    """
    Analyzes correlations between numerical features for DataForge.
    """

    def get_correlation_matrix(self, df):
        numerical_data = df.select_dtypes(
            include=["number"]
        )

        if numerical_data.shape[1] < 2:
            return pd.DataFrame()

        return numerical_data.corr()

    def get_high_correlations(self, df, threshold=0.7):
        correlation_matrix = self.get_correlation_matrix(df)

        if correlation_matrix.empty:
            return []

        high_correlations = []

        columns = correlation_matrix.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                correlation = correlation_matrix.iloc[i, j]

                if abs(correlation) >= threshold:
                    high_correlations.append({
                        "feature_1": columns[i],
                        "feature_2": columns[j],
                        "correlation": correlation
                    })

        return high_correlations

    def analyze(self, df, threshold=0.7):
        print("\nDataForge Correlation Analysis")

        correlation_matrix = self.get_correlation_matrix(df)

        high_correlations = self.get_high_correlations(
            df,
            threshold
        )

        if correlation_matrix.empty:
            print("Not enough numerical columns for correlation analysis.")
        else:
            print("Correlation analysis completed.")

        if high_correlations:
            print("\nHigh correlations detected:")

            for item in high_correlations:
                print(
                    f"{item['feature_1']} <-> "
                    f"{item['feature_2']}: "
                    f"{item['correlation']:.4f}"
                )
        else:
            print("No high correlations detected.")

        return {
            "correlation_matrix": correlation_matrix,
            "high_correlations": high_correlations
        }