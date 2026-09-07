from sklearn.model_selection import train_test_split

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.imputer import DataImputer
from src.detection.problem_detector import ProblemDetector
from src.preprocessing.encoder import DataEncoder
from src.preprocessing.scaler import DataScaler
from src.imbalance.smote_handler import SMOTEHandler

from src.automl.automl_engine import AutoMLEngine
from src.automl.model_selector import ModelSelector
from src.automl.model_persistence import ModelPersistence

from src.prediction.predictor import Predictor

from src.insights.statistics import DataStatistics
from src.insights.correlations import CorrelationAnalyzer
from src.insights.feature_importance import FeatureImportanceAnalyzer


class DataForgePipeline:
    """
    Main machine learning pipeline for DataForge.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

        self.cleaner = DataCleaner()
        self.imputer = DataImputer()
        self.detector = ProblemDetector()
        self.encoder = DataEncoder()
        self.scaler = DataScaler()
        self.smote_handler = SMOTEHandler()

        self.automl_engine = AutoMLEngine()
        self.model_persistence = ModelPersistence()

        self.statistics = DataStatistics()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.feature_importance_analyzer = (
            FeatureImportanceAnalyzer()
        )

        self.selector = None
        self.predictor = None

        self.cleaned_data = None
        self.problem_info = None
        self.feature_columns = []

        self.numerical_feature_columns = []
        self.categorical_feature_columns = []

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.trained_models = {}
        self.evaluation_results = {}

        self.failed_training_models = {}
        self.failed_evaluation_models = {}

        self.best_model_name = None
        self.best_model = None
        self.best_model_path = None
        self.model_package_path = None
        self.best_score = None

        self.insights = {}

    def reset_state(self):
        """
        Reset results from a previous pipeline run.
        """

        self.automl_engine = AutoMLEngine()
        self.model_persistence = ModelPersistence()

        self.selector = None
        self.predictor = None

        self.cleaned_data = None
        self.problem_info = None
        self.feature_columns = []

        self.numerical_feature_columns = []
        self.categorical_feature_columns = []

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.trained_models = {}
        self.evaluation_results = {}

        self.failed_training_models = {}
        self.failed_evaluation_models = {}

        self.best_model_name = None
        self.best_model = None
        self.best_model_path = None
        self.model_package_path = None
        self.best_score = None

        self.insights = {}

        self.imputer = DataImputer()
        self.encoder = DataEncoder()
        self.scaler = DataScaler()
        self.smote_handler = SMOTEHandler()

    def run(self, file_path, target_column=None):
        """
        Run the complete DataForge machine learning pipeline.
        """

        self.reset_state()

        print("\nStarting DataForge pipeline...")

        # 1. Structural cleaning
        self.cleaned_data = self.cleaner.clean(
            file_path
        )

        # 2. Detect target and problem type
        self.problem_info = self.detector.analyze(
            self.cleaned_data,
            target_column
        )

        target_column = (
            self.problem_info["target_column"]
        )

        problem_type = (
            self.problem_info["problem_type"]
        )

        # 3. Remove rows with missing target values
        missing_target = int(
            self.cleaned_data[target_column].isnull().sum()
        )

        if missing_target > 0:
            print(
                f"\nRows with missing target values: "
                f"{missing_target}"
            )

            self.cleaned_data = (
                self.cleaned_data.dropna(
                    subset=[target_column]
                ).reset_index(drop=True)
            )

            print(
                "Rows with missing target values removed."
            )

        if self.cleaned_data.empty:
            raise ValueError(
                "No data remains after removing "
                "rows with missing target values."
            )

        # 4. Generate dataset insights
        print("\nGenerating dataset insights...")

        statistics_result = self.statistics.analyze(
            self.cleaned_data
        )

        correlation_result = (
            self.correlation_analyzer.analyze(
                self.cleaned_data
            )
        )

        self.insights["statistics"] = (
            statistics_result
        )

        self.insights["correlations"] = (
            correlation_result
        )

        # 5. Separate features and target
        X = self.cleaned_data.drop(
            columns=[target_column]
        )

        y = self.cleaned_data[target_column]

        self.feature_columns = X.columns.tolist()

        self.numerical_feature_columns = (
            X.select_dtypes(
                include=["number"]
            ).columns.tolist()
        )

        self.categorical_feature_columns = (
            X.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
        )

        print(
            f"\nFeatures shape: {X.shape}"
        )

        print(
            f"Target column: {target_column}"
        )

        print(
            f"Problem type: {problem_type}"
        )

        print(
            f"Original numerical features: "
            f"{self.numerical_feature_columns}"
        )

        print(
            f"Original categorical features: "
            f"{self.categorical_feature_columns}"
        )

        # 6. Train/test split
        print("\nSplitting data...")

        if problem_type == "Classification":
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            ) = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

        elif problem_type == "Regression":
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            ) = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state
            )

        else:
            raise ValueError(
                f"Unsupported problem type: "
                f"{problem_type}"
            )

        print(
            f"Training data shape: "
            f"{self.X_train.shape}"
        )

        print(
            f"Testing data shape: "
            f"{self.X_test.shape}"
        )

        # 7. Fit imputer only on training data
        print(
            "\nFitting missing value handler "
            "on training data..."
        )

        self.X_train = (
            self.imputer.fit_transform(
                self.X_train
            )
        )

        self.X_test = (
            self.imputer.transform(
                self.X_test
            )
        )

        # 8. Fit encoder only on training data
        print(
            "\nFitting encoder on training data..."
        )

        self.encoder.fit(
            self.X_train
        )

        self.X_train = (
            self.encoder.transform(
                self.X_train
            )
        )

        self.X_test = (
            self.encoder.transform(
                self.X_test
            )
        )

        # 9. Scale only the original numerical features
        print(
            "\nFitting scaler on original numerical features..."
        )

        encoded_numerical_columns = [
            column
            for column in self.X_train.columns
            if column in self.numerical_feature_columns
        ]

        self.scaler.fit(
            self.X_train,
            encoded_numerical_columns
        )

        self.X_train = (
            self.scaler.transform(
                self.X_train
            )
        )

        self.X_test = (
            self.scaler.transform(
                self.X_test
            )
        )

        # 10. Handle class imbalance
        if problem_type == "Classification":
            print(
                "\nChecking class imbalance..."
            )

            (
                self.X_train,
                self.y_train
            ) = self.smote_handler.apply_smote(
                self.X_train,
                self.y_train
            )

        # 11. Train and evaluate models using AutoML
        print("\nStarting AutoML...")

        automl_results = (
            self.automl_engine.run(
                X_train=self.X_train,
                X_test=self.X_test,
                y_train=self.y_train,
                y_test=self.y_test,
                problem_type=problem_type
            )
        )

        self.trained_models = (
            automl_results["trained_models"]
        )

        self.evaluation_results = (
            automl_results["evaluation_results"]
        )

        self.failed_training_models = (
            automl_results["failed_training_models"]
        )

        self.failed_evaluation_models = (
            automl_results[
                "failed_evaluation_models"
            ]
        )

        # 12. Select best model
        if problem_type == "Classification":
            self.selector = ModelSelector(
                metric="f1_score",
                higher_is_better=True
            )

        else:
            self.selector = ModelSelector(
                metric="r2_score",
                higher_is_better=True
            )

        (
            self.best_model_name,
            self.best_score
        ) = self.selector.select_best_model(
            self.evaluation_results,
            self.trained_models
        )

        self.best_model = (
            self.selector.best_model
        )

        # 13. Save best model
        self.best_model_path = (
            self.selector.save_best_model()
        )

        # 14. Feature importance
        feature_names = (
            self.X_train.columns.tolist()
        )

        feature_importance_result = (
            self.feature_importance_analyzer.analyze(
                self.best_model,
                feature_names
            )
        )

        self.insights["feature_importance"] = (
            feature_importance_result
        )

        # 15. Create predictor using original feature schema
        self.predictor = Predictor(
            model=self.best_model,
            imputer=self.imputer,
            encoder=self.encoder,
            scaler=self.scaler,
            feature_columns=self.feature_columns
        )

        # 16. Save complete model package
        print(
            "\nSaving complete DataForge model package..."
        )

        self.model_package_path = (
            self.model_persistence.save(
                model=self.best_model,
                imputer=self.imputer,
                encoder=self.encoder,
                scaler=self.scaler,
                problem_info=self.problem_info,
                feature_columns=self.feature_columns
            )
        )

        print("\nAutoML completed.")

        print(
            f"Best model: "
            f"{self.best_model_name}"
        )

        print(
            f"Best score: "
            f"{self.best_score:.4f}"
        )

        print(
            f"Best model saved: "
            f"{self.best_model_path}"
        )

        print(
            f"Complete model package saved: "
            f"{self.model_package_path}"
        )

        return {
            "problem_info": self.problem_info,
            "evaluation_results": self.evaluation_results,
            "failed_training_models": (
                self.failed_training_models
            ),
            "failed_evaluation_models": (
                self.failed_evaluation_models
            ),
            "best_model_name": self.best_model_name,
            "best_score": self.best_score,
            "best_model_path": self.best_model_path,
            "model_package_path": self.model_package_path,
            "insights": self.insights
        }