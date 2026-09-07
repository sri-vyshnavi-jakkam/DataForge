from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class ClassificationModels:
    """
    Provides generic classification models for DataForge.
    """

    def get_models(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            "SVM": CalibratedClassifierCV(
                SVC(),
                ensemble=False
            ),
            "KNN": KNeighborsClassifier()
        }

        return models