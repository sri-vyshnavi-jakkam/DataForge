from collections import Counter

from imblearn.over_sampling import SMOTE


class SMOTEHandler:
    """
    Handles class imbalance in DataForge.
    """

    def __init__(
        self,
        imbalance_threshold=0.5,
        random_state=42
    ):
        self.imbalance_threshold = imbalance_threshold
        self.random_state = random_state
        self.smote_applied = False

    def check_imbalance(self, y):
        """
        Check whether the target classes are imbalanced.
        """

        class_counts = Counter(y)

        if len(class_counts) < 2:
            return False

        majority_count = max(
            class_counts.values()
        )

        minority_count = min(
            class_counts.values()
        )

        imbalance_ratio = (
            minority_count / majority_count
        )

        return (
            imbalance_ratio
            < self.imbalance_threshold
        )

    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to the training data when necessary.
        """

        self.smote_applied = False

        class_counts = Counter(y_train)

        print("\nDataForge Imbalance Handling")

        print(
            f"Original class distribution: "
            f"{class_counts}"
        )

        if len(class_counts) < 2:
            print(
                "Only one class is present. "
                "SMOTE is not applicable."
            )

            return X_train, y_train

        is_imbalanced = self.check_imbalance(
            y_train
        )

        if not is_imbalanced:
            print(
                "Dataset is reasonably balanced."
            )

            print(
                "SMOTE not required."
            )

            return X_train, y_train

        minority_count = min(
            class_counts.values()
        )

        if minority_count < 2:
            print(
                "Too few minority samples for SMOTE."
            )

            print(
                "Returning the original training data."
            )

            return X_train, y_train

        # SMOTE requires k_neighbors to be
        # smaller than the minority class size.
        k_neighbors = min(
            5,
            minority_count - 1
        )

        print(
            "Imbalance detected."
        )

        print(
            f"Minority class samples: "
            f"{minority_count}"
        )

        print(
            f"SMOTE k_neighbors: "
            f"{k_neighbors}"
        )

        smote = SMOTE(
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )

        try:
            X_resampled, y_resampled = (
                smote.fit_resample(
                    X_train,
                    y_train
                )
            )

        except ValueError as error:
            print(
                f"SMOTE could not be applied: "
                f"{error}"
            )

            print(
                "Returning the original "
                "training data."
            )

            return X_train, y_train

        self.smote_applied = True

        print(
            f"New class distribution: "
            f"{Counter(y_resampled)}"
        )

        print(
            "SMOTE applied successfully."
        )

        return X_resampled, y_resampled