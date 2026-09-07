from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.automl.automl_engine import AutoMLEngine


data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

engine = AutoMLEngine()

result = engine.run(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    problem_type="Classification"
)

print("\nEvaluation Results:")

for name, metrics in result["evaluation_results"].items():
    print(
        name,
        metrics
    )

print("\nAutoML test completed successfully.")