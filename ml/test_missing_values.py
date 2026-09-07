import pandas as pd

from src.pipeline import DataForgePipeline


data = pd.DataFrame({
    "age": [25, 30, None, 40, 45, 50, 28, 35, None, 60],
    "salary": [30000, 40000, 50000, None, 70000, 80000, 35000, None, 60000, 90000],
    "city": [
        "Hyderabad",
        "Delhi",
        "Chennai",
        "Mumbai",
        "Hyderabad",
        "Delhi",
        None,
        "Chennai",
        "Mumbai",
        "Hyderabad"
    ],
    "outcome": [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
})

dataset_path = "missing_values_test.csv"

data.to_csv(
    dataset_path,
    index=False
)

print("Test dataset created.")
print(data)

pipeline = DataForgePipeline()

result = pipeline.run(
    file_path=dataset_path,
    target_column="outcome"
)

print("\nMissing-value test completed successfully.")
print("Best model:", result["best_model_name"])
print("Best score:", result["best_score"])