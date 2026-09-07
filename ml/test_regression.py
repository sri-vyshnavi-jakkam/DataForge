import pandas as pd

from src.pipeline import DataForgePipeline


data = pd.DataFrame({
    "age": [
        22, 25, 28, 30, 32,
        35, 38, 40, 42, 45,
        48, 50, 52, 55, 58,
        60, 62, 65, 68, 70
    ],

    "experience": [
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 20
    ],

    "city": [
        "Hyderabad", "Delhi", "Mumbai", "Chennai", "Hyderabad",
        "Delhi", "Mumbai", "Chennai", "Hyderabad", "Delhi",
        "Mumbai", "Chennai", "Hyderabad", "Delhi", "Mumbai",
        "Chennai", "Hyderabad", "Delhi", "Mumbai", "Chennai"
    ],

    "salary": [
        28000, 32000, 36000, 40000, 44000,
        48000, 52000, 56000, 60000, 64000,
        68000, 72000, 76000, 80000, 84000,
        88000, 92000, 96000, 100000, 110000
    ]
})


dataset_path = "regression_test.csv"

data.to_csv(
    dataset_path,
    index=False
)

print("Regression test dataset created.")
print(data)


pipeline = DataForgePipeline()

result = pipeline.run(
    file_path=dataset_path,
    target_column="salary"
)


print("\nRegression test completed successfully.")
print("Best model:", result["best_model_name"])
print("Best R2 score:", result["best_score"])