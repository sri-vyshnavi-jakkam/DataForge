import pandas as pd

from src.pipeline import DataForgePipeline


data = pd.DataFrame({
    "age": [22, 25, 28, 30, 35, 40, 45, 50, 55, 60,
            24, 32, 38, 42, 48, 52, 58, 27, 33, 47],

    "salary": [25000, 30000, 35000, 40000, 45000, 50000, 55000,
               60000, 65000, 70000, 28000, 42000, 48000, 52000,
               58000, 62000, 68000, 33000, 43000, 57000],

    "city": [
        "Hyderabad", "Delhi", "Mumbai", "Chennai", "Hyderabad",
        "Delhi", "Mumbai", "Chennai", "Hyderabad", "Delhi",
        "Mumbai", "Chennai", "Hyderabad", "Delhi", "Mumbai",
        "Chennai", "Hyderabad", "Delhi", "Mumbai", "Chennai"
    ],

    "education": [
        "BTech", "BSc", "MTech", "BTech", "MBA",
        "BSc", "MTech", "MBA", "BTech", "BSc",
        "MTech", "MBA", "BTech", "BSc", "MTech",
        "MBA", "BTech", "BSc", "MTech", "MBA"
    ],

    "outcome": [
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 1, 1, 0, 1,
        1, 1, 0, 1, 1
    ]
})


dataset_path = "categorical_test.csv"

data.to_csv(
    dataset_path,
    index=False
)

print("Categorical test dataset created.")
print(data)


pipeline = DataForgePipeline()

result = pipeline.run(
    file_path=dataset_path,
    target_column="outcome"
)


print("\nCategorical-data test completed successfully.")
print("Best model:", result["best_model_name"])
print("Best score:", result["best_score"])