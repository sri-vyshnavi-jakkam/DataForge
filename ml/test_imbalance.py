import pandas as pd

from src.pipeline import DataForgePipeline


data = {
    "age": [
        22, 25, 28, 30, 32,
        35, 38, 40, 42, 45,
        48, 50, 52, 55, 58,
        60, 62, 65, 68, 70,
        72, 74, 76, 78, 80,
        82, 84, 86, 88, 90
    ],
    "salary": [
        28000, 30000, 32000, 34000, 36000,
        38000, 40000, 42000, 44000, 46000,
        48000, 50000, 52000, 54000, 56000,
        58000, 60000, 62000, 64000, 66000,
        68000, 70000, 72000, 74000, 76000,
        78000, 80000, 82000, 84000, 86000
    ],
    "city": [
        "Hyderabad", "Delhi", "Mumbai", "Chennai", "Hyderabad",
        "Delhi", "Mumbai", "Chennai", "Hyderabad", "Delhi",
        "Mumbai", "Chennai", "Hyderabad", "Delhi", "Mumbai",
        "Chennai", "Hyderabad", "Delhi", "Mumbai", "Chennai",
        "Hyderabad", "Delhi", "Mumbai", "Chennai", "Hyderabad",
        "Delhi", "Mumbai", "Chennai", "Hyderabad", "Delhi"
    ],
    "outcome": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 1
    ]
}


df = pd.DataFrame(data)

file_path = "imbalance_test_data.csv"

df.to_csv(
    file_path,
    index=False
)

print("Imbalanced test dataset created.")
print(df["outcome"].value_counts())


pipeline = DataForgePipeline()

result = pipeline.run(
    file_path=file_path,
    target_column="outcome"
)

print("\nImbalance test completed successfully.")
print(
    f"Best model: "
    f"{result['best_model_name']}"
)
print(
    f"Best score: "
    f"{result['best_score']}"
)