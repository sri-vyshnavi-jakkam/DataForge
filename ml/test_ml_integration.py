from src.pipeline import DataForgePipeline
from src.prediction.model_service import ModelService


dataset_path = "pipeline_test_data.csv"

pipeline = DataForgePipeline()

result = pipeline.run(
    file_path=dataset_path
)

print("\nDataForge training completed.")
print(
    "Best model:",
    result["best_model_name"]
)
print(
    "Best score:",
    result["best_score"]
)
print(
    "Model package:",
    result["model_package_path"]
)


model_service = ModelService(
    model_path=result["model_package_path"]
)

new_data = {
    "age": 35,
    "salary": 55000,
    "city": "Hyderabad"
}

prediction = model_service.predict(
    new_data
)

print("\nDataForge prediction completed.")
print("Input:", new_data)
print("Prediction:", prediction[0])