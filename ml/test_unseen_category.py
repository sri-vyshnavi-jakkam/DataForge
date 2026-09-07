from src.prediction.model_service import ModelService


model_service = ModelService(
    model_path="models/dataforge_model.pkl"
)

new_data = {
    "age": 35,
    "salary": 55000,
    "city": "Bangalore"
}

prediction = model_service.predict(
    new_data
)

print("\nUnseen-category prediction test completed.")
print("Input:", new_data)
print("Prediction:", prediction[0])