from src.automl.model_persistence import ModelPersistence
from src.prediction.predictor import Predictor


persistence = ModelPersistence()

model_package = persistence.load(
    "models/dataforge_model.pkl"
)

predictor = Predictor(
    model=model_package["model"],
    imputer=model_package["imputer"],
    encoder=model_package["encoder"],
    scaler=model_package["scaler"],
    feature_columns=model_package["feature_columns"]
)

new_data = {
    "age": 35,
    "salary": 55000,
    "city": "Hyderabad"
}

prediction = predictor.predict(
    new_data
)

print("\nDataForge prediction test completed.")
print(f"Input: {new_data}")
print(f"Prediction: {prediction[0]}")