from src.automl.model_persistence import ModelPersistence


persistence = ModelPersistence()

model_package = persistence.load(
    "models/dataforge_model.pkl"
)

print("\nModel package loaded successfully.")

print(
    "Model:",
    type(model_package["model"]).__name__
)

print(
    "Problem type:",
    model_package["problem_info"]["problem_type"]
)

print(
    "Target column:",
    model_package["problem_info"]["target_column"]
)

print(
    "Feature columns:",
    model_package["feature_columns"]
)