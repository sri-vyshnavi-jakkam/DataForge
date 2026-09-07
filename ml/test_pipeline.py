from src.pipeline import DataForgePipeline


pipeline = DataForgePipeline()

result = pipeline.run(
    "pipeline_test_data.csv"
)

print("\nPipeline test completed successfully.")

print(
    "Best model:",
    result["best_model_name"]
)

print(
    "Evaluation results:"
)

print(
    result["evaluation_results"]
)