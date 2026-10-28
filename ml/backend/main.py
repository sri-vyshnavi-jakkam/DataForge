
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.pipeline import DataForgePipeline

app = FastAPI(
    title="DataForge API",
    description="FastAPI backend for the DataForge AutoML platform",
    version="1.0.0",
)

latest_result = None
latest_pipeline = None


def make_json_safe(value: Any):
    """Convert pipeline outputs into JSON-compatible values."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item") and callable(value.item):
        try:
            return make_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


@app.get("/")
def home():
    return {"message": "DataForge API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    global latest_result, latest_pipeline

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    if not target_column.strip():
        raise HTTPException(
            status_code=400,
            detail="Please select a target column.",
        )

    temp_path = None

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty.",
            )

        with NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Call the existing DataForge ML pipeline.
        pipeline = DataForgePipeline()

        result = pipeline.run(
            str(temp_path),
            target_column=target_column,
        )

        # Keep the trained pipeline available for later prediction features.
        latest_pipeline = pipeline
        latest_result = make_json_safe(result)

        return {
            "status": "success",
            "message": "DataForge training completed successfully.",
            **latest_result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"DataForge training failed: {exc}",
        ) from exc

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/results")
def get_results():
    if latest_result is None:
        raise HTTPException(
            status_code=404,
            detail="No training results available. Train a model first.",
        )

    return latest_result
