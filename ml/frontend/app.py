
import hashlib
import json

import os

import requests
import pandas as pd
import streamlit as st

API_URL = st.secrets.get(
    "DATAFORGE_API_URL",
    os.getenv("DATAFORGE_API_URL", "http://127.0.0.1:8000")
).rstrip("/")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataForge AutoML",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 DataForge AutoML")
st.write(
    "Upload a dataset, select the target column, "
    "and let DataForge train and evaluate machine-learning models."
)


# --------------------------------------------------
# BACKEND CONNECTION
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    health_response.raise_for_status()
    st.success("FastAPI backend connected successfully.")

except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. "
        "Make sure the backend is running on port 8000."
    )
    st.stop()


# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------

st.subheader("1. Upload your dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        # Read the uploaded dataset.
        dataframe = pd.read_csv(uploaded_file)

        if dataframe.empty:
            st.error("The uploaded CSV contains no data.")
            st.stop()

        if len(dataframe.columns) < 2:
            st.error(
                "The dataset must contain a target column "
                "and at least one feature column."
            )
            st.stop()

        if dataframe.columns.duplicated().any():
            st.error(
                "Your CSV contains duplicate column names. "
                "Please rename them before training."
            )
            st.stop()

        # --------------------------------------------------
        # DATASET PREVIEW
        # --------------------------------------------------

        st.subheader("2. Preview your dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", len(dataframe))
        col2.metric("Columns", len(dataframe.columns))
        col3.metric(
            "Missing values",
            int(dataframe.isna().sum().sum()),
        )

        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )

        # --------------------------------------------------
        # TARGET COLUMN
        # --------------------------------------------------

        st.subheader("3. Select the target column")

        target_column = st.selectbox(
            "What do you want the model to predict?",
            options=list(dataframe.columns),
        )

        # Identify the dataset and selected target.
        dataset_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        current_signature = (
            f"{dataset_hash}:{target_column}"
        )

        # Remove results belonging to a different dataset
        # or target column.
        previous_signature = st.session_state.get(
            "training_signature"
        )

        if (
            previous_signature is not None
            and previous_signature != current_signature
        ):
            st.session_state.pop(
                "training_result",
                None,
            )
            st.session_state.pop(
                "training_signature",
                None,
            )

        # --------------------------------------------------
        # COLUMN INFORMATION
        # --------------------------------------------------

        with st.expander("Column information"):
            column_info = pd.DataFrame({
                "Column": dataframe.columns,
                "Data type": [
                    str(dtype)
                    for dtype in dataframe.dtypes
                ],
                "Missing values": [
                    int(dataframe[column].isna().sum())
                    for column in dataframe.columns
                ],
                "Unique values": [
                    int(dataframe[column].nunique())
                    for column in dataframe.columns
                ],
            })

            st.dataframe(
                column_info,
                use_container_width=True,
                hide_index=True,
            )

        # --------------------------------------------------
        # TRAIN MODELS
        # --------------------------------------------------

        st.subheader("4. Train your models")

        if st.button(
            "🚀 Train Models",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Training models and evaluating performance..."
            ):
                try:
                    response = requests.post(
                        f"{API_URL}/train",
                        files={
                            "file": (
                                uploaded_file.name,
                                file_bytes,
                                "text/csv",
                            )
                        },
                        data={
                            "target_column": target_column,
                        },
                        timeout=600,
                    )

                    if not response.ok:
                        try:
                            error_data = response.json()
                            error_message = error_data.get(
                                "detail",
                                error_data,
                            )
                        except ValueError:
                            error_message = response.text

                        st.error(
                            f"Training failed "
                            f"(HTTP {response.status_code}): "
                            f"{error_message}"
                        )

                    else:
                        training_result = response.json()

                        st.session_state[
                            "training_result"
                        ] = training_result

                        st.session_state[
                            "training_signature"
                        ] = current_signature

                        st.success(
                            "Training completed successfully!"
                        )

                        st.rerun()

                except requests.Timeout:
                    st.error(
                        "Training timed out. Check Terminal 2 "
                        "for backend errors and whether training "
                        "is still running."
                    )

                except requests.RequestException as exc:
                    st.error(
                        f"Could not complete the training request: {exc}"
                    )

                except ValueError as exc:
                    st.error(
                        f"The backend returned invalid JSON: {exc}"
                    )

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV contains no readable data.")

    except UnicodeDecodeError:
        st.error(
            "The CSV encoding could not be read. "
            "Please save the file using UTF-8 encoding."
        )

    except pd.errors.ParserError as exc:
        st.error(f"Could not parse the CSV file: {exc}")

    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")


# --------------------------------------------------
# DISPLAY TRAINING RESULTS
# --------------------------------------------------

result = st.session_state.get("training_result")

if result:
    st.divider()
    st.header("📊 Training Results")

    problem_info = result.get("problem_info", {})
    evaluation_results = result.get(
        "evaluation_results", {}
    )
    best_model = result.get(
        "best_model_name", "Not available"
    )
    best_score = result.get("best_score")
    insights = result.get("insights", {})

    # --------------------------------------------------
    # PROBLEM INFORMATION
    # --------------------------------------------------

    st.subheader("Problem information")

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Target column",
        problem_info.get("target_column", "N/A"),
    )

    info2.metric(
        "Problem type",
        problem_info.get("problem_type", "N/A"),
    )

    info3.metric(
        "Target unique values",
        problem_info.get("unique_values", "N/A"),
    )

    # --------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------

    st.subheader("Best model")

    best1, best2 = st.columns(2)

    best1.metric(
        "Selected model",
        best_model,
    )

    best2.metric(
        "Best score",
        f"{best_score:.4f}"
        if isinstance(best_score, (int, float))
        else "N/A",
    )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    st.subheader("Model comparison")

    if evaluation_results:
        metrics_df = pd.DataFrame.from_dict(
            evaluation_results,
            orient="index",
        )

        metrics_df.index.name = "Model"
        metrics_df = metrics_df.reset_index()

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        # Use an appropriate metric for the problem type.
        problem_type = problem_info.get(
            "problem_type", ""
        ).lower()

        if problem_type == "classification":
            score_column = (
                "f1_score"
                if "f1_score" in metrics_df.columns
                else "accuracy"
                if "accuracy" in metrics_df.columns
                else None
            )

        else:
            score_column = (
                "r2"
                if "r2" in metrics_df.columns
                else None
            )

        if score_column:
            chart_data = metrics_df[
                ["Model", score_column]
            ].set_index("Model")

            st.write(
                f"**Model comparison by {score_column}**"
            )

            st.bar_chart(chart_data)

        else:
            st.info(
                "No suitable comparison metric was returned "
                "by the backend."
            )

    else:
        st.info("No model evaluation results are available.")

    # --------------------------------------------------
    # MODEL ERRORS
    # --------------------------------------------------

    failed_models = result.get(
        "failed_training_models", {}
    )

    failed_evaluations = result.get(
        "failed_evaluation_models", {}
    )

    if failed_models or failed_evaluations:
        with st.expander("Models with errors"):
            st.json({
                "training_errors": failed_models,
                "evaluation_errors": failed_evaluations,
            })

    # --------------------------------------------------
    # DATASET INSIGHTS
    # --------------------------------------------------

    st.subheader("Dataset insights")

    statistics = insights.get("statistics", {})
    dataset_info = statistics.get(
        "dataset_info", {}
    )

    if dataset_info:
        a, b, c, d = st.columns(4)

        a.metric(
            "Dataset rows",
            dataset_info.get("rows", "N/A"),
        )

        b.metric(
            "Dataset columns",
            dataset_info.get("columns", "N/A"),
        )

        c.metric(
            "Duplicate rows",
            dataset_info.get("duplicate_rows", "N/A"),
        )

        d.metric(
            "Missing values",
            dataset_info.get("missing_values", "N/A"),
        )

        st.write("**Numerical columns**")

        st.write(
            ", ".join(
                dataset_info.get(
                    "numerical_columns", []
                )
            ) or "None"
        )

        st.write("**Categorical columns**")

        st.write(
            ", ".join(
                dataset_info.get(
                    "categorical_columns", []
                )
            ) or "None"
        )

    # Descriptive statistics
    descriptive_stats = statistics.get("statistics")

    if descriptive_stats:
        with st.expander("Descriptive statistics"):
            st.code(str(descriptive_stats))

    # Missing values
    missing_info = statistics.get("missing_values")

    if missing_info:
        with st.expander("Missing-value details"):
            st.code(str(missing_info))

    # Correlation analysis
    correlations = insights.get("correlations", {})

    correlation_matrix = correlations.get(
        "correlation_matrix"
    )

    if correlation_matrix:
        with st.expander("Correlation analysis"):
            st.code(str(correlation_matrix))

            high_correlations = correlations.get(
                "high_correlations", []
            )

            if high_correlations:
                st.write("**Highly correlated columns**")

                st.dataframe(
                    pd.DataFrame(high_correlations),
                    use_container_width=True,
                    hide_index=True,
                )

    # Feature importance
    feature_importance = insights.get(
        "feature_importance"
    )

    if feature_importance:
        with st.expander("Feature importance"):
            st.code(str(feature_importance))

    # --------------------------------------------------
    # SAVED MODEL INFORMATION
    # --------------------------------------------------

    st.subheader("Saved model")

    st.write(
        "Best model path:",
        result.get(
            "best_model_path", "Not available"
        ),
    )

    st.write(
        "Model package path:",
        result.get(
            "model_package_path", "Not available"
        ),
    )

    st.caption(
        "These paths identify files on the backend machine. "
        "They are not direct download links."
    )

    # --------------------------------------------------
    # DOWNLOAD TRAINING RESULTS
    # --------------------------------------------------

    st.download_button(
        "📥 Download training results as JSON",
        data=json.dumps(
            result,
            indent=2,
            allow_nan=False,
        ),
        file_name="dataforge_results.json",
        mime="application/json",
        use_container_width=True,
    )

    # --------------------------------------------------
    # FETCH LATEST BACKEND RESULTS
    # --------------------------------------------------

    if st.button("🔄 Fetch latest backend results"):
        try:
            latest_response = requests.get(
                f"{API_URL}/results",
                timeout=10,
            )

            latest_response.raise_for_status()

            latest_result = latest_response.json()

            st.session_state[
                "training_result"
            ] = latest_result

            # Associate the fetched result with the current
            # selection only when its target matches.
            latest_problem = latest_result.get(
                "problem_info", {}
            )

            if (
                latest_problem.get("target_column")
                == target_column
            ):
                st.session_state[
                    "training_signature"
                ] = current_signature
            else:
                st.session_state.pop(
                    "training_signature", None
                )

            st.rerun()

        except requests.RequestException as exc:
            st.error(
                f"Could not fetch results: {exc}"
            )

        except ValueError as exc:
            st.error(
                f"The backend returned invalid JSON: {exc}"
            )

else:
    st.info(
        "Upload a CSV, select a target column, and click "
        "'Train Models' to see your results here."
    )
