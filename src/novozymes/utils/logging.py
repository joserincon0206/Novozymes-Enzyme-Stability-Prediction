"""MLflow logger utility."""

from lightning.pytorch.loggers import MLFlowLogger


def get_mlflow_logger(
    experiment_name: str,
    run_name: str,
    tracking_uri: str = "http://localhost:5000",
    tags: dict[str, str] | None = None,
) -> MLFlowLogger:
    """Create an MLFlowLogger instance."""
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags or {},
    )
