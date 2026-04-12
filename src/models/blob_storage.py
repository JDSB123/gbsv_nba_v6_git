"""Azure Blob Storage helper for model artifact upload/download.

When ``AZURE_STORAGE_CONNECTION_STRING`` or ``AZURE_STORAGE_ACCOUNT_URL`` is
set, artifacts are stored/loaded from the ``models`` container in Azure Blob
Storage.  Otherwise, falls back to the local ``artifacts/`` directory.
"""

import logging
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
_CONTAINER_NAME = "models"


def _get_blob_client(blob_name: str):
    """Return a BlobClient for the given blob name, or *None* if not configured."""
    settings = get_settings()
    conn_str = settings.azure_storage_connection_string
    account_url = settings.azure_storage_account_url

    if not conn_str and not account_url:
        return None

    from azure.storage.blob import BlobServiceClient

    if conn_str:
        service = BlobServiceClient.from_connection_string(conn_str)
    else:
        from azure.identity import DefaultAzureCredential

        service = BlobServiceClient(account_url, credential=DefaultAzureCredential())

    return service.get_blob_client(_CONTAINER_NAME, blob_name)


def upload_artifact(filename: str) -> bool:
    """Upload a local artifact file to blob storage.  Returns True on success."""
    local_path = ARTIFACTS_DIR / filename
    if not local_path.exists():
        logger.warning("Artifact not found for upload: %s", local_path)
        return False

    client = _get_blob_client(filename)
    if client is None:
        logger.debug("Blob storage not configured — skipping upload of %s", filename)
        return False

    try:
        with open(local_path, "rb") as f:
            client.upload_blob(f, overwrite=True)
        logger.info("Uploaded artifact to blob: %s", filename)
        return True
    except Exception:
        logger.warning("Failed to upload artifact %s to blob", filename, exc_info=True)
        return False


def download_artifact(filename: str) -> bool:
    """Download a blob artifact to the local artifacts directory.  Returns True on success."""
    client = _get_blob_client(filename)
    if client is None:
        return False

    local_path = ARTIFACTS_DIR / filename
    try:
        data = client.download_blob().readall()
        local_path.write_bytes(data)
        logger.info("Downloaded artifact from blob: %s", filename)
        return True
    except Exception:
        logger.debug("Blob artifact %s not available", filename, exc_info=True)
        return False


def sync_artifacts_down() -> int:
    """Download all known artifact files from blob storage.  Returns count downloaded."""
    artifact_names = [
        "model_home_fg.json",
        "model_away_fg.json",
        "model_home_1h.json",
        "model_away_1h.json",
        "metrics.json",
        "imputation.json",
        "best_params.json",
        "feature_importance.json",
        "calibration.json",
        "trained_feature_cols.json",
        "ensemble_meta.json",
        "model_home_fg_lgb.txt",
        "model_away_fg_lgb.txt",
        "model_home_1h_lgb.txt",
        "model_away_1h_lgb.txt",
        "ood_detector.json",
        "shap_importance.json",
        "model_home_fg_q10.json",
        "model_home_fg_q90.json",
        "model_away_fg_q10.json",
        "model_away_fg_q90.json",
        "model_home_1h_q10.json",
        "model_home_1h_q90.json",
        "model_away_1h_q10.json",
        "model_away_1h_q90.json",
    ]
    count = 0
    for name in artifact_names:
        if download_artifact(name):
            count += 1
    return count


def sync_artifacts_up() -> int:
    """Upload all artifact files in the artifacts directory to blob.  Returns count uploaded."""
    count = 0
    for path in ARTIFACTS_DIR.glob("*"):
        if path.is_file() and path.suffix in (".json", ".txt") and upload_artifact(path.name):
            count += 1
    return count
