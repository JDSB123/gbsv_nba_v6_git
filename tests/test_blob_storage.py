"""Tests for src.models.blob_storage — Azure Blob upload/download/sync."""

from unittest.mock import MagicMock, patch

from src.models.blob_storage import (
    ARTIFACTS_DIR,
    _get_blob_client,
    download_artifact,
    sync_artifacts_down,
    sync_artifacts_up,
    upload_artifact,
)


class TestGetBlobClient:
    def test_returns_none_when_not_configured(self):
        with patch("src.models.blob_storage.get_settings") as mock:
            mock.return_value.azure_storage_connection_string = ""
            mock.return_value.azure_storage_account_url = ""
            assert _get_blob_client("model.json") is None

    def test_uses_connection_string(self):
        with patch("src.models.blob_storage.get_settings") as mock_settings:
            mock_settings.return_value.azure_storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net"
            mock_settings.return_value.azure_storage_account_url = ""
            with patch("azure.storage.blob.BlobServiceClient.from_connection_string") as mock_bsc:
                mock_svc = MagicMock()
                mock_bsc.return_value = mock_svc
                _get_blob_client("test.json")
                mock_bsc.assert_called_once()
                mock_svc.get_blob_client.assert_called_once_with("models", "test.json")


class TestUploadArtifact:
    def test_missing_file_returns_false(self):
        assert upload_artifact("nonexistent_file_xyz.json") is False

    def test_no_blob_config_returns_false(self, tmp_path):
        test_file = ARTIFACTS_DIR / "_test_upload_probe.json"
        test_file.write_text("{}")
        try:
            with patch("src.models.blob_storage.get_settings") as mock:
                mock.return_value.azure_storage_connection_string = ""
                mock.return_value.azure_storage_account_url = ""
                assert upload_artifact("_test_upload_probe.json") is False
        finally:
            test_file.unlink(missing_ok=True)

    def test_successful_upload(self):
        test_file = ARTIFACTS_DIR / "_test_upload_ok.json"
        test_file.write_text("{}")
        try:
            mock_client = MagicMock()
            with patch("src.models.blob_storage._get_blob_client", return_value=mock_client):
                result = upload_artifact("_test_upload_ok.json")
                assert result is True
                mock_client.upload_blob.assert_called_once()
        finally:
            test_file.unlink(missing_ok=True)

    def test_upload_exception_returns_false(self):
        test_file = ARTIFACTS_DIR / "_test_upload_err.json"
        test_file.write_text("{}")
        try:
            mock_client = MagicMock()
            mock_client.upload_blob.side_effect = Exception("Blob error")
            with patch("src.models.blob_storage._get_blob_client", return_value=mock_client):
                assert upload_artifact("_test_upload_err.json") is False
        finally:
            test_file.unlink(missing_ok=True)


class TestDownloadArtifact:
    def test_no_blob_config_returns_false(self):
        with patch("src.models.blob_storage._get_blob_client", return_value=None):
            assert download_artifact("model.json") is False

    def test_successful_download(self):
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = b'{"test": true}'
        target = ARTIFACTS_DIR / "_test_download_ok.json"
        try:
            with patch("src.models.blob_storage._get_blob_client", return_value=mock_client):
                result = download_artifact("_test_download_ok.json")
                assert result is True
                assert target.exists()
                assert target.read_text() == '{"test": true}'
        finally:
            target.unlink(missing_ok=True)

    def test_download_exception_returns_false(self):
        mock_client = MagicMock()
        mock_client.download_blob.side_effect = Exception("Not found")
        with patch("src.models.blob_storage._get_blob_client", return_value=mock_client):
            assert download_artifact("nonexistent.json") is False


class TestSyncArtifactsDown:
    def test_counts_successful_downloads(self):
        def fake_download(name):
            return name.endswith("fg.json")

        with patch("src.models.blob_storage.download_artifact", side_effect=fake_download):
            count = sync_artifacts_down()
            assert count > 0

    def test_all_fail_returns_zero(self):
        with patch("src.models.blob_storage.download_artifact", return_value=False):
            assert sync_artifacts_down() == 0


class TestSyncArtifactsUp:
    def test_uploads_json_and_txt_files(self):
        (ARTIFACTS_DIR / "_test_sync_up.json").write_text("{}")
        try:
            with patch("src.models.blob_storage.upload_artifact", return_value=True):
                count = sync_artifacts_up()
                assert count >= 1
        finally:
            (ARTIFACTS_DIR / "_test_sync_up.json").unlink(missing_ok=True)
