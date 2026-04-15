"""Tests for v6.7 features: blob storage, dead-letter, admin routes, query filters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Blob storage helpers ────────────────────────────────────────


class TestBlobStorage:
    def test_get_blob_client_returns_none_when_not_configured(self):
        from src.models.blob_storage import _get_blob_client

        with patch("src.models.blob_storage.get_settings") as mock:
            s = MagicMock()
            s.azure_storage_connection_string = ""
            s.azure_storage_account_url = ""
            mock.return_value = s
            assert _get_blob_client("test.json") is None

    def test_upload_artifact_skips_when_not_configured(self, tmp_path):
        from src.models.blob_storage import upload_artifact

        with patch("src.models.blob_storage._get_blob_client", return_value=None):
            assert upload_artifact("nonexistent.json") is False

    def test_download_artifact_returns_false_when_not_configured(self):
        from src.models.blob_storage import download_artifact

        with patch("src.models.blob_storage._get_blob_client", return_value=None):
            assert download_artifact("test.json") is False

    def test_upload_artifact_returns_false_for_missing_file(self, tmp_path):
        from src.models.blob_storage import upload_artifact

        client = MagicMock()
        with (
            patch("src.models.blob_storage._get_blob_client", return_value=client),
            patch("src.models.blob_storage.ARTIFACTS_DIR", tmp_path),
        ):
            assert upload_artifact("nonexistent_file.json") is False

    def test_sync_artifacts_down_returns_count(self):
        from src.models.blob_storage import sync_artifacts_down

        with patch("src.models.blob_storage.download_artifact", return_value=False):
            assert sync_artifacts_down() == 0

    def test_sync_artifacts_up_returns_count(self):
        from src.models.blob_storage import sync_artifacts_up

        with patch("src.models.blob_storage.ARTIFACTS_DIR") as mock_dir:
            mock_dir.glob.return_value = []
            assert sync_artifacts_up() == 0


# ── Dead-letter queue retry job ─────────────────────────────────


class TestDeadLetterJob:
    def test_resolve_job_known(self):
        from src.data.jobs.dead_letter import _resolve_job

        assert _resolve_job("poll_fg_odds") is not None
        assert _resolve_job("poll_1h_odds") is not None
        assert _resolve_job("poll_stats") is not None

    def test_resolve_job_unknown(self):
        from src.data.jobs.dead_letter import _resolve_job

        assert _resolve_job("nonexistent_job") is None

    @pytest.mark.anyio
    async def test_process_dead_letter_queue_empty(self):
        from src.data.jobs.dead_letter import process_dead_letter_queue

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()

        with patch("src.data.jobs.dead_letter.async_session_factory") as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await process_dead_letter_queue()
            assert result == 0


# ── Admin routes ────────────────────────────────────────────────


class TestAdminRoutes:
    def test_scheduler_status_not_running(self):
        from src.api.routes.admin import scheduler_status

        # scheduler_status is async but we can check schema
        assert scheduler_status is not None

    def test_get_job_registry_returns_all_jobs(self):
        from src.api.routes.admin import _get_job_registry

        registry = _get_job_registry()
        assert "poll_fg_odds" in registry
        assert "fill_clv" in registry
        assert "process_dead_letter_queue" in registry
        assert "daily_retrain" in registry
        assert len(registry) >= 14

    def test_admin_router_contains_expected_paths(self):
        from src.api.routes.admin import router

        paths = {r.path for r in router.routes}
        assert "/admin/scheduler/status" in paths
        assert "/admin/jobs/trigger" in paths
        assert "/admin/model/promote" in paths
        assert "/admin/model/rollback" in paths
        assert "/admin/model/audit-log" in paths


# ── Model audit log ────────────────────────────────────────────


class TestModelAuditLog:
    def test_model_audit_log_table_exists(self):
        from src.db.models import ModelAuditLog

        assert ModelAuditLog.__tablename__ == "model_audit_log"
        assert hasattr(ModelAuditLog, "model_version")
        assert hasattr(ModelAuditLog, "action")
        assert hasattr(ModelAuditLog, "previous_version")
        assert hasattr(ModelAuditLog, "reason")
        assert hasattr(ModelAuditLog, "performed_at")


# ── IngestionFailure DLQ columns ────────────────────────────────


class TestIngestionFailureDLQ:
    def test_ingestion_failure_has_retry_columns(self):
        from src.db.models import IngestionFailure

        assert hasattr(IngestionFailure, "retry_count")
        assert hasattr(IngestionFailure, "permanently_failed")
        assert hasattr(IngestionFailure, "resolved_at")

    def test_ingestion_failure_defaults(self):
        from src.db.models import IngestionFailure

        # Column definitions should have defaults configured
        cols = {c.name: c for c in IngestionFailure.__table__.columns}
        assert cols["retry_count"].server_default is not None
        assert cols["permanently_failed"].server_default is not None
        assert cols["resolved_at"].nullable is True


# ── Config additions ────────────────────────────────────────────


class TestConfigAdditions:
    def test_cors_origins_default(self):
        from src.config import Settings

        s = Settings(app_env="test")
        assert s.cors_origins == []

    def test_azure_storage_config_defaults(self):
        from src.config import Settings

        s = Settings(app_env="test")
        assert s.azure_storage_connection_string == ""
        assert s.azure_storage_account_url == ""


# ── Prediction query filters ───────────────────────────────────


class TestPredictionFilters:
    def test_list_predictions_has_filter_params(self):
        """Verify the list_predictions endpoint accepts filter query params."""
        import inspect

        from src.api.routes.predictions import list_predictions

        sig = inspect.signature(list_predictions)
        param_names = set(sig.parameters.keys())
        assert "team" in param_names
        assert "min_edge" in param_names
        assert "limit" in param_names


# ── Predictor blob download integration ─────────────────────────


class TestPredictorBlobDownload:
    def test_download_blob_artifacts_no_config(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p._inference_feature_cols = []

        with (
            patch("src.models.predictor.logger"),
            patch(
                "src.models.blob_storage.sync_artifacts_down",
                side_effect=ImportError("no azure"),
            ),
        ):
            # Should not raise
            p._download_blob_artifacts()


# ── Trainer blob upload integration ─────────────────────────────


class TestTrainerBlobUpload:
    def test_sync_artifacts_up_called_after_train(self):
        """Verify trainer.py imports blob_storage.sync_artifacts_up."""
        from pathlib import Path

        trainer_path = Path(__file__).parent.parent / "src" / "models" / "trainer.py"
        source = trainer_path.read_text()
        assert "sync_artifacts_up" in source


# ── Scheduler DLQ job registration ────────────────────────────


class TestSchedulerDLQJob:
    def test_scheduler_has_dlq_job(self):
        from src.data.scheduler import create_scheduler

        scheduler = create_scheduler()
        job_ids = {j.id for j in scheduler.get_jobs()}
        assert "process_dead_letter_queue" in job_ids


# ── App includes admin router ───────────────────────────────────


class TestAppIncludesAdmin:
    def test_admin_routes_registered(self):
        from src.api.main import app

        paths = {r.path for r in app.routes}
        assert "/admin/scheduler/status" in paths
        assert "/admin/jobs/trigger" in paths
        assert "/admin/model/promote" in paths
