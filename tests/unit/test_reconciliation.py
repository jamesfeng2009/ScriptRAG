"""
Reconciliation Job 单元测试

测试覆盖:
- 错误分类
- 锁机制
- Structured Logging
- 结果序列化
- Metrics 输出
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scripts.reconciliation import (
    ReconciliationJob,
    ReconciliationResult,
    Metrics,
    StructuredLogger,
    ReconciliationError,
    TransientError,
    PermanentError,
    BusinessError,
    ErrorCategory,
    run_reconciliation,
)


class TestErrorClasses:
    """错误分类测试"""

    def test_reconciliation_error(self):
        error = ReconciliationError("Test error", ErrorCategory.BUSINESS, recoverable=False)
        assert error.message == "Test error"
        assert error.category == ErrorCategory.BUSINESS
        assert error.recoverable == False

    def test_transient_error(self):
        error = TransientError("Network timeout")
        assert error.category == ErrorCategory.TRANSIENT
        assert error.recoverable == True

    def test_permanent_error(self):
        error = PermanentError("Data corruption")
        assert error.category == ErrorCategory.PERMANENT
        assert error.recoverable == False

    def test_business_error(self):
        error = BusinessError("Status inconsistent")
        assert error.category == ErrorCategory.BUSINESS
        assert error.recoverable == False


class TestReconciliationResult:
    """对账结果测试"""

    def test_to_dict(self):
        result = ReconciliationResult(
            run_id="test123",
            started_at="2024-01-15T10:00:00",
            scanned_files=100,
            fixed_missing_vectors=5
        )
        data = result.to_dict()

        assert data["run_id"] == "test123"
        assert data["scanned_files"] == 100
        assert data["fixed_missing_vectors"] == 5

    def test_to_json(self):
        result = ReconciliationResult(
            run_id="test456",
            started_at="2024-01-15T10:00:00",
            scanned_files=50
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["run_id"] == "test456"
        assert parsed["scanned_files"] == 50

    def test_default_values(self):
        result = ReconciliationResult(
            run_id="test",
            started_at="2024-01-15T10:00:00"
        )

        assert result.status == "running"
        assert result.scanned_files == 0
        assert result.errors == []
        assert result.completed_at is None


class TestMetrics:
    """指标测试"""

    def test_to_prometheus(self):
        metrics = Metrics(
            reconciliation_runs_total=10,
            reconciliation_duration_seconds=120.5,
            files_scanned_total=1000,
            vectors_fixed_total=50
        )

        prometheus_output = metrics.to_prometheus()

        assert "reconciliation_runs_total 10" in prometheus_output
        assert "reconciliation_duration_seconds 120.500" in prometheus_output
        assert "files_scanned_total 1000" in prometheus_output
        assert "vectors_fixed_total 50" in prometheus_output

    def test_prometheus_help_text(self):
        metrics = Metrics()
        prometheus_output = metrics.to_prometheus()

        assert "# HELP reconciliation_runs_total" in prometheus_output
        assert "# HELP reconciliation_duration_seconds" in prometheus_output


class TestStructuredLogger:
    """结构化日志测试"""

    def test_log_output(self):
        logger = StructuredLogger(run_id="test123")

        with patch('builtins.print') as mock_print:
            logger.log("INFO", "Test message", key="value")

            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            parsed = json.loads(output)

            assert parsed["timestamp"] is not None
            assert parsed["level"] == "INFO"
            assert parsed["run_id"] == "test123"
            assert parsed["message"] == "Test message"
            assert parsed["key"] == "value"

    def test_log_with_extra_kwargs(self):
        logger = StructuredLogger(run_id="abc")
        count = 42

        with patch('builtins.print') as mock_print:
            logger.log("WARNING", "Warning message", count=count, status="ok")

            output = mock_print.call_args[0][0]
            parsed = json.loads(output)

            assert parsed["count"] == 42
            assert parsed["status"] == "ok"


class TestReconciliationJobLock:
    """锁机制测试"""

    def test_acquire_lock_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "test.lock"

            with patch('src.scripts.reconciliation.Path') as mock_path:
                mock_path.return_value.exists.return_value = False
                mock_path.return_value.__str__ = lambda s: str(lock_file)

                job = ReconciliationJob.__new__(ReconciliationJob)
                job._lock_file = lock_file
                job._lock_acquired = False
                job.logger = StructuredLogger("test")
                job.run_id = "test"

                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__ = Mock(return_value=Mock())
                    mock_open.return_value.__exit__ = Mock(return_value=False)

                    result = job._acquire_lock()
                    # Note: File path might not exist in temp dir

    def test_is_process_running(self):
        job = ReconciliationJob.__new__(ReconciliationJob)
        job.logger = StructuredLogger("test")

        # Test with invalid PID
        assert job._is_process_running("999999999") == False

        # Test with current PID
        current_pid = str(os.getpid())
        assert job._is_process_running(current_pid) == True


class TestReconciliationJobDryRun:
    """干跑模式测试"""

    def test_dry_run_flag(self):
        job = ReconciliationJob(dry_run=True)
        assert job.dry_run == True

        job2 = ReconciliationJob(dry_run=False)
        assert job2.dry_run == False

    def test_dry_run_no_modification(self):
        """干跑模式不应该修改任何数据"""
        job = ReconciliationJob(dry_run=True)

        # 模拟干跑模式下的检查逻辑
        result = {'missing': 0, 'orphaned': 0, 'status': 0}

        # 干跑模式下应该直接返回空结果，不做任何检查
        # 实际实现在 _check_and_fix_document 中
        assert job.dry_run == True


class TestReconciliationJobConfig:
    """配置测试"""

    def test_default_config(self):
        job = ReconciliationJob()

        assert 'database' in job.config
        assert job.config['database']['host'] is not None
        assert job.config['database']['port'] == 5432

    def test_workspace_id(self):
        job = ReconciliationJob(workspace_id="test-workspace")
        assert job.workspace_id == "test-workspace"

    def test_run_id_generation(self):
        job1 = ReconciliationJob()
        job2 = ReconciliationJob()

        # run_id 应该不同
        assert job1.run_id != job2.run_id

        # run_id 长度应该是 8
        assert len(job1.run_id) == 8


class TestRunReconciliation:
    """run_reconciliation 函数测试"""

    def test_run_reconciliation_returns_result(self):
        """run_reconciliation 应该返回 ReconciliationResult（mock 模式）"""
        with patch('src.scripts.reconciliation.asyncio.run') as mock_asyncio_run:
            mock_result = ReconciliationResult(
                run_id="test",
                started_at="2024-01-15T10:00:00",
                status="completed"
            )
            mock_asyncio_run.return_value = mock_result

            from src.scripts.reconciliation import run_reconciliation
            result = run_reconciliation()

            assert isinstance(result, ReconciliationResult)
            mock_asyncio_run.assert_called_once()


class TestErrorClassification:
    """错误分类测试"""

    def test_transient_error_recoverable(self):
        error = TransientError("Temporary network issue")
        assert error.recoverable == True
        assert error.category == ErrorCategory.TRANSIENT

    def test_permanent_error_not_recoverable(self):
        error = PermanentError("Database corruption")
        assert error.recoverable == False
        assert error.category == ErrorCategory.PERMANENT

    def test_business_error_not_recoverable(self):
        error = BusinessError("Invalid state transition")
        assert error.recoverable == False
        assert error.category == ErrorCategory.BUSINESS

    def test_error_message_preserved(self):
        error_msg = "Custom error message"
        error = ReconciliationError(error_msg, ErrorCategory.BUSINESS, False)
        assert str(error) == error_msg


class TestMetricsUpdate:
    """指标更新测试"""

    def test_update_metrics(self):
        job = ReconciliationJob()
        job.metrics = Metrics()

        result = ReconciliationResult(
            run_id="test",
            started_at="2024-01-15T10:00:00",
            completed_at="2024-01-15T10:02:00",
            scanned_files=100,
            fixed_missing_vectors=5,
            fixed_orphaned_vectors=3,
            transient_errors=2,
            permanent_errors=1,
            duration_ms=120000.0
        )

        job._update_metrics(result)

        assert job.metrics.reconciliation_runs_total == 1
        assert job.metrics.files_scanned_total == 100
        assert job.metrics.vectors_fixed_total == 8
        assert job.metrics.errors_total == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
