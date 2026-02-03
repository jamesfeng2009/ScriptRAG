"""
Reconciliation Job - 数据对账脚本

Source of Truth 架构：
- SQL 数据库是账本：所有文件状态以 SQL 表为准
- 向量数据库是缓存：只是加速检索的索引
- 对账脚本：定时修复 SQL 和向量库之间的不一致

严谨性保证（非分布式版）：
1. 单进程锁：防止同一进程重复执行
2. 幂等性：多次执行结果一致
3. Structured Logging：JSON 格式日志
4. 指标埋点：Prometheus 兼容
5. 错误分类：区分可恢复/不可恢复错误

运行方式：
1. 手动触发: python -m src.scripts.reconciliation --manual
2. 定时任务: 0 2 * * * python /path/to/reconciliation.py --cron
"""

import asyncio
import logging
import json
import argparse
import sys
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
from functools import wraps

import yaml


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    TRANSIENT = "transient"      # 临时错误（网络抖动）
    PERMANENT = "permanent"       # 永久错误（数据损坏）
    BUSINESS = "business"         # 业务错误（状态不一致）


class ReconciliationError(Exception):
    """对账基础异常"""

    def __init__(self, message: str, category: ErrorCategory, recoverable: bool):
        super().__init__(message)
        self.message = message
        self.category = category
        self.recoverable = recoverable


class TransientError(ReconciliationError):
    """临时错误，可重试"""

    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.TRANSIENT, recoverable=True)


class PermanentError(ReconciliationError):
    """永久错误，不可重试"""

    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.PERMANENT, recoverable=False)


class BusinessError(ReconciliationError):
    """业务错误，需要人工介入"""

    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.BUSINESS, recoverable=False)


@dataclass
class ReconciliationResult:
    """对账结果"""
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    scanned_files: int = 0
    fixed_missing_vectors: int = 0
    fixed_orphaned_vectors: int = 0
    fixed_failed_status: int = 0
    transient_errors: int = 0
    permanent_errors: int = 0
    duration_ms: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class Metrics:
    """Prometheus 兼容指标"""
    reconciliation_runs_total: int = 0
    reconciliation_duration_seconds: float = 0.0
    files_scanned_total: int = 0
    vectors_fixed_total: int = 0
    errors_total: int = 0
    last_run_timestamp: float = 0.0

    def to_prometheus(self) -> str:
        lines = [
            "# HELP reconciliation_runs_total Total number of reconciliation runs",
            f"reconciliation_runs_total {self.reconciliation_runs_total}",
            "# HELP reconciliation_duration_seconds Duration of reconciliation in seconds",
            f"reconciliation_duration_seconds {self.reconciliation_duration_seconds:.3f}",
            "# HELP files_scanned_total Total number of files scanned",
            f"files_scanned_total {self.files_scanned_total}",
            "# HELP vectors_fixed_total Total number of vectors fixed",
            f"vectors_fixed_total {self.vectors_fixed_total}",
            "# HELP errors_total Total number of errors",
            f"errors_total {self.errors_total}",
            "# HELP last_run_timestamp Timestamp of last run",
            f"last_run_timestamp {self.last_run_timestamp}",
        ]
        return "\n".join(lines)


class StructuredLogger:
    """结构化日志"""

    def __init__(self, run_id: str):
        self.run_id = run_id

    def log(self, level: str, message: str, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "run_id": self.run_id,
            "message": message,
            **kwargs
        }
        print(json.dumps(log_entry, ensure_ascii=False))


class ReconciliationJob:
    """
    对账作业

    设计原则（非分布式）：
    1. 单进程锁：防止重复执行
    2. 幂等性：多次执行结果一致
    3. 可观测性：Structured Logging + Metrics
    4. 错误隔离：区分可恢复/不可恢复错误
    """

    def __init__(
        self,
        config_path: str = None,
        workspace_id: str = "default",
        dry_run: bool = False
    ):
        self.run_id = str(uuid.uuid4())[:8]
        self.workspace_id = workspace_id
        self.dry_run = dry_run
        self.config = self._load_config(config_path)
        self.logger = StructuredLogger(self.run_id)
        self.metrics = Metrics()
        self._lock_file = Path("/tmp/reconciliation.lock")
        self._lock_acquired = False

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置"""
        if config_path is None:
            config_path = os.environ.get(
                'RECONCILIATION_CONFIG',
                str(Path(__file__).parent / 'config.yaml')
            )

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

        return {
            'database': {
                'host': os.environ.get('POSTGRES_HOST', 'localhost'),
                'port': int(os.environ.get('POSTGRES_PORT', 5432)),
                'database': os.environ.get('POSTGRES_DB', 'Screenplay'),
                'user': os.environ.get('POSTGRES_USER', 'postgres'),
                'password': os.environ.get('POSTGRES_PASSWORD', '123456'),
            },
            'batch_size': 100,
        }

    def _acquire_lock(self) -> bool:
        """单进程文件锁"""
        try:
            if self._lock_file.exists():
                with open(self._lock_file, 'r') as f:
                    pid = f.read().strip()
                    if pid and self._is_process_running(pid):
                        self.logger.log("WARNING", "Another instance is running", pid=pid)
                        return False
                    else:
                        self.logger.log("INFO", "Stale lock file found, removing")
                        self._lock_file.unlink()
        except Exception:
            pass

        try:
            with open(self._lock_file, 'w') as f:
                f.write(str(os.getpid()))
            self._lock_acquired = True
            self.logger.log("INFO", "Lock acquired", pid=os.getpid())
            return True
        except Exception as e:
            self.logger.log("ERROR", "Failed to acquire lock", error=str(e))
            return False

    def _release_lock(self):
        """释放文件锁"""
        if self._lock_acquired:
            try:
                if self._lock_file.exists():
                    self._lock_file.unlink()
                self.logger.log("INFO", "Lock released")
            except Exception as e:
                self.logger.log("ERROR", "Failed to release lock", error=str(e))
            self._lock_acquired = False

    def _is_process_running(self, pid: str) -> bool:
        """检查进程是否在运行"""
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @asynccontextmanager
    async def distributed_lock(self):
        """单进程锁（替代分布式锁）"""
        if self._acquire_lock():
            try:
                yield
            finally:
                self._release_lock()
        else:
            raise TransientError("Another instance is running")

    async def run(self) -> ReconciliationResult:
        """执行对账"""
        start_time = time.time()
        result = ReconciliationResult(
            run_id=self.run_id,
            started_at=datetime.now().isoformat()
        )

        try:
            self.logger.log("INFO", "Reconciliation started")

            async with self.distributed_lock():
                await self._execute_reconciliation(result)

        except ReconciliationError as e:
            result.status = "failed"
            result.errors.append({
                "type": e.category.value,
                "message": e.message,
                "recoverable": e.recoverable
            })
            self.logger.log("ERROR", f"Reconciliation failed: {e.message}")

        except Exception as e:
            result.status = "failed"
            result.errors.append({
                "type": "unknown",
                "message": str(e),
                "recoverable": False
            })
            self.logger.log("CRITICAL", f"Unexpected error: {e}")

        finally:
            result.completed_at = datetime.now().isoformat()
            result.duration_ms = (time.time() - start_time) * 1000
            result.status = "completed" if result.status == "running" else result.status

            self._update_metrics(result)
            self.logger.log("INFO", "Reconciliation finished", **asdict(result))

        return result

    async def _execute_reconciliation(self, result: ReconciliationResult):
        """执行对账逻辑"""
        from src.services.database.postgres import PostgresService
        from src.services.rag.document_repository import DocumentRepository, FileStatus

        db_config = self.config['database']
        pg = PostgresService(db_config)
        repo = DocumentRepository(pg)

        try:
            await pg.connect()
            await repo.initialize()

            indexed_files = repo.get_all_indexed()
            total = len(indexed_files)

            self.logger.log("INFO", "Scanning files", total=total)

            for i, doc_file in enumerate(indexed_files):
                result.scanned_files += 1

                if i % 100 == 0:
                    self.logger.log(
                        "INFO",
                        "Progress",
                        progress=f"{i}/{total}",
                        percentage=f"{i/total*100:.1f}%"
                    )

                try:
                    fix_result = await self._check_and_fix_document(
                        pg, repo, doc_file
                    )

                    result.fixed_missing_vectors += fix_result['missing']
                    result.fixed_orphaned_vectors += fix_result['orphaned']
                    result.fixed_failed_status += fix_result['status']

                except PermanentError as e:
                    result.permanent_errors += 1
                    result.errors.append({
                        "file_id": doc_file.id,
                        "type": "permanent",
                        "message": e.message
                    })

                except TransientError as e:
                    result.transient_errors += 1
                    if len(result.errors) < 10:
                        result.errors.append({
                            "file_id": doc_file.id,
                            "type": "transient",
                            "message": e.message
                        })

        finally:
            await repo.close()
            await pg.disconnect()

    async def _check_and_fix_document(
        self,
        pg,
        repo,
        doc_file
    ) -> Dict[str, int]:
        """检查并修复单个文档"""
        result = {'missing': 0, 'orphaned': 0, 'status': 0}

        if self.dry_run:
            return result

        vector_count = await self._count_vectors(pg, doc_file.id)

        if vector_count == 0:
            self.logger.log(
                "WARNING",
                "Missing vectors detected",
                file_id=doc_file.id,
                expected_chunks=doc_file.chunk_count,
                actual_vectors=0
            )

            repo.update_status(
                doc_file.id,
                FileStatus.FAILED,
                error_msg="Reconciliation: 向量缺失"
            )
            result['missing'] = 1
            result['status'] = 1

        elif vector_count != doc_file.chunk_count:
            self.logger.log(
                "WARNING",
                "Vector count mismatch",
                file_id=doc_file.id,
                expected=doc_file.chunk_count,
                actual=vector_count
            )

            await self._cleanup_vectors(pg, doc_file.id)

            repo.update_status(
                doc_file.id,
                FileStatus.FAILED,
                error_msg=f"Reconciliation: 向量数量不匹配 ({vector_count} vs {doc_file.chunk_count})"
            )
            result['orphaned'] = 1
            result['status'] = 1

        return result

    async def _count_vectors(self, pg, source_id: str) -> int:
        """统计向量数量"""
        result = await pg.fetch(
            "SELECT COUNT(*) as cnt FROM code_documents "
            "WHERE workspace_id = $1 AND file_path LIKE $2",
            self.workspace_id,
            f"{source_id}_%"
        )
        return result[0]['cnt'] if result else 0

    async def _cleanup_vectors(self, pg, source_id: str):
        """清理向量"""
        await pg.execute(
            "DELETE FROM code_documents "
            "WHERE workspace_id = $1 AND file_path LIKE $2",
            self.workspace_id,
            f"{source_id}_%"
        )

    def _update_metrics(self, result: ReconciliationResult):
        """更新指标"""
        self.metrics.reconciliation_runs_total += 1
        self.metrics.reconciliation_duration_seconds = result.duration_ms / 1000
        self.metrics.files_scanned_total += result.scanned_files
        self.metrics.vectors_fixed_total += (
            result.fixed_missing_vectors + result.fixed_orphaned_vectors
        )
        self.metrics.errors_total += (
            result.transient_errors + result.permanent_errors
        )
        self.metrics.last_run_timestamp = time.time()

    def get_metrics(self) -> str:
        """获取 Prometheus 格式指标"""
        return self.metrics.to_prometheus()


def run_reconciliation(config_path: str = None, dry_run: bool = False) -> ReconciliationResult:
    """运行对账"""
    job = ReconciliationJob(config_path, dry_run=dry_run)
    return asyncio.run(job.run())


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='数据对账脚本（严谨版）')
    parser.add_argument('--manual', action='store_true', help='手动触发')
    parser.add_argument('--cron', action='store_true', help='定时任务模式')
    parser.add_argument('--dry-run', action='store_true', help='演练模式（不修改数据）')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--metrics', action='store_true', help='输出 Prometheus 指标')
    parser.add_argument('--lock-timeout', type=int, help='锁超时时间（秒）')

    args = parser.parse_args()

    if args.metrics:
        job = ReconciliationJob(args.config)
        print(job.get_metrics())
        return 0

    try:
        result = run_reconciliation(args.config, dry_run=args.dry_run)
        print(result.to_json())

        if result.status == "failed" and result.permanent_errors > 0:
            return 2
        return 0 if result.status == "completed" else 1

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    sys.exit(main())
