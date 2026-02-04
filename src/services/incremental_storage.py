"""增量存储优化模块

功能：
1. 差量存储（Delta Storage）：只存储状态变化的部分
2. 状态压缩（State Compression）：压缩历史数据
3. 智能合并（Smart Merge）：合并相邻的状态快照
4. 清理策略（Cleanup Policy）：自动清理过期数据

设计原则：
- 减少内存占用
- 加速状态恢复
- 保持数据完整性
"""

import json
import zlib
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    ZLIB = "zlib"
    JSON = "json"


@dataclass
class DeltaState:
    """差量状态

    Attributes:
        changes: 变化字段字典
        timestamp: 时间戳
        compression: 压缩类型
    """
    changes: Dict[str, Any]
    timestamp: str
    compression: str = CompressionType.NONE.value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeltaState':
        """从字典创建"""
        return cls(**data)


@dataclass
class CompressedSnapshot:
    """压缩快照

    Attributes:
        state_hash: 状态哈希
        compressed_data: 压缩后的数据
        compression_type: 压缩类型
        original_size: 原始大小
        compressed_size: 压缩后大小
        timestamp: 创建时间
    """
    state_hash: str
    compressed_data: str
    compression_type: str
    original_size: int
    compressed_size: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressedSnapshot':
        """从字典创建"""
        return cls(**data)

    def get_decompression_ratio(self) -> float:
        """获取压缩比"""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100


class DeltaStorage:
    """差量存储管理器

    管理状态变更的差量记录，支持：
    1. 记录状态变更
    2. 计算状态差异
    3. 重建历史状态
    4. 压缩历史数据
    """

    def __init__(
        self,
        max_delta_chain: int = 100,
        compression_threshold: int = 50,
        enable_compression: bool = True
    ):
        """
        Args:
            max_delta_chain: 最大差量链长度（超过此值创建快照）
            compression_threshold: 压缩阈值（差量数量）
            enable_compression: 是否启用压缩
        """
        self.max_delta_chain = max_delta_chain
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression

        self._delta_chain: List[DeltaState] = []
        self._snapshots: List[CompressedSnapshot] = []
        self._last_snapshot_index: int = -1
        self._current_state: Dict[str, Any] = {}

    def record_change(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        changed_fields: Optional[List[str]] = None
    ) -> DeltaState:
        """
        记录状态变更

        Args:
            old_state: 旧状态
            new_state: 新状态
            changed_fields: 指定变化的字段（可选，自动计算）

        Returns:
            DeltaState 差量状态
        """
        if changed_fields is None:
            changed_fields = self._compute_changed_fields(old_state, new_state)

        changes = {}
        for field_name in changed_fields:
            old_value = old_state.get(field_name)
            new_value = new_state.get(field_name)
            changes[field_name] = {
                "old": old_value,
                "new": new_value
            }

        delta = DeltaState(
            changes=changes,
            timestamp=datetime.now().isoformat(),
            compression=CompressionType.ZLIB.value if self.enable_compression else CompressionType.NONE.value
        )

        self._delta_chain.append(delta)
        self._current_state = new_state.copy()

        self._check_compression_needed()

        return delta

    def _compute_changed_fields(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any]
    ) -> List[str]:
        """计算状态变化的字段"""
        changed = []
        all_keys = set(old_state.keys()) | set(new_state.keys())

        for key in all_keys:
            old_val = old_state.get(key)
            new_val = new_state.get(key)

            if old_val != new_val:
                changed.append(key)

        return changed

    def _check_compression_needed(self):
        """检查是否需要压缩"""
        if len(self._delta_chain) >= self.max_delta_chain:
            self.create_snapshot()

    def create_snapshot(self) -> Optional[CompressedSnapshot]:
        """创建压缩快照"""
        if not self._current_state:
            return None

        state_json = json.dumps(self._current_state, default=str, ensure_ascii=False)
        original_size = len(state_json)

        if self.enable_compression:
            compressed_data = zlib.compress(state_json.encode('utf-8'))
            compressed_data = base64.b64encode(compressed_data).decode('ascii')
            compression_type = CompressionType.ZLIB.value
        else:
            compressed_data = state_json
            compression_type = CompressionType.NONE.value

        state_hash = hashlib.sha256(state_json.encode('utf-8')).hexdigest()[:16]

        snapshot = CompressedSnapshot(
            state_hash=state_hash,
            compressed_data=compressed_data,
            compression_type=compression_type,
            original_size=original_size,
            compressed_size=len(compressed_data),
            timestamp=datetime.now().isoformat()
        )

        self._snapshots.append(snapshot)
        self._last_snapshot_index = len(self._snapshots) - 1
        self._delta_chain = []

        logger.info(
            f"Created snapshot with compression ratio: {snapshot.get_decompression_ratio():.2f}%"
        )

        return snapshot

    def get_state_at(self, index: int) -> Optional[Dict[str, Any]]:
        """
        获取指定索引的状态

        Args:
            index: 状态索引

        Returns:
            状态字典，如果不存在返回 None
        """
        if index < 0:
            return None

        if index > self._last_snapshot_index:
            start_from = self._last_snapshot_index + 1
            deltas_needed = index - self._last_snapshot_index
        else:
            start_from = 0
            deltas_needed = index - self._last_snapshot_index

        state = self._get_snapshot_state(self._last_snapshot_index) if self._last_snapshot_index >= 0 else {}

        for i, delta in enumerate(self._delta_chain):
            if start_from + i < index:
                state = self._apply_delta(state, delta)

        return state

    def _get_snapshot_state(self, snapshot_index: int) -> Dict[str, Any]:
        """从快照获取状态"""
        if snapshot_index < 0 or snapshot_index >= len(self._snapshots):
            return {}

        snapshot = self._snapshots[snapshot_index]

        if snapshot.compression_type == CompressionType.ZLIB.value:
            compressed_data = base64.b64decode(snapshot.compressed_data.encode('ascii'))
            decompressed = zlib.decompress(compressed_data)
            return json.loads(decompressed.decode('utf-8'))
        else:
            return json.loads(snapshot.compressed_data)

    def _apply_delta(
        self,
        state: Dict[str, Any],
        delta: DeltaState
    ) -> Dict[str, Any]:
        """应用差量到状态"""
        new_state = state.copy()

        for field_name, change in delta.changes.items():
            new_state[field_name] = change["new"]

        return new_state

    def get_delta_chain_length(self) -> int:
        """获取差量链长度"""
        return len(self._delta_chain)

    def get_snapshots_count(self) -> int:
        """获取快照数量"""
        return len(self._snapshots)

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        total_original = sum(s.original_size for s in self._snapshots)
        total_compressed = sum(s.compressed_size for s in self._snapshots)

        return {
            "snapshots_count": len(self._snapshots),
            "delta_chain_length": len(self._delta_chain),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "compression_ratio": (
                (1 - total_compressed / total_original) * 100
                if total_original > 0 else 0
            ),
            "delta_chain_saved": len(self._delta_chain) * 100
        }


class IncrementalCleanupPolicy:
    """增量清理策略

    管理历史数据的自动清理：
    1. 基于时间的清理
    2. 基于数量的清理
    3. 基于访问频率的清理
    """

    def __init__(
        self,
        max_history_count: int = 1000,
        max_history_days: int = 30,
        min_access_count: int = 3,
        cleanup_interval_hours: int = 24
    ):
        """
        Args:
            max_history_count: 最大历史记录数
            max_history_days: 最大保留天数
            min_access_count: 最小访问次数（低于此值标记为低频）
            cleanup_interval_hours: 清理间隔（小时）
        """
        self.max_history_count = max_history_count
        self.max_history_days = max_history_days
        self.min_access_count = min_access_count
        self.cleanup_interval_hours = cleanup_interval_hours

        self._last_cleanup: Optional[datetime] = None
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._sessions_metadata: Dict[str, Dict[str, Any]] = {}

    def record_access(self, session_id: str):
        """记录会话访问"""
        self._access_counts[session_id] += 1

    def register_session(
        self,
        session_id: str,
        created_at: datetime,
        last_accessed: Optional[datetime] = None
    ):
        """注册会话"""
        self._sessions_metadata[session_id] = {
            "created_at": created_at.isoformat(),
            "last_accessed": (last_accessed or created_at).isoformat(),
            "access_count": 0
        }

    def update_access(self, session_id: str, accessed_at: datetime):
        """更新访问信息"""
        if session_id in self._sessions_metadata:
            self._sessions_metadata[session_id]["last_accessed"] = accessed_at.isoformat()
            self._access_counts[session_id] += 1
            self._sessions_metadata[session_id]["access_count"] = self._access_counts[session_id]

    def get_sessions_to_cleanup(self) -> Tuple[List[str], str]:
        """
        获取需要清理的会话列表

        Returns:
            (sessions_to_remove, reason)
        """
        to_remove = []
        reason = ""

        now = datetime.now()
        cutoff_time = now - timedelta(days=self.max_history_days)

        for session_id, metadata in self._sessions_metadata.items():
            created_at = datetime.fromisoformat(metadata["created_at"])
            last_accessed = datetime.fromisoformat(metadata["last_accessed"])
            access_count = metadata["access_count"]

            reasons = []

            if created_at < cutoff_time:
                reasons.append(f"older_than_{self.max_history_days}_days")

            if access_count < self.min_access_count and (now - last_accessed) > timedelta(days=7):
                reasons.append("low_access_frequency")

            if reasons:
                to_remove.append(session_id)
                reason = "; ".join(reasons)

        if len(self._sessions_metadata) > self.max_history_count:
            excess = len(self._sessions_metadata) - self.max_history_count
            sorted_sessions = sorted(
                self._sessions_metadata.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            for session_id, _ in sorted_sessions[:excess]:
                if session_id not in to_remove:
                    to_remove.append(session_id)
            reason = f"exceed_max_count_{self.max_history_count}"

        return to_remove, reason

    def should_cleanup(self) -> bool:
        """判断是否应该执行清理"""
        if self._last_cleanup is None:
            return True

        next_cleanup = self._last_cleanup + timedelta(hours=self.cleanup_interval_hours)
        return datetime.now() >= next_cleanup

    def mark_cleanup_done(self):
        """标记清理完成"""
        self._last_cleanup = datetime.now()


class StateDiffCalculator:
    """状态差异计算器

    计算两个状态之间的差异，支持：
    1. 完整差异
    2. 字段级别差异
    3. 嵌套对象差异
    """

    @staticmethod
    def compute_full_diff(
        old_state: Dict[str, Any],
        new_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算完整差异"""
        return {
            "added": StateDiffCalculator._find_added(old_state, new_state),
            "removed": StateDiffCalculator._find_removed(old_state, new_state),
            "changed": StateDiffCalculator._find_changed(old_state, new_state),
            "unchanged": StateDiffCalculator._find_unchanged(old_state, new_state)
        }

    @staticmethod
    def _find_added(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        """查找新增的字段"""
        added = {}
        for key, value in new_state.items():
            if key not in old_state:
                added[key] = value
        return added

    @staticmethod
    def _find_removed(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        """查找移除的字段"""
        removed = {}
        for key, value in old_state.items():
            if key not in new_state:
                removed[key] = value
        return removed

    @staticmethod
    def _find_changed(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        """查找变更的字段"""
        changed = {}
        for key, new_val in new_state.items():
            if key in old_state:
                old_val = old_state[key]
                if old_val != new_val:
                    changed[key] = {
                        "from": old_val,
                        "to": new_val
                    }
        return changed

    @staticmethod
    def _find_unchanged(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> List[str]:
        """查找未变更的字段"""
        unchanged = []
        for key, new_val in new_state.items():
            if key in old_state:
                old_val = old_state[key]
                if old_val == new_val:
                    unchanged.append(key)
        return unchanged

    @staticmethod
    def compute_delta(
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        priority_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """计算差量（优先字段优先）"""
        changed = StateDiffCalculator._find_changed(old_state, new_state)
        added = StateDiffCalculator._find_added(old_state, new_state)
        removed = StateDiffCalculator._find_removed(old_state, new_state)

        delta = {}

        if priority_fields:
            for field in priority_fields:
                if field in changed:
                    delta[field] = changed[field]
                elif field in added:
                    delta[field] = {"to": added[field]}
                elif field in removed:
                    delta[field] = {"from": removed[field], "to": None}

        delta.update(changed)
        delta.update(added)
        delta.update(removed)

        return delta


class IncrementalStorageOptimizer:
    """增量存储优化器

    整合差量存储、压缩和清理策略，提供统一的优化接口。

    Usage:
        optimizer = IncrementalStorageOptimizer()
        optimizer.record_step(old_state, new_state)
        stats = optimizer.get_optimization_stats()
    """

    def __init__(
        self,
        enable_compression: bool = True,
        max_delta_chain: int = 100,
        compression_threshold: int = 50,
        max_history_count: int = 1000,
        max_history_days: int = 30
    ):
        """
        Args:
            enable_compression: 是否启用压缩
            max_delta_chain: 最大差量链长度
            compression_threshold: 压缩阈值
            max_history_count: 最大历史记录数
            max_history_days: 最大保留天数
        """
        self.delta_storage = DeltaStorage(
            max_delta_chain=max_delta_chain,
            compression_threshold=compression_threshold,
            enable_compression=enable_compression
        )

        self.cleanup_policy = IncrementalCleanupPolicy(
            max_history_count=max_history_count,
            max_history_days=max_history_days
        )

        self._step_count = 0

    def record_step(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        changed_fields: Optional[List[str]] = None
    ) -> DeltaState:
        """记录一步状态变更"""
        self._step_count += 1
        return self.delta_storage.record_change(old_state, new_state, changed_fields)

    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.delta_storage._current_state.copy()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        delta_stats = self.delta_storage.get_storage_stats()

        return {
            "total_steps": self._step_count,
            "snapshots_count": delta_stats["snapshots_count"],
            "delta_chain_length": delta_stats["delta_chain_length"],
            "compression_ratio": delta_stats["compression_ratio"],
            "estimated_memory_saved": delta_stats["total_original_size"] - delta_stats["total_compressed_size"],
            "cleanup_policy": {
                "max_history_count": self.cleanup_policy.max_history_count,
                "max_history_days": self.cleanup_policy.max_history_days
            }
        }

    def estimate_storage_size(
        self,
        state: Dict[str, Any],
        steps: int
    ) -> Dict[str, Any]:
        """估算存储大小

        Args:
            state: 状态示例
            steps: 步数

        Returns:
            存储大小估算
        """
        state_json = json.dumps(state, default=str, ensure_ascii=False)
        original_size = len(state_json)

        if self.delta_storage.enable_compression:
            compressed_size = len(zlib.compress(state_json.encode('utf-8')))
        else:
            compressed_size = original_size

        return {
            "single_step_original": original_size,
            "single_step_compressed": compressed_size,
            "full_storage_original": original_size * steps,
            "full_storage_compressed": compressed_size * steps,
            "estimated_savings_percent": (1 - compressed_size / original_size) * 100
        }
