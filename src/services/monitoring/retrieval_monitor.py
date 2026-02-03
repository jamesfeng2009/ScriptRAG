"""检索监控器 - 跟踪性能和质量问题指标

本模块为检索系统实现全面的监控：
1. 性能指标（延迟、吞吐量）
2. 质量指标（相似度、多样性）
3. 缓存指标（命中率、效率）
4. 错误跟踪和告警
"""

import logging
import time
from typing import List, Dict, Any, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
from pydantic import BaseModel
import numpy as np


logger = logging.getLogger(__name__)


class MonitoringConfig(BaseModel):
    """监控配置"""
    enabled: bool = True
    
    # 指标收集
    metrics_enabled: bool = True
    collection_interval: int = 60  # 秒
    retention_days: int = 30
    max_samples: int = 10000  # 内存中保留的最大样本数
    
    # 质量跟踪
    quality_tracking_enabled: bool = True
    min_samples_for_alert: int = 10
    quality_alert_threshold: float = 0.7
    
    # 告警
    alerts_enabled: bool = True
    quality_degradation_threshold: float = 0.15  # 15% 下降
    error_rate_threshold: float = 0.05  # 5% 错误
    latency_p95_threshold: float = 1000  # 1 秒


class MetricsCollector:
    """
    收集和聚合性能指标
    
    跟踪：
    - 查询延迟（p50, p95, p99）
    - 缓存命中率
    - 错误率
    - 吞吐量
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        初始化指标收集器
        
        Args:
            max_samples: 要保留的最大样本数
        """
        self.max_samples = max_samples
        
        # 时间序列数据（使用 deque 实现高效的 FIFO）
        self.query_latencies: deque = deque(maxlen=max_samples)
        self.query_timestamps: deque = deque(maxlen=max_samples)
        
        # 缓存指标
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        
        # 错误跟踪
        self.errors: deque = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        # LLM 调用跟踪
        self.llm_calls = 0
        self.llm_call_latencies: deque = deque(maxlen=max_samples)
    
    def record_query(self, latency: float) -> None:
        """
        记录查询执行
        
        Args:
            latency: 查询延迟（毫秒）
        """
        self.query_latencies.append(latency)
        self.query_timestamps.append(time.time())
    
    def record_cache_hit(self, cache_type: str) -> None:
        """
        记录缓存命中
        
        Args:
            cache_type: 缓存类型（query_expansion, embedding, result）
        """
        self.cache_hits[cache_type] += 1
    
    def record_cache_miss(self, cache_type: str) -> None:
        """
        记录缓存未命中
        
        Args:
            cache_type: 缓存类型
        """
        self.cache_misses[cache_type] += 1
    
    def record_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        记录错误发生
        
        Args:
            error_type: 错误类型
            details: 错误详情
        """
        self.errors.append({
            'type': error_type,
            'details': details,
            'timestamp': time.time()
        })
        self.error_counts[error_type] += 1
    
    def record_llm_call(self, latency: float) -> None:
        """
        记录 LLM API 调用
        
        Args:
            latency: 调用延迟（毫秒）
        """
        self.llm_calls += 1
        self.llm_call_latencies.append(latency)
    
    def get_latency_percentiles(self, time_window: Optional[int] = None) -> Dict[str, float]:
        """
        计算延迟百分位
        
        Args:
            time_window: 时间窗口（秒）（None 表示所有数据）
            
        Returns:
            包含 p50, p95, p99 延迟的字典
        """
        latencies = self._get_recent_latencies(time_window)
        
        if not latencies:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'avg': 0.0}
        
        return {
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'avg': float(np.mean(latencies))
        }
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取缓存统计
        
        Returns:
            按类型分类的缓存统计字典
        """
        stats = {}
        
        for cache_type in set(list(self.cache_hits.keys()) + list(self.cache_misses.keys())):
            hits = self.cache_hits[cache_type]
            misses = self.cache_misses[cache_type]
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0
            
            stats[cache_type] = {
                'hits': hits,
                'misses': misses,
                'total_requests': total,
                'hit_rate': hit_rate
            }
        
        return stats
    
    def get_error_stats(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        获取错误统计
        
        Args:
            time_window: 时间窗口（秒）
            
        Returns:
            包含错误统计的字典
        """
        recent_errors = self._get_recent_errors(time_window)
        
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error['type']] += 1
        
        total_queries = len(self._get_recent_latencies(time_window))
        error_rate = len(recent_errors) / total_queries if total_queries > 0 else 0.0
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': error_rate,
            'errors_by_type': dict(error_types)
        }
    
    def get_throughput(self, time_window: int = 60) -> float:
        """
        计算每秒查询数
        
        Args:
            time_window: 时间窗口（秒）
            
        Returns:
            每秒查询数
        """
        recent_queries = len(self._get_recent_latencies(time_window))
        return recent_queries / time_window if time_window > 0 else 0.0
    
    def _get_recent_latencies(self, time_window: Optional[int] = None) -> List[float]:
        """获取时间窗口内的延迟"""
        if time_window is None:
            return list(self.query_latencies)
        
        cutoff_time = time.time() - time_window
        recent = []
        
        for latency, timestamp in zip(self.query_latencies, self.query_timestamps):
            if timestamp >= cutoff_time:
                recent.append(latency)
        
        return recent
    
    def _get_recent_errors(self, time_window: Optional[int] = None) -> List[Dict]:
        """获取时间窗口内的错误"""
        if time_window is None:
            return list(self.errors)
        
        cutoff_time = time.time() - time_window
        return [e for e in self.errors if e['timestamp'] >= cutoff_time]


class QualityTracker:
    """
    跟踪检索质量指标随时间变化
    
    监控：
    - 平均相似度分数
    - 结果多样性
    - 标记分布（废弃、安全等）
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        初始化质量跟踪器
        
        Args:
            max_samples: 要保留的最大样本数
        """
        self.max_samples = max_samples
        
        # 质量指标
        self.similarity_scores: deque = deque(maxlen=max_samples)
        self.diversity_scores: deque = deque(maxlen=max_samples)
        self.result_counts: deque = deque(maxlen=max_samples)
        
        # 标记跟踪
        self.marker_counts = defaultdict(int)
        
        # 时间戳
        self.timestamps: deque = deque(maxlen=max_samples)
    
    def record_results(self, results: List[Any]) -> None:
        """
        记录检索结果用于质量跟踪
        
        Args:
            results: 检索结果列表
        """
        if not results:
            return
        
        # 计算平均相似度
        similarities = [r.similarity for r in results if hasattr(r, 'similarity')]
        if similarities:
            avg_similarity = np.mean(similarities)
            self.similarity_scores.append(avg_similarity)
        
        # 计算多样性（简化：唯一文件路径 / 总结果数）
        file_paths = [r.file_path for r in results if hasattr(r, 'file_path')]
        if file_paths:
            diversity = len(set(file_paths)) / len(file_paths)
            self.diversity_scores.append(diversity)
        
        # 跟踪结果数量
        self.result_counts.append(len(results))
        
        # 跟踪标记
        for result in results:
            if hasattr(result, 'has_deprecated') and result.has_deprecated:
                self.marker_counts['deprecated'] += 1
            if hasattr(result, 'has_security') and result.has_security:
                self.marker_counts['security'] += 1
            if hasattr(result, 'has_fixme') and result.has_fixme:
                self.marker_counts['fixme'] += 1
            if hasattr(result, 'has_todo') and result.has_todo:
                self.marker_counts['todo'] += 1
        
        self.timestamps.append(time.time())
    
    def get_quality_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        获取质量指标
        
        Args:
            time_window: 时间窗口（秒）
            
        Returns:
            包含质量指标的字典
        """
        recent_similarities = self._get_recent_values(self.similarity_scores, time_window)
        recent_diversities = self._get_recent_values(self.diversity_scores, time_window)
        recent_counts = self._get_recent_values(self.result_counts, time_window)
        
        return {
            'avg_similarity': float(np.mean(recent_similarities)) if recent_similarities else 0.0,
            'min_similarity': float(np.min(recent_similarities)) if recent_similarities else 0.0,
            'max_similarity': float(np.max(recent_similarities)) if recent_similarities else 0.0,
            'avg_diversity': float(np.mean(recent_diversities)) if recent_diversities else 0.0,
            'avg_result_count': float(np.mean(recent_counts)) if recent_counts else 0.0,
            'marker_distribution': dict(self.marker_counts),
            'sample_count': len(recent_similarities)
        }
    
    def detect_quality_degradation(
        self,
        baseline_window: int = 3600,  # 1 小时
        current_window: int = 300,     # 5 分钟
        threshold: float = 0.15        # 15% 下降
    ) -> Optional[Dict[str, Any]]:
        """
        检测质量下降
        
        Args:
            baseline_window: 基线时间窗口（秒）
            current_window: 当前时间窗口（秒）
            threshold: 下降阈值（0.15 = 15% 下降）
            
        Returns:
            如果检测到下降则返回告警字典，否则返回 None
        """
        baseline_metrics = self.get_quality_metrics(baseline_window)
        current_metrics = self.get_quality_metrics(current_window)
        
        if baseline_metrics['sample_count'] < 10 or current_metrics['sample_count'] < 5:
            return None  # 数据不足
        
        baseline_similarity = baseline_metrics['avg_similarity']
        current_similarity = current_metrics['avg_similarity']
        
        if baseline_similarity == 0:
            return None
        
        degradation = (baseline_similarity - current_similarity) / baseline_similarity
        
        if degradation > threshold:
            return {
                'type': 'quality_degradation',
                'severity': 'warning',
                'baseline_similarity': baseline_similarity,
                'current_similarity': current_similarity,
                'degradation_percent': degradation * 100,
                'threshold_percent': threshold * 100,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _get_recent_values(self, values: deque, time_window: Optional[int] = None) -> List[float]:
        """获取时间窗口内的值"""
        if time_window is None:
            return list(values)
        
        cutoff_time = time.time() - time_window
        recent = []
        
        for value, timestamp in zip(values, self.timestamps):
            if timestamp >= cutoff_time:
                recent.append(value)
        
        return recent


class RetrievalMonitor:
    """
    检索系统的主要监控接口
    
    整合指标收集、质量跟踪和告警
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        初始化检索监控器
        
        Args:
            config: 监控配置
        """
        self.config = config or MonitoringConfig()
        
        # 初始化组件
        self.metrics_collector = MetricsCollector(
            max_samples=self.config.max_samples
        ) if self.config.metrics_enabled else None
        
        self.quality_tracker = QualityTracker(
            max_samples=self.config.max_samples
        ) if self.config.quality_tracking_enabled else None
        
        # 告警历史
        self.alerts: deque = deque(maxlen=1000)
        
        logger.info("RetrievalMonitor initialized")
    
    def record_query(
        self,
        query: str,
        latency: float,
        results: List[Any],
        cache_hits: Dict[str, bool] = None
    ) -> None:
        """
        记录查询执行
        
        Args:
            query: 查询文本
            latency: 总延迟（毫秒）
            results: 检索结果
            cache_hits: 按类型分类的缓存命中/未命中字典
        """
        if not self.config.enabled:
            return
        
        # 记录指标
        if self.metrics_collector:
            self.metrics_collector.record_query(latency)
            
            # 记录缓存命中/未命中
            if cache_hits:
                for cache_type, hit in cache_hits.items():
                    if hit:
                        self.metrics_collector.record_cache_hit(cache_type)
                    else:
                        self.metrics_collector.record_cache_miss(cache_type)
        
        # 记录质量
        if self.quality_tracker:
            self.quality_tracker.record_results(results)
        
        # 检查告警
        if self.config.alerts_enabled:
            self._check_alerts()
    
    def record_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        记录错误发生
        
        Args:
            error_type: 错误类型
            details: 错误详情
        """
        if not self.config.enabled or not self.metrics_collector:
            return
        
        self.metrics_collector.record_error(error_type, details)
        logger.warning(f"Error recorded: {error_type}")
    
    def record_llm_call(self, latency: float) -> None:
        """
        记录 LLM API 调用
        
        Args:
            latency: 调用延迟（毫秒）
        """
        if not self.config.enabled or not self.metrics_collector:
            return
        
        self.metrics_collector.record_llm_call(latency)
    
    def get_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        获取综合指标
        
        Args:
            time_window: 时间窗口（秒）（None 表示所有数据）
            
        Returns:
            包含所有指标的字典
        """
        metrics = {
            'enabled': self.config.enabled,
            'timestamp': datetime.now().isoformat(),
            'time_window': time_window
        }
        
        if self.metrics_collector:
            metrics['performance'] = {
                'latency': self.metrics_collector.get_latency_percentiles(time_window),
                'throughput': self.metrics_collector.get_throughput(time_window or 60),
                'cache': self.metrics_collector.get_cache_stats(),
                'errors': self.metrics_collector.get_error_stats(time_window),
                'llm_calls': self.metrics_collector.llm_calls
            }
        
        if self.quality_tracker:
            metrics['quality'] = self.quality_tracker.get_quality_metrics(time_window)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取监控摘要
        
        Returns:
            摘要字典
        """
        return {
            'status': 'healthy',
            'metrics_1h': self.get_metrics(3600),
            'metrics_5m': self.get_metrics(300),
            'recent_alerts': list(self.alerts)[-10:] if self.alerts else []
        }
    
    def _check_alerts(self) -> None:
        """检查告警条件"""
        if not self.config.alerts_enabled:
            return
        
        # 检查质量下降
        if self.quality_tracker:
            alert = self.quality_tracker.detect_quality_degradation(
                threshold=self.config.quality_degradation_threshold
            )
            if alert:
                self._trigger_alert(alert)
        
        # 检查错误率
        if self.metrics_collector:
            error_stats = self.metrics_collector.get_error_stats(300)  # 最近 5 分钟
            if error_stats['error_rate'] > self.config.error_rate_threshold:
                self._trigger_alert({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'error_rate': error_stats['error_rate'],
                    'threshold': self.config.error_rate_threshold,
                    'timestamp': datetime.now().isoformat()
                })
        
        # 检查延迟
        if self.metrics_collector:
            latency = self.metrics_collector.get_latency_percentiles(300)  # 最近 5 分钟
            if latency['p95'] > self.config.latency_p95_threshold:
                self._trigger_alert({
                    'type': 'high_latency',
                    'severity': 'warning',
                    'p95_latency': latency['p95'],
                    'threshold': self.config.latency_p95_threshold,
                    'timestamp': datetime.now().isoformat()
                })
    
    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        """
        触发告警
        
        Args:
            alert: 告警字典
        """
        self.alerts.append(alert)
        logger.warning(f"Alert triggered: {alert['type']} - {alert}")
