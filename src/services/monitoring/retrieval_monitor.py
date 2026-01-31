"""Retrieval Monitor - Track performance and quality metrics

This module implements comprehensive monitoring for the retrieval system:
1. Performance metrics (latency, throughput)
2. Quality metrics (similarity, diversity)
3. Cache metrics (hit rates, efficiency)
4. Error tracking and alerting
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
    """Monitoring configuration"""
    enabled: bool = True
    
    # Metrics collection
    metrics_enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_days: int = 30
    max_samples: int = 10000  # Maximum samples to keep in memory
    
    # Quality tracking
    quality_tracking_enabled: bool = True
    min_samples_for_alert: int = 10
    quality_alert_threshold: float = 0.7
    
    # Alerts
    alerts_enabled: bool = True
    quality_degradation_threshold: float = 0.15  # 15% drop
    error_rate_threshold: float = 0.05  # 5% errors
    latency_p95_threshold: float = 1000  # 1 second


class MetricsCollector:
    """
    Collects and aggregates performance metrics
    
    Tracks:
    - Query latency (p50, p95, p99)
    - Cache hit rates
    - Error rates
    - Throughput
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics collector
        
        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        
        # Time-series data (using deque for efficient FIFO)
        self.query_latencies: deque = deque(maxlen=max_samples)
        self.query_timestamps: deque = deque(maxlen=max_samples)
        
        # Cache metrics
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        
        # Error tracking
        self.errors: deque = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        # LLM call tracking
        self.llm_calls = 0
        self.llm_call_latencies: deque = deque(maxlen=max_samples)
    
    def record_query(self, latency: float) -> None:
        """
        Record query execution
        
        Args:
            latency: Query latency in milliseconds
        """
        self.query_latencies.append(latency)
        self.query_timestamps.append(time.time())
    
    def record_cache_hit(self, cache_type: str) -> None:
        """
        Record cache hit
        
        Args:
            cache_type: Type of cache (query_expansion, embedding, result)
        """
        self.cache_hits[cache_type] += 1
    
    def record_cache_miss(self, cache_type: str) -> None:
        """
        Record cache miss
        
        Args:
            cache_type: Type of cache
        """
        self.cache_misses[cache_type] += 1
    
    def record_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        Record error occurrence
        
        Args:
            error_type: Type of error
            details: Error details
        """
        self.errors.append({
            'type': error_type,
            'details': details,
            'timestamp': time.time()
        })
        self.error_counts[error_type] += 1
    
    def record_llm_call(self, latency: float) -> None:
        """
        Record LLM API call
        
        Args:
            latency: Call latency in milliseconds
        """
        self.llm_calls += 1
        self.llm_call_latencies.append(latency)
    
    def get_latency_percentiles(self, time_window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate latency percentiles
        
        Args:
            time_window: Time window in seconds (None for all data)
            
        Returns:
            Dictionary with p50, p95, p99 latencies
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
        Get cache statistics
        
        Returns:
            Dictionary with cache stats by type
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
        Get error statistics
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Dictionary with error stats
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
        Calculate queries per second
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Queries per second
        """
        recent_queries = len(self._get_recent_latencies(time_window))
        return recent_queries / time_window if time_window > 0 else 0.0
    
    def _get_recent_latencies(self, time_window: Optional[int] = None) -> List[float]:
        """Get latencies within time window"""
        if time_window is None:
            return list(self.query_latencies)
        
        cutoff_time = time.time() - time_window
        recent = []
        
        for latency, timestamp in zip(self.query_latencies, self.query_timestamps):
            if timestamp >= cutoff_time:
                recent.append(latency)
        
        return recent
    
    def _get_recent_errors(self, time_window: Optional[int] = None) -> List[Dict]:
        """Get errors within time window"""
        if time_window is None:
            return list(self.errors)
        
        cutoff_time = time.time() - time_window
        return [e for e in self.errors if e['timestamp'] >= cutoff_time]


class QualityTracker:
    """
    Tracks retrieval quality metrics over time
    
    Monitors:
    - Average similarity scores
    - Result diversity
    - Marker distribution (deprecated, security, etc.)
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize quality tracker
        
        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        
        # Quality metrics
        self.similarity_scores: deque = deque(maxlen=max_samples)
        self.diversity_scores: deque = deque(maxlen=max_samples)
        self.result_counts: deque = deque(maxlen=max_samples)
        
        # Marker tracking
        self.marker_counts = defaultdict(int)
        
        # Timestamps
        self.timestamps: deque = deque(maxlen=max_samples)
    
    def record_results(self, results: List[Any]) -> None:
        """
        Record retrieval results for quality tracking
        
        Args:
            results: List of retrieval results
        """
        if not results:
            return
        
        # Calculate average similarity
        similarities = [r.similarity for r in results if hasattr(r, 'similarity')]
        if similarities:
            avg_similarity = np.mean(similarities)
            self.similarity_scores.append(avg_similarity)
        
        # Calculate diversity (simplified: unique file paths / total results)
        file_paths = [r.file_path for r in results if hasattr(r, 'file_path')]
        if file_paths:
            diversity = len(set(file_paths)) / len(file_paths)
            self.diversity_scores.append(diversity)
        
        # Track result count
        self.result_counts.append(len(results))
        
        # Track markers
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
        Get quality metrics
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Dictionary with quality metrics
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
        baseline_window: int = 3600,  # 1 hour
        current_window: int = 300,     # 5 minutes
        threshold: float = 0.15        # 15% drop
    ) -> Optional[Dict[str, Any]]:
        """
        Detect quality degradation
        
        Args:
            baseline_window: Baseline time window in seconds
            current_window: Current time window in seconds
            threshold: Degradation threshold (0.15 = 15% drop)
            
        Returns:
            Alert dictionary if degradation detected, None otherwise
        """
        baseline_metrics = self.get_quality_metrics(baseline_window)
        current_metrics = self.get_quality_metrics(current_window)
        
        if baseline_metrics['sample_count'] < 10 or current_metrics['sample_count'] < 5:
            return None  # Not enough data
        
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
        """Get values within time window"""
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
    Main monitoring interface for retrieval system
    
    Combines metrics collection, quality tracking, and alerting
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize retrieval monitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            max_samples=self.config.max_samples
        ) if self.config.metrics_enabled else None
        
        self.quality_tracker = QualityTracker(
            max_samples=self.config.max_samples
        ) if self.config.quality_tracking_enabled else None
        
        # Alert history
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
        Record query execution
        
        Args:
            query: Query text
            latency: Total latency in milliseconds
            results: Retrieval results
            cache_hits: Dictionary of cache hit/miss by type
        """
        if not self.config.enabled:
            return
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_query(latency)
            
            # Record cache hits/misses
            if cache_hits:
                for cache_type, hit in cache_hits.items():
                    if hit:
                        self.metrics_collector.record_cache_hit(cache_type)
                    else:
                        self.metrics_collector.record_cache_miss(cache_type)
        
        # Record quality
        if self.quality_tracker:
            self.quality_tracker.record_results(results)
        
        # Check for alerts
        if self.config.alerts_enabled:
            self._check_alerts()
    
    def record_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        Record error occurrence
        
        Args:
            error_type: Type of error
            details: Error details
        """
        if not self.config.enabled or not self.metrics_collector:
            return
        
        self.metrics_collector.record_error(error_type, details)
        logger.warning(f"Error recorded: {error_type}")
    
    def record_llm_call(self, latency: float) -> None:
        """
        Record LLM API call
        
        Args:
            latency: Call latency in milliseconds
        """
        if not self.config.enabled or not self.metrics_collector:
            return
        
        self.metrics_collector.record_llm_call(latency)
    
    def get_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive metrics
        
        Args:
            time_window: Time window in seconds (None for all data)
            
        Returns:
            Dictionary with all metrics
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
        Get monitoring summary
        
        Returns:
            Summary dictionary
        """
        return {
            'status': 'healthy',
            'metrics_1h': self.get_metrics(3600),
            'metrics_5m': self.get_metrics(300),
            'recent_alerts': list(self.alerts)[-10:] if self.alerts else []
        }
    
    def _check_alerts(self) -> None:
        """Check for alert conditions"""
        if not self.config.alerts_enabled:
            return
        
        # Check quality degradation
        if self.quality_tracker:
            alert = self.quality_tracker.detect_quality_degradation(
                threshold=self.config.quality_degradation_threshold
            )
            if alert:
                self._trigger_alert(alert)
        
        # Check error rate
        if self.metrics_collector:
            error_stats = self.metrics_collector.get_error_stats(300)  # Last 5 minutes
            if error_stats['error_rate'] > self.config.error_rate_threshold:
                self._trigger_alert({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'error_rate': error_stats['error_rate'],
                    'threshold': self.config.error_rate_threshold,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Check latency
        if self.metrics_collector:
            latency = self.metrics_collector.get_latency_percentiles(300)  # Last 5 minutes
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
        Trigger an alert
        
        Args:
            alert: Alert dictionary
        """
        self.alerts.append(alert)
        logger.warning(f"Alert triggered: {alert['type']} - {alert}")
