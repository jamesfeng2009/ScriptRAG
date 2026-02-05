"""技能使用监控器 - 跟踪技能切换和使用模式

本模块实现技能使用情况的全面监控：
1. 技能切换频率和模式
2. 技能使用统计和分布
3. 触发器效果分析
4. 性能指标跟踪
5. 使用模式分析和告警
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path

from src.domain.skills import (
    SkillLayer,
    SkillConfig,
    SKILLS,
    default_skill_manager,
)
from src.domain.skill_router import SignalRouter, ContentAnalysis


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SkillMetric:
    """技能指标"""
    skill_name: str
    layer: str
    usage_count: int
    total_confidence: float
    avg_confidence: float
    avg_selection_time_ms: float
    transition_count: int
    last_used: Optional[str]


@dataclass
class TransitionMetric:
    """切换指标"""
    from_skill: Optional[str]
    to_skill: str
    timestamp: str
    reason: str
    confidence: float
    routing_time_ms: float


@dataclass
class TriggerMetric:
    """触发器指标"""
    trigger_type: str
    trigger_value: str
    matched_count: int
    avg_confidence: float
    skills_triggered: List[str]


class SkillMonitoringConfig:
    """监控配置"""

    def __init__(
        self,
        enabled: bool = True,
        retention_days: int = 30,
        max_transitions: int = 10000,
        metrics_interval: int = 60,
        alert_enabled: bool = True,
        low_confidence_threshold: float = 0.5,
        high_switch_rate_threshold: float = 0.8,
        log_file: str = "logs/skill_usage.log"
    ):
        self.enabled = enabled
        self.retention_days = retention_days
        self.max_transitions = max_transitions
        self.metrics_interval = metrics_interval
        self.alert_enabled = alert_enabled
        self.low_confidence_threshold = low_confidence_threshold
        self.high_switch_rate_threshold = high_switch_rate_threshold
        self.log_file = log_file

        self._setup_logging()

    def _setup_logging(self) -> None:
        """配置日志"""
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )

        if not logger.handlers:
            logger.addHandler(file_handler)


class SkillMonitor:
    """技能使用监控器

    功能：
    - 记录技能选择和切换
    - 计算使用统计
    - 分析触发器效果
    - 生成告警
    """

    def __init__(
        self,
        config: SkillMonitoringConfig = None
    ):
        """初始化监控器"""
        self.config = config or SkillMonitoringConfig()

        self._skill_metrics: Dict[str, SkillMetric] = {}
        self._transition_history: deque = deque(maxlen=self.config.max_transitions)
        self._trigger_metrics: Dict[str, TriggerMetric] = {}
        self._confidence_history: deque = deque(maxlen=1000)
        self._session_starts: Dict[str, datetime] = {}
        self._skill_session_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._initialize_metrics()

        logger.info("SkillMonitor initialized")

    def _initialize_metrics(self) -> None:
        """初始化技能指标"""
        for skill_name, config in SKILLS.items():
            self._skill_metrics[skill_name] = SkillMetric(
                skill_name=skill_name,
                layer=config.layer.value,
                usage_count=0,
                total_confidence=0.0,
                avg_confidence=0.0,
                avg_selection_time_ms=0.0,
                transition_count=0,
                last_used=None,
            )

    def record_selection(
        self,
        skill_name: str,
        confidence: float,
        selection_time_ms: float,
        analysis: ContentAnalysis = None,
        session_id: Optional[str] = None
    ) -> None:
        """记录技能选择

        Args:
            skill_name: 选择的技能
            confidence: 置信度
            selection_time_ms: 选择耗时（毫秒）
            analysis: 内容分析结果
            session_id: 会话 ID
        """
        if not self.config.enabled:
            return

        metric = self._skill_metrics.get(skill_name)
        if metric:
            metric.usage_count += 1
            metric.total_confidence += confidence
            metric.avg_confidence = metric.total_confidence / metric.usage_count
            metric.avg_selection_time_ms = (
                (metric.avg_selection_time_ms * (metric.usage_count - 1) + selection_time_ms)
                / metric.usage_count
            )
            metric.last_used = datetime.now().isoformat()

        self._confidence_history.append({
            'skill': skill_name,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

        if session_id:
            self._skill_session_counts[session_id][skill_name] += 1

        if analysis:
            self._update_trigger_metrics(skill_name, analysis)

        self._log_selection(skill_name, confidence, selection_time_ms)

    def record_transition(
        self,
        from_skill: Optional[str],
        to_skill: str,
        reason: str,
        confidence: float,
        routing_time_ms: float
    ) -> None:
        """记录技能切换

        Args:
            from_skill: 源技能
            to_skill: 目标技能
            reason: 切换原因
            confidence: 置信度
            routing_time_ms: 路由耗时
        """
        if not self.config.enabled:
            return

        transition = TransitionMetric(
            from_skill=from_skill,
            to_skill=to_skill,
            timestamp=datetime.now().isoformat(),
            reason=reason,
            confidence=confidence,
            routing_time_ms=routing_time_ms
        )

        self._transition_history.append(transition)

        if to_skill in self._skill_metrics:
            self._skill_metrics[to_skill].transition_count += 1

        self._log_transition(transition)

        if self.config.alert_enabled:
            self._check_alerts(transition)

    def _update_trigger_metrics(
        self,
        skill_name: str,
        analysis: ContentAnalysis
    ) -> None:
        """更新触发器指标"""
        for marker in analysis.markers:
            key = f"marker:{marker}"
            self._update_single_trigger(key, marker, skill_name, analysis.complexity)

        for keyword in analysis.keywords:
            key = f"keyword:{keyword}"
            self._update_single_trigger(key, keyword, skill_name, analysis.complexity)

        if analysis.domain:
            key = f"domain:{analysis.domain}"
            self._update_single_trigger(key, analysis.domain, skill_name, analysis.complexity)

        if analysis.complexity:
            key = f"complexity:{analysis.complexity}"
            self._update_single_trigger(key, analysis.complexity, skill_name, analysis.complexity)

    def _update_single_trigger(
        self,
        key: str,
        value: str,
        skill_name: str,
        complexity: str
    ) -> None:
        """更新单个触发器指标"""
        if key not in self._trigger_metrics:
            self._trigger_metrics[key] = TriggerMetric(
                trigger_type=key.split(':')[0],
                trigger_value=value,
                matched_count=0,
                avg_confidence=0.0,
                skills_triggered=set()
            )

        metric = self._trigger_metrics[key]
        metric.matched_count += 1
        metric.skills_triggered.add(skill_name)

    def _log_selection(
        self,
        skill_name: str,
        confidence: float,
        time_ms: float
    ) -> None:
        """日志记录选择"""
        logger.info(
            f"Skill selected: {skill_name} "
            f"(confidence: {confidence:.2f}, time: {time_ms:.1f}ms)"
        )

    def _log_transition(self, transition: TransitionMetric) -> None:
        """日志记录切换"""
        logger.info(
            f"Skill transition: {transition.from_skill} -> {transition.to_skill} "
            f"(reason: {transition.reason}, confidence: {transition.confidence:.2f})"
        )

    def _check_alerts(self, transition: TransitionMetric) -> None:
        """检查告警条件"""
        alerts = []

        if transition.confidence < self.config.low_confidence_threshold:
            alerts.append((
                AlertLevel.WARNING,
                f"Low confidence transition to {transition.to_skill}: {transition.confidence:.2f}"
            ))

        recent_transitions = [
            t for t in self._transition_history
            if datetime.fromisoformat(t.timestamp) > datetime.now() - timedelta(minutes=5)
        ]
        if len(recent_transitions) > 5:
            alerts.append((
                AlertLevel.INFO,
                f"High switch rate detected: {len(recent_transitions)} switches in 5 minutes"
            ))

        for level, message in alerts:
            self._send_alert(level, message)

    def _send_alert(self, level: AlertLevel, message: str) -> None:
        """发送告警"""
        alert_data = {
            'level': level.value,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

        if level == AlertLevel.ERROR or level == AlertLevel.CRITICAL:
            logger.error(f"ALERT: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")

    def get_skill_statistics(self) -> Dict[str, Any]:
        """获取技能统计"""
        total_usage = sum(m.usage_count for m in self._skill_metrics.values())

        layer_distribution = defaultdict(int)
        for skill, metric in self._skill_metrics.items():
            layer_distribution[metric.layer] += metric.usage_count

        top_skills = sorted(
            self._skill_metrics.values(),
            key=lambda x: x.usage_count,
            reverse=True
        )[:5]

        avg_confidence = (
            sum(m.avg_confidence for m in self._skill_metrics.values())
            / len(self._skill_metrics) if self._skill_metrics else 0
        )

        return {
            'total_usage': total_usage,
            'layer_distribution': dict(layer_distribution),
            'top_skills': [
                {
                    'name': s.skill_name,
                    'count': s.usage_count,
                    'avg_confidence': s.avg_confidence,
                    'layer': s.layer
                }
                for s in top_skills
            ],
            'average_confidence': avg_confidence,
            'unique_skills_used': sum(1 for m in self._skill_metrics.values() if m.usage_count > 0),
        }

    def get_transition_statistics(self) -> Dict[str, Any]:
        """获取切换统计"""
        if not self._transition_history:
            return {
                'total_transitions': 0,
                'avg_confidence': 0,
                'avg_routing_time_ms': 0,
                'most_common_transitions': []
            }

        avg_confidence = sum(t.confidence for t in self._transition_history) / len(self._transition_history)
        avg_time = sum(t.routing_time_ms for t in self._transition_history) / len(self._transition_history)

        transition_counts = defaultdict(int)
        for t in self._transition_history:
            key = f"{t.from_skill or 'None'} -> {t.to_skill}"
            transition_counts[key] += 1

        most_common = sorted(
            transition_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_transitions': len(self._transition_history),
            'avg_confidence': avg_confidence,
            'avg_routing_time_ms': avg_time,
            'most_common_transitions': [
                {'transition': k, 'count': v} for k, v in most_common
            ],
            'recent_transitions': [
                {
                    'from': t.from_skill,
                    'to': t.to_skill,
                    'reason': t.reason,
                    'timestamp': t.timestamp
                }
                for t in list(self._transition_history)[-10:]
            ]
        }

    def get_trigger_effectiveness(self) -> Dict[str, Any]:
        """获取触发器效果"""
        effectiveness = []
        for key, metric in self._trigger_metrics.items():
            effectiveness.append({
                'trigger': key,
                'matched_count': metric.matched_count,
                'skills_triggered': list(metric.skills_triggered),
                'type': metric.trigger_type
            })

        effectiveness.sort(key=lambda x: x['matched_count'], reverse=True)

        return {
            'total_triggers': len(self._trigger_metrics),
            'effectiveness': effectiveness[:20],
            'marker_triggers': [
                e for e in effectiveness if e['trigger'].startswith('marker')
            ],
            'keyword_triggers': [
                e for e in effectiveness if e['trigger'].startswith('keyword')
            ],
            'domain_triggers': [
                e for e in effectiveness if e['trigger'].startswith('domain')
            ]
        }

    def get_confidence_distribution(self) -> Dict[str, Any]:
        """获取置信度分布"""
        if not self._confidence_history:
            return {'distribution': {}, 'avg': 0}

        confidences = [c['confidence'] for c in self._confidence_history]

        distribution = {
            '0.0-0.2': sum(1 for c in confidences if 0.0 <= c < 0.2),
            '0.2-0.4': sum(1 for c in confidences if 0.2 <= c < 0.4),
            '0.4-0.6': sum(1 for c in confidences if 0.4 <= c < 0.6),
            '0.6-0.8': sum(1 for c in confidences if 0.6 <= c < 0.8),
            '0.8-1.0': sum(1 for c in confidences if 0.8 <= c <= 1.0),
        }

        return {
            'distribution': distribution,
            'avg': sum(confidences) / len(confidences) if confidences else 0,
            'total_records': len(confidences)
        }

    def get_session_report(self, session_id: str) -> Dict[str, Any]:
        """获取会话报告"""
        if session_id not in self._skill_session_counts:
            return {'error': 'Session not found'}

        skill_counts = self._skill_session_counts[session_id]
        total = sum(skill_counts.values())

        return {
            'session_id': session_id,
            'total_queries': total,
            'skill_distribution': dict(skill_counts),
            'dominant_skill': max(skill_counts.items(), key=lambda x: x[1])[0] if skill_counts else None,
        }

    def export_report(self, output_path: str) -> None:
        """导出完整报告"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'skill_statistics': self.get_skill_statistics(),
            'transition_statistics': self.get_transition_statistics(),
            'trigger_effectiveness': self.get_trigger_effectiveness(),
            'confidence_distribution': self.get_confidence_distribution(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Report exported to: {output_path}")

    def reset_session(self, session_id: str) -> None:
        """重置会话数据"""
        if session_id in self._skill_session_counts:
            del self._skill_session_counts[session_id]

    def clear_all(self) -> None:
        """清除所有数据"""
        self._transition_history.clear()
        self._confidence_history.clear()
        self._skill_session_counts.clear()
        self._trigger_metrics.clear()

        for metric in self._skill_metrics.values():
            metric.usage_count = 0
            metric.total_confidence = 0
            metric.avg_confidence = 0
            metric.transition_count = 0
            metric.last_used = None

        logger.info("All monitoring data cleared")


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self._alert_rules: List = []
        self._alert_callbacks: List = []

    def add_rule(self, name: str, condition: callable, threshold: Any) -> None:
        """添加告警规则"""
        self._alert_rules.append({
            'name': name,
            'condition': condition,
            'threshold': threshold
        })

    def add_callback(self, callback: callable) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def check_and_alert(self, monitor: SkillMonitor) -> List[Dict[str, Any]]:
        """检查并发送告警"""
        alerts = []

        stats = monitor.get_skill_statistics()
        transitions = monitor.get_transition_statistics()

        if stats['average_confidence'] < 0.5:
            alerts.append({
                'level': 'warning',
                'message': f'Average confidence is low: {stats["average_confidence"]:.2f}'
            })

        if transitions['total_transitions'] > 0 and transitions['avg_routing_time_ms'] > 100:
            alerts.append({
                'level': 'info',
                'message': f'Average routing time is high: {transitions["avg_routing_time_ms"]:.1f}ms'
            })

        for callback in self._alert_callbacks:
            try:
                callback(alerts)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        return alerts


def create_skill_monitor(
    log_file: str = "logs/skill_usage.log",
    retention_days: int = 30
) -> SkillMonitor:
    """创建技能监控器工厂函数"""
    config = SkillMonitoringConfig(
        log_file=log_file,
        retention_days=retention_days,
        alert_enabled=True
    )
    return SkillMonitor(config)


default_monitor = SkillMonitor()
default_alert_manager = AlertManager()
