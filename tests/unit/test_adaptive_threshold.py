"""Unit Tests for Adaptive Threshold Strategy"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.retrieval.adaptive_threshold import (
    AdaptiveThresholdStrategy,
    AdaptiveThresholdConfig,
    CliffEdgeCutoff,
    HybridCutoffStrategy,
    ThresholdAnalysisResult
)


class TestAdaptiveThresholdStrategy:
    """Test cases for AdaptiveThresholdStrategy"""
    
    def test_calculate_adaptive_threshold_simple_query(self):
        """简单查询应该提高阈值"""
        config = AdaptiveThresholdConfig(base_min_score=0.65)
        strategy = AdaptiveThresholdStrategy(config)
        
        # 简单查询 (complexity=0) 应该提高阈值
        threshold = strategy._calculate_adaptive_threshold(0.0)
        assert threshold > 0.65
        assert threshold <= 0.75  # 调整预期上限
    
    def test_calculate_adaptive_threshold_complex_query(self):
        """复杂查询应该降低阈值"""
        config = AdaptiveThresholdConfig(base_min_score=0.65)
        strategy = AdaptiveThresholdStrategy(config)
        
        # 复杂查询 (complexity=1) 应该降低阈值
        threshold = strategy._calculate_adaptive_threshold(1.0)
        assert threshold < 0.65
        assert threshold >= 0.55
    
    def test_detect_cliff(self):
        """测试悬崖检测"""
        strategy = AdaptiveThresholdStrategy()
        
        # 无悬崖
        results = [0.95, 0.90, 0.85, 0.80]
        cliff = strategy._detect_cliff(results, [])
        assert not cliff['detected']
        
        # 有悬崖（下降超过20%）
        results = [0.95, 0.70, 0.68, 0.65]
        cliff = strategy._detect_cliff(results, [])
        assert cliff['detected']
        assert cliff['index'] == 1
        assert cliff['drop_rate'] > 0.20
    
    def test_analyze_and_filter_no_results(self):
        """空结果测试"""
        strategy = AdaptiveThresholdStrategy()
        
        indices, result = strategy.analyze_and_filter([], 0.5)
        
        assert indices == []
        assert result.total_candidates == 0
        assert result.retained_count == 0
    
    def test_analyze_and_filter_with_cliff(self):
        """悬崖截断测试"""
        strategy = AdaptiveThresholdStrategy()
        
        # 分数：0.95, 0.72 (悬崖), 0.70, 0.68
        results = [0.95, 0.72, 0.70, 0.68]
        indices, result = strategy.analyze_and_filter(results, 0.5)
        
        # 应该截断到悬崖位置
        assert result.cliff_detected
        assert result.retained_count == 1
        assert result.cliff_index == 1
    
    def test_analyze_and_filter_with_min_score(self):
        """最低分数限制测试"""
        config = AdaptiveThresholdConfig(base_min_score=0.65)
        strategy = AdaptiveThresholdStrategy(config)
        
        # 分数都高于阈值
        results = [0.95, 0.80, 0.70, 0.68]
        indices, result = strategy.analyze_and_filter(results, 0.5)
        
        assert len(indices) == 4
        assert not result.min_score_enforced


class TestCliffEdgeCutoff:
    """Test cases for CliffEdgeCutoff"""
    
    def test_cutoff_no_cliff(self):
        """无悬崖时返回所有结果"""
        cutter = CliffEdgeCutoff(drop_threshold=0.20, min_score=0.55, max_results=10)
        
        results = [0.95, 0.90, 0.85, 0.80]
        indices, cliff_index = cutter.cutoff(results)
        
        assert indices == [0, 1, 2, 3]
        assert cliff_index is None
    
    def test_cutoff_with_cliff(self):
        """悬崖截断"""
        cutter = CliffEdgeCutoff(drop_threshold=0.20, min_score=0.55, max_results=10)
        
        # 从0.95降到0.70，下降26% > 20%
        results = [0.95, 0.70, 0.68, 0.65]
        indices, cliff_index = cutter.cutoff(results)
        
        assert indices == [0]
        assert cliff_index == 1
    
    def test_cutoff_max_results(self):
        """最大数量限制"""
        cutter = CliffEdgeCutoff(drop_threshold=0.20, min_score=0.55, max_results=3)
        
        results = [0.95, 0.90, 0.85, 0.80, 0.75]
        indices, cliff_index = cutter.cutoff(results)
        
        assert len(indices) == 3
    
    def test_cutoff_empty_results(self):
        """空结果"""
        cutter = CliffEdgeCutoff()
        indices, cliff_index = cutter.cutoff([])
        
        assert indices == []
        assert cliff_index is None


class TestHybridCutoffStrategy:
    """Test cases for HybridCutoffStrategy"""
    
    def test_filter_no_results(self):
        """空结果"""
        strategy = HybridCutoffStrategy()
        indices, metadata = strategy.filter([])
        
        assert indices == []
        assert metadata['total'] == 0
    
    def test_filter_below_min_score(self):
        """低于最低分数（无悬崖情况）"""
        strategy = HybridCutoffStrategy(
            min_score=0.60,
            drop_threshold=0.20,
            max_results=10
        )
        
        # 使用无悬崖的分数：逐步下降，不超过20%
        results = [0.95, 0.85, 0.75, 0.65, 0.55]
        indices, metadata = strategy.filter(results)
        
        # 0.65 >= 0.60, 0.55 < 0.60
        # 所以索引 0, 1, 2, 3 应该保留
        assert 3 in indices  # 0.65 >= 0.60
        assert 4 not in indices  # 0.55 < 0.60, filtered out
        assert metadata['min_score_enforced']
    
    def test_filter_with_cliff(self):
        """悬崖优先于最低分数"""
        strategy = HybridCutoffStrategy(
            min_score=0.60,
            drop_threshold=0.20,
            max_results=10
        )
        
        # 悬崖在索引2，最低分数在索引3
        results = [0.95, 0.85, 0.60, 0.55]
        indices, metadata = strategy.filter(results)
        
        # 悬崖检测应该触发
        assert metadata['cliff_detected']
        # 只保留悬崖前的结果
        assert len(indices) <= 2


class TestAdaptiveThresholdConfig:
    """Test cases for AdaptiveThresholdConfig"""
    
    def test_default_config(self):
        """默认配置"""
        config = AdaptiveThresholdConfig()
        
        assert config.base_min_score == 0.65
        assert config.cliff_drop_threshold == 0.20
        assert config.min_score_floor == 0.55
        assert config.max_results == 20
    
    def test_custom_config(self):
        """自定义配置"""
        config = AdaptiveThresholdConfig(
            base_min_score=0.70,
            cliff_drop_threshold=0.15,
            min_score_floor=0.50,
            max_results=30
        )
        
        assert config.base_min_score == 0.70
        assert config.cliff_drop_threshold == 0.15
        assert config.min_score_floor == 0.50
        assert config.max_results == 30


class TestThresholdAnalysisResult:
    """Test cases for ThresholdAnalysisResult"""
    
    def test_result_creation(self):
        """结果创建"""
        result = ThresholdAnalysisResult(
            applied_threshold=0.65,
            total_candidates=10,
            retained_count=5,
            cliff_detected=True,
            cliff_index=2,
            cliff_drop_rate=0.25,
            min_score_enforced=False,
            reasoning="Cliff detected"
        )
        
        assert result.applied_threshold == 0.65
        assert result.retained_count == 5
        assert result.cliff_detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
