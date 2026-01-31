# Quick Start Guide - Enhanced RAG System

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install watchdog cachetools numpy
```

### 2. Basic Usage

```python
from src.services.retrieval_service import RetrievalService, RetrievalConfig
from src.services.cache.retrieval_cache import RetrievalCache
from src.services.monitoring.retrieval_monitor import RetrievalMonitor

# Create retrieval service with all enhancements
retrieval_service = RetrievalService(
    vector_db_service=your_vector_db,
    llm_service=your_llm_service,
    config=RetrievalConfig(),  # Uses smart defaults
    cache=RetrievalCache(),     # Automatic caching
    monitor=RetrievalMonitor()  # Automatic monitoring
)

# Use it
results = await retrieval_service.hybrid_retrieve(
    workspace_id="my_workspace",
    query="how to implement authentication",
    top_k=5
)
```

That's it! You now have:
- ✅ Query expansion
- ✅ Multi-factor reranking
- ✅ Intelligent caching
- ✅ Real-time monitoring

---

## Common Configurations

### High Performance (Aggressive Caching)
```python
from src.services.cache.retrieval_cache import CacheConfig

cache = RetrievalCache(CacheConfig(
    query_expansion_ttl=7200,  # 2 hours
    embedding_ttl=172800,       # 48 hours
    result_ttl=600              # 10 minutes
))
```

### High Quality (More Expansions, Strict Filtering)
```python
config = RetrievalConfig(
    enable_query_expansion=True,
    expansion_top_k=15,         # More candidates
    enable_reranking=True,
    enable_diversity=True,
    diversity_threshold=0.75    # Stricter filtering
)
```

### Low Latency (Minimal Processing)
```python
config = RetrievalConfig(
    enable_query_expansion=False,  # Skip expansion
    enable_reranking=True,
    enable_diversity=False,        # Skip filtering
    rerank_top_k=3                 # Fewer results
)
```

---

## Monitoring Your System

### Get Real-Time Metrics
```python
# Get last 5 minutes of metrics
metrics = retrieval_service.monitor.get_metrics(time_window=300)

print(f"P95 Latency: {metrics['performance']['latency']['p95']:.0f}ms")
print(f"Cache Hit Rate: {metrics['performance']['cache']['embedding']['hit_rate']:.1%}")
print(f"Avg Quality: {metrics['quality']['avg_similarity']:.2f}")
```

### Check Cache Performance
```python
stats = retrieval_service.cache.get_stats()

for cache_type, cache_stats in stats.items():
    if cache_stats:
        print(f"{cache_type}: {cache_stats['hit_rate']:.1%} hit rate")
```

### Get System Summary
```python
summary = retrieval_service.monitor.get_summary()
print(f"Status: {summary['status']}")
print(f"Alerts: {len(summary['recent_alerts'])}")
```

---

## Cache Management

### Invalidate After Updates
```python
# After updating workspace code
retrieval_service.cache.invalidate_workspace("workspace_id")
```

### Clear All Caches
```python
# During development or testing
retrieval_service.cache.clear_all()
```

---

## Skill Configuration (Phase 1)

### Load Custom Skills
```python
from src.domain.skill_loader import SkillConfigLoader

loader = SkillConfigLoader()
skills = loader.load_from_file("config/skills.yaml")

# Use with Writer Agent
from src.domain.agents.writer import Writer
writer = Writer(llm_service, skill_manager=skills)
```

### Edit Skills
Just edit `config/skills.yaml` - changes are automatically reloaded in development!

---

## Troubleshooting

### High Latency?
1. Check cache hit rates: `cache.get_stats()`
2. Disable query expansion if not needed
3. Reduce `expansion_top_k`

### Low Quality Results?
1. Enable query expansion: `enable_query_expansion=True`
2. Increase `expansion_top_k`
3. Adjust reranking weights
4. Lower diversity threshold

### High LLM Costs?
1. Increase cache TTLs
2. Increase cache sizes
3. Check cache hit rates
4. Consider disabling query expansion for simple queries

### Memory Issues?
1. Reduce cache sizes in `CacheConfig`
2. Reduce `max_samples` in `MonitoringConfig`
3. Decrease TTLs to expire entries faster

---

## Performance Tips

### Warm Up Caches
```python
# Run common queries on startup
common_queries = ["authentication", "database", "api"]
for query in common_queries:
    await retrieval_service.hybrid_retrieve("workspace", query)
```

### Batch Operations
```python
# Process multiple queries efficiently
queries = ["query1", "query2", "query3"]
results = await asyncio.gather(*[
    retrieval_service.hybrid_retrieve("workspace", q)
    for q in queries
])
```

### Monitor Performance
```python
import time

start = time.time()
results = await retrieval_service.hybrid_retrieve(...)
latency = (time.time() - start) * 1000

print(f"Query took {latency:.0f}ms")
```

---

## Configuration Reference

### RetrievalConfig
```python
RetrievalConfig(
    # Vector search
    vector_top_k=5,
    vector_similarity_threshold=0.7,
    
    # Keyword search
    keyword_boost_factor=1.5,
    
    # Merging
    vector_weight=0.6,
    keyword_weight=0.4,
    
    # Query expansion (Phase 2)
    enable_query_expansion=True,
    expansion_top_k=10,
    
    # Reranking (Phase 2)
    enable_reranking=True,
    rerank_top_k=5,
    
    # Diversity (Phase 2)
    enable_diversity=True,
    diversity_threshold=0.85,
    
    # Monitoring (Phase 2)
    enable_quality_monitoring=True
)
```

### CacheConfig (Phase 3)
```python
CacheConfig(
    enabled=True,
    
    # Query expansion cache
    query_expansion_enabled=True,
    query_expansion_ttl=3600,      # 1 hour
    query_expansion_max_size=1000,
    
    # Embedding cache
    embedding_enabled=True,
    embedding_ttl=86400,           # 24 hours
    embedding_max_size=10000,
    
    # Result cache
    result_enabled=True,
    result_ttl=300,                # 5 minutes
    result_max_size=500
)
```

### MonitoringConfig (Phase 3)
```python
MonitoringConfig(
    enabled=True,
    
    # Metrics
    metrics_enabled=True,
    max_samples=10000,
    
    # Quality tracking
    quality_tracking_enabled=True,
    
    # Alerts
    alerts_enabled=True,
    quality_degradation_threshold=0.15,  # 15% drop
    error_rate_threshold=0.05,           # 5% errors
    latency_p95_threshold=1000           # 1 second
)
```

---

## Testing

### Run All Tests
```bash
# All phases
pytest tests/unit/test_skill_loader.py \
       tests/unit/test_query_expansion.py \
       tests/unit/test_reranker.py \
       tests/unit/test_retrieval_cache.py \
       tests/unit/test_retrieval_monitor.py -v

# Should see: 86 passed
```

### Run Specific Phase Tests
```bash
# Phase 1
pytest tests/unit/test_skill_loader.py -v

# Phase 2
pytest tests/unit/test_query_expansion.py tests/unit/test_reranker.py -v

# Phase 3
pytest tests/unit/test_retrieval_cache.py tests/unit/test_retrieval_monitor.py -v
```

---

## Documentation

- **Phase 1**: `docs/SKILL_CONFIGURATION_GUIDE.md`
- **Phase 2**: `PHASE2_COMPLETION_REPORT.md`
- **Phase 3**: `PHASE3_COMPLETION_REPORT.md`
- **Complete Summary**: `COMPLETE_ENHANCEMENT_SUMMARY.md`

---

## Support

For issues or questions:
1. Check the completion reports for detailed documentation
2. Review test files for usage examples
3. Check configuration examples above
4. Monitor system metrics for debugging

---

*Quick Start Guide - All Phases Complete ✅*
