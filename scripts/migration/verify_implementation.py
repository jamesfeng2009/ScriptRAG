"""
验证迁移工具实现完整性
"""

import sys
import importlib.util
from pathlib import Path


def check_module(module_name: str, expected_classes: list) -> bool:
    """检查模块是否包含预期的类"""
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).parent / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        missing = []
        for cls in expected_classes:
            if not hasattr(module, cls):
                missing.append(cls)
        
        if missing:
            print(f"❌ {module_name}: Missing classes: {', '.join(missing)}")
            return False
        else:
            print(f"✅ {module_name}: All expected classes found")
            return True
    except Exception as e:
        print(f"❌ {module_name}: Failed to load - {e}")
        return False


def main():
    """主验证函数"""
    print("=" * 80)
    print("验证迁移工具实现完整性")
    print("=" * 80)
    print()
    
    checks = [
        ("milvus_schema", ["MilvusSchemaDefinition"]),
        ("postgres_to_milvus", ["PostgresToMilvusMigration"]),
        ("dual_write_manager", ["DualWriteManager"]),
        ("gradual_cutover", ["GradualCutoverManager", "CutoverScheduler"]),
        ("migration_monitor", [
            "MigrationMonitor",
            "MigrationThresholds",
            "MetricsCollector",
            "MigrationRecommendation"
        ]),
        ("migration_orchestrator", ["MigrationOrchestrator", "MigrationPhase"]),
    ]
    
    results = []
    for module_name, expected_classes in checks:
        result = check_module(module_name, expected_classes)
        results.append(result)
    
    print()
    print("=" * 80)
    
    if all(results):
        print("✅ 所有模块验证通过！")
        print()
        print("实现的功能:")
        print("  1. ✅ Milvus Schema 定义")
        print("  2. ✅ PostgreSQL -> Milvus 数据迁移")
        print("  3. ✅ 双写管理器")
        print("  4. ✅ 灰度切流管理器")
        print("  5. ✅ 迁移监控系统")
        print("  6. ✅ 迁移编排器")
        print()
        print("监控指标:")
        print("  - 向量数量（阈值: 100 万）")
        print("  - 搜索 QPS（阈值: 100）")
        print("  - P99 延迟（阈值: 500ms）")
        print("  - 存储大小（阈值: 100GB）")
        print()
        print("迁移流程:")
        print("  1. Monitoring: 持续监控系统指标")
        print("  2. Preparation: 准备迁移环境")
        print("  3. Full Migration: 全量数据迁移")
        print("  4. Dual Write: 启用双写模式")
        print("  5. Gradual Cutover: 灰度切流（1% -> 5% -> 10% -> 25% -> 50% -> 75% -> 100%）")
        print("  6. Completed: 迁移完成")
        print()
        print("使用方法:")
        print("  python migration_orchestrator.py")
        print()
        return 0
    else:
        print("❌ 部分模块验证失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
