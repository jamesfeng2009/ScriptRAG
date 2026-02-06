"""Knowledge Services - 知识图谱服务

向后兼容导出
"""

from .knowledge_graph_service import (
    KnowledgeGraphService,
    KnowledgeNodeModel as Entity,
    KnowledgeRelationModel as Relation
)
from .graprag_engine import (
    GraphRAGEngine,
    DocumentDependencyGraph,
    GraphTraversalEngine,
    EntityExtractor,
    GraphNode,
    GraphEdge,
    NodeType,
    RelationType
)

__all__ = [
    "KnowledgeGraphService",
    "Entity",
    "Relation",
    "GraphRAGEngine",
    "DocumentDependencyGraph",
    "GraphTraversalEngine",
    "EntityExtractor",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "RelationType",
]
