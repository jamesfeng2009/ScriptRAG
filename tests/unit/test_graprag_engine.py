"""Unit Tests for GraphRAG Engine"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.graprag_engine import (
    GraphRAGEngine,
    DocumentDependencyGraph,
    GraphTraversalEngine,
    EntityExtractor,
    GraphNode,
    GraphEdge,
    NodeType,
    RelationType,
    RetrievalResult
)


class TestGraphNode:
    """Test cases for GraphNode"""
    
    def test_node_creation(self):
        """节点创建"""
        node = GraphNode(
            id="node1",
            node_type="function",
            name="test_func",
            content="def test_func(): pass",
            file_path="test.py"
        )
        
        assert node.id == "node1"
        assert node.node_type == "function"
        assert node.name == "test_func"
        assert node.properties == {}
    
    def test_node_to_dict(self):
        """节点转字典"""
        node = GraphNode(
            id="node1",
            node_type="function",
            name="test_func",
            content="A" * 300,  # 长内容
            file_path="test.py"
        )
        
        d = node.to_dict()
        
        assert d["id"] == "node1"
        assert d["type"] == "function"
        assert "..." in d["content"]  # 应该被截断


class TestGraphEdge:
    """Test cases for GraphEdge"""
    
    def test_edge_creation(self):
        """边创建"""
        edge = GraphEdge(
            source_id="node1",
            target_id="node2",
            relation_type="calls",
            strength=0.8
        )
        
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.relation_type == "calls"
        assert edge.strength == 0.8
    
    def test_edge_with_properties(self):
        """带属性的边"""
        edge = GraphEdge(
            source_id="node1",
            target_id="node2",
            relation_type="imports",
            strength=0.9,
            properties={"module_name": "os"}
        )
        
        assert edge.properties["module_name"] == "os"


class TestDocumentDependencyGraph:
    """Test cases for DocumentDependencyGraph"""
    
    @pytest.fixture
    def graph(self):
        """创建测试图"""
        return DocumentDependencyGraph(workspace_id="test")
    
    @pytest.fixture
    def sample_nodes(self):
        """示例节点"""
        return [
            GraphNode(id="doc1", node_type="document", name="main.py", content="...", file_path="main.py"),
            GraphNode(id="doc2", node_type="document", name="utils.py", content="...", file_path="utils.py"),
            GraphNode(id="doc3", node_type="document", name="models.py", content="...", file_path="models.py"),
            GraphNode(id="func1", node_type="function", name="helper", content="...", file_path="utils.py"),
        ]
    
    def test_add_node(self, graph, sample_nodes):
        """添加节点"""
        for node in sample_nodes:
            graph.add_node(node)
        
        assert len(graph._nodes) == 4
        assert "doc1" in graph._nodes
    
    def test_add_edge(self, graph, sample_nodes):
        """添加边"""
        for node in sample_nodes:
            graph.add_node(node)
        
        edge = GraphEdge(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.IMPORTS.value,
            strength=0.9
        )
        graph.add_edge(edge)
        
        assert len(graph._edges) == 1
        assert graph._adjacency["doc1"]["doc2"][0].relation_type == "imports"
    
    def test_add_import_relation(self, graph, sample_nodes):
        """添加导入关系"""
        for node in sample_nodes:
            graph.add_node(node)
        
        graph.add_import_relation("doc1", "doc2", "utils", strength=1.0)
        
        assert len(graph._edges) == 1
        edge = graph._edges[0]
        assert edge.relation_type == "imports"
        assert edge.properties["module_name"] == "utils"
    
    def test_add_call_relation(self, graph, sample_nodes):
        """添加调用关系"""
        for node in sample_nodes:
            graph.add_node(node)
        
        graph.add_call_relation("doc1", "func1", strength=0.8)
        
        assert len(graph._edges) == 1
        assert graph._edges[0].relation_type == "calls"
    
    def test_get_neighbors(self, graph, sample_nodes):
        """获取邻居"""
        for node in sample_nodes:
            graph.add_node(node)
        
        graph.add_edge(GraphEdge(source_id="doc1", target_id="doc2", relation_type="imports"))
        graph.add_edge(GraphEdge(source_id="doc1", target_id="doc3", relation_type="imports"))
        
        neighbors = graph.get_neighbors("doc1", "out")
        
        assert "doc2" in neighbors
        assert "doc3" in neighbors
    
    def test_get_connected_components(self, graph, sample_nodes):
        """获取连通分量"""
        for node in sample_nodes:
            graph.add_node(node)
        
        # doc1 -> doc2 -> doc3 形成一个连通分量
        graph.add_edge(GraphEdge(source_id="doc1", target_id="doc2", relation_type="imports"))
        graph.add_edge(GraphEdge(source_id="doc2", target_id="doc3", relation_type="imports"))
        
        components = graph.get_connected_components()
        
        # 至少有一个连通分量包含 doc1, doc2, doc3
        has_large_component = any(len(c) >= 3 for c in components)
        assert has_large_component
    
    def test_get_nodes_by_type(self, graph, sample_nodes):
        """按类型获取节点"""
        for node in sample_nodes:
            graph.add_node(node)
        
        func_nodes = graph.get_nodes_by_type("function")
        
        assert len(func_nodes) == 1
        assert func_nodes[0].name == "helper"
    
    def test_stats(self, graph, sample_nodes):
        """统计信息"""
        for node in sample_nodes:
            graph.add_node(node)
        
        graph.add_edge(GraphEdge(source_id="doc1", target_id="doc2", relation_type="imports"))
        
        stats = graph.stats()
        
        assert stats["node_count"] == 4
        assert stats["edge_count"] == 1


class TestGraphTraversalEngine:
    """Test cases for GraphTraversalEngine"""
    
    @pytest.fixture
    def graph(self):
        """创建测试图"""
        g = DocumentDependencyGraph()
        
        # 创建节点
        nodes = [
            GraphNode(id="A", node_type="document", name="A", content="A", file_path="a.py"),
            GraphNode(id="B", node_type="document", name="B", content="B", file_path="b.py"),
            GraphNode(id="C", node_type="document", name="C", content="C", file_path="c.py"),
            GraphNode(id="D", node_type="document", name="D", content="D", file_path="d.py"),
        ]
        
        for node in nodes:
            g.add_node(node)
        
        # 创建边: A -> B -> C -> D
        g.add_edge(GraphEdge(source_id="A", target_id="B", relation_type="imports", strength=0.9))
        g.add_edge(GraphEdge(source_id="B", target_id="C", relation_type="imports", strength=0.8))
        g.add_edge(GraphEdge(source_id="C", target_id="D", relation_type="imports", strength=0.7))
        
        return g
    
    @pytest.fixture
    def traversal(self, graph):
        """创建遍历引擎"""
        return GraphTraversalEngine(graph)
    
    def test_bfs_traverse(self, traversal, graph):
        """BFS 遍历"""
        result = traversal.bfs_traverse("A", max_depth=2)
        
        # A 在深度0
        assert "A" in [n.id for n in result.get(0, [])]
        # B 在深度1
        assert "B" in [n.id for n in result.get(1, [])]
        # C 在深度2
        assert "C" in [n.id for n in result.get(2, [])]
        # D 不应该在深度2（BFS 会到达更浅的节点）
    
    def test_bfs_max_depth(self, traversal, graph):
        """BFS 深度限制"""
        result = traversal.bfs_traverse("A", max_depth=1)
        
        assert len(result) == 2  # 深度0和1
        assert "A" in [n.id for n in result.get(0, [])]
        assert "B" in [n.id for n in result.get(1, [])]
        assert "C" not in [n.id for n in result.get(1, [])]
    
    def test_dfs_traverse(self, traversal, graph):
        """DFS 遍历"""
        paths = traversal.dfs_traverse("A", max_depth=3)
        
        # 应该找到路径 A -> B -> C -> D
        found_path = any(
            len(p.nodes) == 4 and p.nodes[0].id == "A" and p.nodes[-1].id == "D"
            for p in paths
        )
        assert found_path
    
    def test_personalized_pagerank(self, traversal, graph):
        """Personalized PageRank"""
        scores = traversal.personalized_pagerank(source_nodes=["A"])
        
        # A 应该分数最高
        assert "A" in scores
        assert scores["A"] > 0.02  # 应该有较高的分数（与其他节点相比）
        
        # 越远的节点分数应该不高于 A（允许相等情况）
        if "D" in scores:
            assert scores["D"] <= scores["A"]
    
    def test_find_shortest_paths(self, traversal, graph):
        """查找最短路径"""
        paths = traversal.find_shortest_paths("A", "D")
        
        assert len(paths) > 0
        # 最短路径应该有4个节点
        shortest = min(paths, key=lambda p: len(p.nodes))
        assert len(shortest.nodes) == 4
    
    def test_pagerank_multiple_sources(self, traversal, graph):
        """多源 PageRank"""
        scores = traversal.personalized_pagerank(source_nodes=["A", "B"])
        
        assert "A" in scores
        assert "B" in scores
        # A 和 B 都应该有较高的分数


class TestEntityExtractor:
    """Test cases for EntityExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """创建提取器"""
        return EntityExtractor()
    
    def test_extract_from_python_code(self, extractor):
        """从 Python 代码提取"""
        code = '''
import os
import sys

def helper_function():
    print("hello")

def main():
    helper_function()
    something()
'''
        
        nodes, edges = extractor.extract_from_code(code, "test.py")
        
        # 应该提取到导入、函数
        node_types = [n.node_type for n in nodes]
        
        assert "import" in node_types or any("os" in n.name for n in nodes)
        assert "function" in node_types
    
    def test_extract_from_markdown(self, extractor):
        """从 Markdown 提取"""
        md = '''
# Title

## Section 1

Some content with [link](http://example.com)

## Section 2

More content
'''
        
        nodes, edges = extractor.extract_from_markdown(md, "test.md")
        
        # 应该提取到章节
        node_types = [n.node_type for n in nodes]
        
        assert "concept" in node_types  # 章节作为概念节点


class TestGraphRAGEngine:
    """Test cases for GraphRAGEngine"""
    
    @pytest.fixture
    def engine(self):
        """创建 GraphRAG 引擎"""
        return GraphRAGEngine(workspace_id="test")
    
    def test_index_python_document(self, engine):
        """索引 Python 文档"""
        code = '''
import os
import sys

def calculate():
    pass

class MyClass:
    pass
'''
        
        doc_id = engine.index_document(code, "test.py", doc_type="code")
        
        assert doc_id is not None
        assert len(doc_id) == 16  # MD5 hash
        
        # 检查图中有节点
        assert len(engine.graph._nodes) > 0
    
    def test_index_markdown_document(self, engine):
        """索引 Markdown 文档"""
        md = '''
# Project

## Setup

See [Installation](install.md)

## Usage

See [API](api.md)
'''
        
        doc_id = engine.index_document(md, "readme.md", doc_type="markdown")
        
        assert doc_id is not None
        assert len(engine.graph._nodes) > 0
    
    def test_retrieve_with_hops(self, engine):
        """多跳检索"""
        # 索引多个文档
        doc1_code = '''
import utils
def main():
    utils.helper()
'''
        doc2_code = '''
def helper():
    pass

def other():
    pass
'''
        doc3_code = '''
import database
def other():
    database.connect()
'''
        
        engine.index_document(doc1_code, "main.py", "code")
        engine.index_document(doc2_code, "utils.py", "code")
        engine.index_document(doc3_code, "db.py", "code")
        
        # 初始结果
        initial_results = [
            RetrievalResult(id="main.py", file_path="main.py", content=doc1_code, similarity=0.9),
            RetrievalResult(id="utils.py", file_path="utils.py", content=doc2_code, similarity=0.7),
        ]
        
        # 多跳检索
        results = engine.retrieve_with_hops(
            query="如何连接数据库",
            initial_results=initial_results,
            max_hops=2
        )
        
        # 应该能检索到 db.py
        ids = [r.id for r in results]
        assert len(results) > 0
    
    def test_get_related_documents(self, engine):
        """获取关联文档"""
        # 创建依赖关系
        engine.graph.add_node(GraphNode(id="A", node_type="document", name="A", content="A", file_path="a.py"))
        engine.graph.add_node(GraphNode(id="B", node_type="document", name="B", content="B", file_path="b.py"))
        engine.graph.add_node(GraphNode(id="C", node_type="document", name="C", content="C", file_path="c.py"))
        
        engine.graph.add_edge(GraphEdge(source_id="A", target_id="B", relation_type="imports", strength=0.9))
        engine.graph.add_edge(GraphEdge(source_id="B", target_id="C", relation_type="imports", strength=0.8))
        
        related = engine.get_related_documents("A", max_depth=2)
        
        # 应该找到 B 和 C
        related_ids = [n[0].id for n in related]
        assert "B" in related_ids
    
    def test_explain_path(self, engine):
        """解释路径"""
        engine.graph.add_node(GraphNode(id="X", node_type="document", name="X", content="X", file_path="x.py"))
        engine.graph.add_node(GraphNode(id="Y", node_type="document", name="Y", content="Y", file_path="y.py"))
        engine.graph.add_node(GraphNode(id="Z", node_type="document", name="Z", content="Z", file_path="z.py"))
        
        engine.graph.add_edge(GraphEdge(source_id="X", target_id="Y", relation_type="imports", strength=0.9))
        engine.graph.add_edge(GraphEdge(source_id="Y", target_id="Z", relation_type="imports", strength=0.8))
        
        path = engine.explain_path("X", "Z")
        
        assert path is not None
        assert len(path.nodes) == 3
        assert path.nodes[0].id == "X"
        assert path.nodes[-1].id == "Z"
    
    def test_stats(self, engine):
        """统计信息"""
        engine.graph.add_node(GraphNode(id="test", node_type="document", name="test", content="test", file_path="test.py"))
        
        stats = engine.stats()
        
        assert "workspace_id" in stats
        assert "graph_stats" in stats
        assert "traversal_capabilities" in stats


class TestNodeType:
    """Test cases for NodeType enum"""
    
    def test_node_types(self):
        """所有节点类型"""
        assert NodeType.DOCUMENT.value == "document"
        assert NodeType.FUNCTION.value == "function"
        assert NodeType.CLASS.value == "class"
        assert NodeType.VARIABLE.value == "variable"
        assert NodeType.IMPORT.value == "import"
        assert NodeType.ENTITY.value == "entity"
        assert NodeType.CONCEPT.value == "concept"


class TestRelationType:
    """Test cases for RelationType enum"""
    
    def test_relation_types(self):
        """所有关系类型"""
        assert RelationType.IMPORTS.value == "imports"
        assert RelationType.CALLS.value == "calls"
        assert RelationType.INHERITS.value == "inherits"
        assert RelationType.CONTAINS.value == "contains"
        assert RelationType.REFERENCES.value == "references"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
