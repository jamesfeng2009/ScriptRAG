"""GraphRAG Engine - 图谱增强检索与多跳推理

功能：
1. 文档依赖图构建（基于 AST/引用关系）
2. 图遍历算法（Personalized PageRank, BFS, DFS）
3. 多跳检索引擎
4. 实体关系提取与推理

解决的问题：
- 传统向量检索是扁平的，无法理解代码/文档间的引用关系
- 复杂查询需要跨越多个文档才能找到完整答案
- 缺乏上下文关联的检索结果不完整

GraphRAG 核心思想：
1. 构建文档/实体间的引用图
2. 使用图遍历发现关联文档
3. 综合多跳信息生成答案
"""

import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import math

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型"""
    DOCUMENT = "document"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    ENTITY = "entity"  # 文档实体（人名、概念等）
    CONCEPT = "concept"


class RelationType(Enum):
    """关系类型"""
    IMPORTS = "imports"              # 导入关系
    CALLS = "calls"                  # 函数调用
    INHERITS = "inherits"            # 继承关系
    CONTAINS = "contains"            # 包含关系
    REFERENCES = "references"        # 引用关系
    DEFINES = "defines"              # 定义关系
    USES = "uses"                    # 使用关系
    RELATED_TO = "related_to"        # 相关关系
    DEPENDS_ON = "depends_on"        # 依赖关系
    IMPLEMENTS = "implements"        # 实现关系


@dataclass
class GraphNode:
    """图节点"""
    id: str
    node_type: str
    name: str
    content: str
    file_path: str
    embedding: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type,
            "name": self.name,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "file_path": self.file_path,
            "properties": self.properties,
            "metadata": self.metadata
        }


@dataclass
class GraphEdge:
    """图边"""
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type,
            "strength": self.strength,
            "properties": self.properties
        }


@dataclass
class GraphPath:
    """图路径"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "total_strength": self.total_strength,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges]
        }


@dataclass
class RetrievalResult:
    """检索结果"""
    id: str
    file_path: str
    content: str
    similarity: float
    hop_depth: int = 0
    path: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentDependencyGraph:
    """
    文档依赖图
    
    构建和管理文档/实体间的依赖关系图
    """
    
    def __init__(self, workspace_id: str = "default"):
        self.workspace_id = workspace_id
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._adjacency: Dict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))
        self._reverse_adjacency: Dict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)  # 实体名 -> 节点ID
    
    def add_node(self, node: GraphNode) -> None:
        """添加节点"""
        self._nodes[node.id] = node
        
        # 更新实体索引
        self._entity_index[node.name].add(node.id)
        if node.node_type:
            self._entity_index[node.node_type].add(node.id)
    
    def add_edge(self, edge: GraphEdge) -> None:
        """添加边"""
        self._edges.append(edge)
        
        # 更新邻接表
        self._adjacency[edge.source_id][edge.target_id].append(edge)
        self._reverse_adjacency[edge.target_id][edge.source_id].append(edge)
    
    def add_import_relation(
        self,
        from_doc_id: str,
        to_doc_id: str,
        module_name: str,
        strength: float = 1.0
    ) -> None:
        """添加导入关系"""
        edge = GraphEdge(
            source_id=from_doc_id,
            target_id=to_doc_id,
            relation_type=RelationType.IMPORTS.value,
            strength=strength,
            properties={"module_name": module_name}
        )
        self.add_edge(edge)
    
    def add_call_relation(
        self,
        caller_id: str,
        callee_id: str,
        strength: float = 0.8
    ) -> None:
        """添加函数调用关系"""
        edge = GraphEdge(
            source_id=caller_id,
            target_id=callee_id,
            relation_type=RelationType.CALLS.value,
            strength=strength
        )
        self.add_edge(edge)
    
    def add_reference_relation(
        self,
        from_id: str,
        to_id: str,
        reference_type: str = "mentions",
        strength: float = 0.6
    ) -> None:
        """添加引用关系"""
        edge = GraphEdge(
            source_id=from_id,
            target_id=to_id,
            relation_type=RelationType.REFERENCES.value,
            strength=strength,
            properties={"reference_type": reference_type}
        )
        self.add_edge(edge)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点"""
        return self._nodes.get(node_id)
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "out"
    ) -> Dict[str, List[GraphEdge]]:
        """获取邻居节点"""
        if direction == "out":
            return self._adjacency.get(node_id, {})
        else:
            return self._reverse_adjacency.get(node_id, {})
    
    def get_connected_components(self) -> List[Set[str]]:
        """获取连通分量"""
        visited = set()
        components = []
        
        for node_id in self._nodes:
            if node_id not in visited:
                component = set()
                queue = deque([node_id])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    # BFS 获取所有连接的节点
                    for neighbors in self.get_neighbors(current).values():
                        for edge in neighbors:
                            if edge.target_id not in visited:
                                queue.append(edge.target_id)
                
                components.append(component)
        
        return components
    
    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """按类型获取节点"""
        return [n for n in self._nodes.values() if n.node_type == node_type]
    
    def get_nodes_by_entity(self, entity_name: str) -> List[GraphNode]:
        """按实体名获取节点"""
        node_ids = self._entity_index.get(entity_name, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]
    
    def stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": defaultdict(int),
            "connected_components": len(self.get_connected_components())
        }
    
    def clear(self) -> None:
        """清空图"""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        self._entity_index.clear()


class GraphTraversalEngine:
    """
    图遍历引擎
    
    提供多种图遍历算法：
    1. BFS（广度优先搜索）
    2. DFS（深度优先搜索）
    3. Personalized PageRank
    4. 双向搜索
    """
    
    def __init__(self, graph: DocumentDependencyGraph):
        self.graph = graph
    
    def bfs_traverse(
        self,
        start_node_id: str,
        max_depth: int = 3,
        direction: str = "out",
        edge_types: Optional[List[str]] = None
    ) -> Dict[int, List[GraphNode]]:
        """
        BFS 遍历
        
        Returns:
            {depth: [nodes at that depth]}
        """
        edge_types = edge_types or []
        
        result = defaultdict(list)
        visited = {start_node_id}
        queue = deque([(start_node_id, 0)])
        
        while queue:
            node_id, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            node = self.graph.get_node(node_id)
            if node:
                result[depth].append(node)
            
            neighbors = self.graph.get_neighbors(node_id, direction)
            
            for target_id, edges in neighbors.items():
                if target_id in visited:
                    continue
                
                # 过滤边类型
                if edge_types:
                    if not any(e.relation_type in edge_types for e in edges):
                        continue
                
                visited.add(target_id)
                queue.append((target_id, depth + 1))
        
        return dict(result)
    
    def dfs_traverse(
        self,
        start_node_id: str,
        max_depth: int = 3,
        direction: str = "out",
        edge_types: Optional[List[str]] = None
    ) -> List[GraphPath]:
        """
        DFS 遍历
        
        Returns:
            所有找到的路径
        """
        edge_types = edge_types or []
        
        paths = []
        visited_stack = []
        
        def _dfs(
            current_id: str,
            depth: int,
            current_path: List[str],
            current_edges: List[GraphEdge]
        ):
            if depth > max_depth:
                return
            
            current_path.append(current_id)
            
            # 到达目标，返回路径
            if depth >= 1:
                nodes = [self.graph.get_node(nid) for nid in current_path if self.graph.get_node(nid)]
                edges = current_edges.copy()
                total_strength = math.prod([e.strength for e in edges]) if edges else 1.0
                
                paths.append(GraphPath(
                    nodes=[n for n in nodes if n],
                    edges=edges,
                    total_strength=total_strength
                ))
            
            neighbors = self.graph.get_neighbors(current_id, direction)
            
            for target_id, edges in neighbors.items():
                if target_id in current_path:
                    continue
                
                # 过滤边类型
                filtered_edges = [
                    e for e in edges
                    if not edge_types or e.relation_type in edge_types
                ]
                
                if not filtered_edges:
                    continue
                
                _dfs(target_id, depth + 1, current_path, current_edges + [filtered_edges[0]])
            
            current_path.pop()
        
        _dfs(start_node_id, 0, [], [])
        return paths
    
    def personalized_pagerank(
        self,
        source_nodes: List[str],
        max_iterations: int = 100,
        damping_factor: float = 0.85,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Personalized PageRank
        
        从多个源节点出发，计算每个节点的 PageRank 值
        
        Args:
            source_nodes: 源节点列表
            max_iterations: 最大迭代次数
            damping_factor: 阻尼因子（通常 0.85）
            tolerance: 收敛容忍度
            
        Returns:
            {node_id: pagerank_score}
        """
        if not source_nodes:
            return {}
        
        # 初始化
        node_ids = list(self.graph._nodes.keys())
        n = len(node_ids)
        
        if n == 0:
            return {}
        
        # 节点索引映射
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        idx_to_node = {i: nid for i, nid in enumerate(node_ids)}
        
        # 初始化分数
        scores = [0.0] * n
        for source in source_nodes:
            if source in node_to_idx:
                scores[node_to_idx[source]] = 1.0 / len(source_nodes)
        
        # 转移矩阵（邻接表）
        adjacency = self.graph._adjacency
        
        for iteration in range(max_iterations):
            new_scores = [0.0] * n
            
            # PageRank 迭代
            for i, node_id in enumerate(node_ids):
                rank_sum = 0.0
                
                # 获取指向当前节点的边
                incoming = self.graph._reverse_adjacency.get(node_id, {})
                
                for neighbor_id, edges in incoming.items():
                    neighbor_idx = node_to_idx.get(neighbor_id)
                    if neighbor_idx is None:
                        continue
                    
                    # 计算转移概率
                    total_strength = sum(e.strength for e in edges)
                    out_degree = sum(
                        sum(e.strength for e in out_edges)
                        for out_edges in adjacency.get(neighbor_id, {}).values()
                    )
                    
                    if out_degree > 0:
                        rank_sum += scores[neighbor_idx] * (total_strength / out_degree) * damping_factor
            
            # 添加随机跳转
            teleport = (1 - damping_factor) / n
            new_scores = [rank_sum + teleport for rank_sum in new_scores]
            
            # 检查收敛
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            
            if diff < tolerance:
                logger.debug(f"PageRank converged after {iteration + 1} iterations")
                break
        
        # 返回节点分数
        return {
            idx_to_node[i]: scores[i]
            for i in range(n)
            if scores[i] > 0
        }
    
    def find_shortest_paths(
        self,
        start_id: str,
        end_id: str,
        max_paths: int = 5
    ) -> List[GraphPath]:
        """
        查找最短路径（使用 BFS）
        
        Returns:
            最短路径列表
        """
        if start_id == end_id:
            node = self.graph.get_node(start_id)
            if node:
                return [GraphPath(nodes=[node], edges=[], total_strength=1.0)]
            return []
        
        queue = deque([(start_id, [start_id], [])])
        visited = {start_id}
        paths = []
        
        while queue and len(paths) < max_paths:
            current_id, path, edges = queue.popleft()
            
            neighbors = self.graph.get_neighbors(current_id)
            
            for target_id, conn_edges in neighbors.items():
                if target_id in visited and target_id != end_id:
                    continue
                
                new_path = path + [target_id]
                new_edges = edges + [conn_edges[0]]
                
                if target_id == end_id:
                    # 找到路径
                    nodes = [self.graph.get_node(nid) for nid in new_path if self.graph.get_node(nid)]
                    total_strength = math.prod([e.strength for e in new_edges]) if new_edges else 1.0
                    
                    paths.append(GraphPath(
                        nodes=[n for n in nodes if n],
                        edges=new_edges,
                        total_strength=total_strength
                    ))
                else:
                    visited.add(target_id)
                    queue.append((target_id, new_path, new_edges))
        
        # 按路径长度排序
        paths.sort(key=lambda p: len(p.nodes))
        return paths[:max_paths]


class EntityExtractor:
    """
    实体提取器
    
    从代码/文档中提取实体和关系
    """
    
    def __init__(self):
        # 代码模式
        self.import_pattern = re.compile(
            r'^(?:from|import)\s+([\w.]+)',
            re.MULTILINE
        )
        
        self.function_pattern = re.compile(
            r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            re.MULTILINE
        )
        
        self.class_pattern = re.compile(
            r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            re.MULTILINE
        )
        
        self.call_pattern = re.compile(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        )
        
        # 文档模式
        self.section_pattern = re.compile(
            r'^##\s+(.+)$',
            re.MULTILINE
        )
        
        self.link_pattern = re.compile(
            r'\[([^\]]+)\]\(([^)]+)\)'
        )
    
    def extract_from_code(
        self,
        content: str,
        file_path: str
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        从代码中提取实体和关系
        
        Returns:
            (nodes, edges)
        """
        nodes = []
        edges = []
        
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        
        # 提取导入
        imports = self.import_pattern.findall(content)
        for imp in imports:
            imp_id = hashlib.md5(imp.encode()).hexdigest()[:16]
            
            import_node = GraphNode(
                id=imp_id,
                node_type=NodeType.IMPORT.value,
                name=imp,
                content=f"Import: {imp}",
                file_path=file_path
            )
            nodes.append(import_node)
            
            # 导入关系
            edges.append(GraphEdge(
                source_id=doc_id,
                target_id=imp_id,
                relation_type=RelationType.IMPORTS.value,
                strength=0.9
            ))
        
        # 提取函数
        functions = self.function_pattern.findall(content)
        for func_name in functions:
            func_id = hashlib.md5(f"{doc_id}:{func_name}".encode()).hexdigest()[:16]
            
            func_node = GraphNode(
                id=func_id,
                node_type=NodeType.FUNCTION.value,
                name=func_name,
                content=f"Function: {func_name}",
                file_path=file_path,
                properties={"defined_in": doc_id}
            )
            nodes.append(func_node)
            
            # 调用关系
            calls = self.call_pattern.findall(content)
            for call_name in set(calls) - {func_name}:
                if call_name in functions:
                    call_id = hashlib.md5(f"{doc_id}:{call_name}".encode()).hexdigest()[:16]
                    edges.append(GraphEdge(
                        source_id=func_id,
                        target_id=call_id,
                        relation_type=RelationType.CALLS.value,
                        strength=0.8
                    ))
        
        # 提取类
        classes = self.class_pattern.findall(content)
        for class_name in classes:
            class_id = hashlib.md5(f"{doc_id}:{class_name}".encode()).hexdigest()[:16]
            
            class_node = GraphNode(
                id=class_id,
                node_type=NodeType.CLASS.value,
                name=class_name,
                content=f"Class: {class_name}",
                file_path=file_path
            )
            nodes.append(class_node)
        
        return nodes, edges
    
    def extract_from_markdown(
        self,
        content: str,
        file_path: str
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        从 Markdown 中提取实体和关系
        
        Returns:
            (nodes, edges)
        """
        nodes = []
        edges = []
        
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        
        # 提取章节
        sections = self.section_pattern.findall(content)
        for i, section in enumerate(sections):
            section_id = hashlib.md5(f"{doc_id}:section:{i}".encode()).hexdigest()[:16]
            
            section_node = GraphNode(
                id=section_id,
                node_type=NodeType.CONCEPT.value,
                name=section,
                content=f"Section: {section}",
                file_path=file_path
            )
            nodes.append(section_node)
            
            # 章节包含关系
            edges.append(GraphEdge(
                source_id=doc_id,
                target_id=section_id,
                relation_type=RelationType.CONTAINS.value,
                strength=0.7
            ))
        
        # 提取链接
        links = self.link_pattern.findall(content)
        for link_text, link_target in links:
            link_id = hashlib.md5(link_target.encode()).hexdigest()[:16]
            
            link_node = GraphNode(
                id=link_id,
                node_type=NodeType.ENTITY.value,
                name=link_text,
                content=f"Link: {link_text} -> {link_target}",
                file_path=file_path,
                properties={"target": link_target}
            )
            nodes.append(link_node)
            
            # 引用关系
            edges.append(GraphEdge(
                source_id=doc_id,
                target_id=link_id,
                relation_type=RelationType.REFERENCES.value,
                strength=0.6
            ))
        
        return nodes, edges


class GraphRAGEngine:
    """
    GraphRAG 检索引擎
    
    整合图检索和多跳推理
    """
    
    def __init__(self, workspace_id: str = "default"):
        self.workspace_id = workspace_id
        self.graph = DocumentDependencyGraph(workspace_id)
        self.traversal = GraphTraversalEngine(self.graph)
        self.extractor = EntityExtractor()
    
    def index_document(
        self,
        content: str,
        file_path: str,
        doc_type: str = "code"
    ) -> str:
        """
        索引文档
        
        Args:
            content: 文档内容
            file_path: 文件路径
            doc_type: 文档类型（code/markdown）
            
        Returns:
            文档 ID
        """
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        
        # 创建文档节点
        doc_node = GraphNode(
            id=doc_id,
            node_type=NodeType.DOCUMENT.value,
            name=file_path.split("/")[-1],
            content=content,
            file_path=file_path
        )
        self.graph.add_node(doc_node)
        
        # 提取实体和关系
        if doc_type == "code":
            nodes, edges = self.extractor.extract_from_code(content, file_path)
        else:
            nodes, edges = self.extractor.extract_from_markdown(content, file_path)
        
        # 添加节点和边
        for node in nodes:
            self.graph.add_node(node)
        
        for edge in edges:
            # 重新映射边的源和目标为文档节点
            if edge.source_id.startswith(doc_id) or edge.source_id in [n.id for n in nodes]:
                new_edge = GraphEdge(
                    source_id=doc_id,
                    target_id=edge.target_id,
                    relation_type=edge.relation_type,
                    strength=edge.strength,
                    properties=edge.properties
                )
                self.graph.add_edge(new_edge)
            else:
                self.graph.add_edge(edge)
        
        logger.info(f"Indexed document: {file_path} ({doc_id})")
        return doc_id
    
    def retrieve_with_hops(
        self,
        query: str,
        initial_results: List[RetrievalResult],
        max_hops: int = 2,
        max_results_per_hop: int = 5
    ) -> List[RetrievalResult]:
        """
        多跳检索
        
        Args:
            query: 查询文本
            initial_results: 初始检索结果
            max_hops: 最大跳数
            max_results_per_hop: 每跳最大结果数
            
        Returns:
            多跳检索结果
        """
        if not initial_results:
            return []
        
        # 第一跳：使用初始结果
        all_results = []
        seen_ids = set()
        
        for result in initial_results:
            if result.id not in seen_ids:
                result.hop_depth = 0
                all_results.append(result)
                seen_ids.add(result.id)
        
        # 多跳探索
        current_hop_docs = [r.id for r in initial_results]
        
        for hop in range(1, max_hops + 1):
            hop_results = []
            
            # 使用 PageRank 发现关联文档
            pagerank_scores = self.traversal.personalized_pagerank(
                source_nodes=current_hop_docs,
                max_iterations=50
            )
            
            # 排序并选择 top-k
            sorted_docs = sorted(
                pagerank_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_results_per_hop]
            
            for doc_id, score in sorted_docs:
                if doc_id in seen_ids:
                    continue
                
                node = self.graph.get_node(doc_id)
                if not node:
                    continue
                
                # 获取到该文档的路径
                paths = []
                for seed in current_hop_docs:
                    found_paths = self.traversal.find_shortest_paths(seed, doc_id, max_paths=3)
                    paths.extend(found_paths)
                
                hop_results.append(RetrievalResult(
                    id=doc_id,
                    file_path=node.file_path,
                    content=node.content,
                    similarity=score,
                    hop_depth=hop,
                    path=[p.nodes[-1].id for p in paths[:1]] if paths else None,
                    metadata={
                        "node_type": node.node_type,
                        "paths": [p.to_dict() for p in paths[:1]]
                    }
                ))
                
                seen_ids.add(doc_id)
            
            all_results.extend(hop_results)
            current_hop_docs = [r.id for r in hop_results]
            
            if not current_hop_docs:
                break
        
        # 按相似度排序返回
        all_results.sort(key=lambda x: (x.similarity, -x.hop_depth), reverse=True)
        
        return all_results
    
    def get_related_documents(
        self,
        doc_id: str,
        max_depth: int = 2,
        max_results: int = 10
    ) -> List[Tuple[GraphNode, float, List[str]]]:
        """
        获取与指定文档相关的文档
        
        Returns:
            [(节点, 关联强度, 关系类型列表), ...]
        """
        # 使用 PageRank 计算关联强度
        pagerank_scores = self.traversal.personalized_pagerank(
            source_nodes=[doc_id],
            max_iterations=50
        )
        
        # 获取直接邻居
        neighbors = self.graph.get_neighbors(doc_id)
        
        related = []
        
        # 直接邻居（强关联）
        for target_id, edges in neighbors.items():
            node = self.graph.get_node(target_id)
            if not node:
                continue
            
            total_strength = sum(e.strength for e in edges)
            relation_types = list(set(e.relation_type for e in edges))
            
            related.append((node, total_strength, relation_types))
        
        # PageRank 高分节点（间接关联）
        sorted_scores = sorted(
            pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_results]
        
        for node_id, score in sorted_scores:
            if node_id == doc_id:
                continue
            
            node = self.graph.get_node(node_id)
            if not node:
                continue
            
            # 避免重复
            if node.id in [n[0].id for n in related]:
                continue
            
            related.append((node, score, ["indirect"]))
        
        # 按关联强度排序
        related.sort(key=lambda x: x[1], reverse=True)
        
        return related[:max_results]
    
    def explain_path(
        self,
        start_id: str,
        end_id: str
    ) -> Optional[GraphPath]:
        """
        解释两个节点之间的路径
        
        Returns:
            路径信息
        """
        paths = self.traversal.find_shortest_paths(start_id, end_id, max_paths=1)
        
        if paths:
            return paths[0]
        
        return None
    
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "workspace_id": self.workspace_id,
            "graph_stats": self.graph.stats(),
            "traversal_capabilities": [
                "bfs_traverse",
                "dfs_traverse",
                "personalized_pagerank",
                "find_shortest_paths"
            ]
        }
