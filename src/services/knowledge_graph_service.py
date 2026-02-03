"""Knowledge Graph Service - 知识图谱管理服务"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy import text, select, insert, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..config import get_database_config

logger = logging.getLogger(__name__)


class KnowledgeNodeModel:
    """知识节点数据类"""
    def __init__(
        self,
        id: str,
        workspace_id: str,
        node_type: str,
        name: str,
        content: Optional[str] = None,
        embedding: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
        line_number: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.id = id
        self.workspace_id = workspace_id
        self.node_type = node_type
        self.name = name
        self.content = content
        self.embedding = embedding
        self.properties = properties or {}
        self.source_file = source_file
        self.line_number = line_number
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()


class KnowledgeRelationModel:
    """知识关系数据类"""
    def __init__(
        self,
        id: str,
        workspace_id: str,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        strength: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.workspace_id = workspace_id
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.relation_type = relation_type
        self.strength = strength
        self.properties = properties or {}
        self.created_at = created_at or datetime.now()


class KnowledgeGraphService:
    """知识图谱服务"""
    
    def __init__(self, config=None):
        """
        初始化知识图谱服务
        
        Args:
            config: 数据库配置对象（可选，默认从 config.py 加载）
        """
        self._config = config
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def _get_db_url(self) -> str:
        """获取数据库连接 URL"""
        if self._config is not None:
            db_config = self._config
        else:
            from ..config import get_database_config
            db_config = get_database_config()
        return f"postgresql+asyncpg://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
    
    async def initialize(self):
        """初始化连接池"""
        try:
            self._engine = create_async_engine(
                self._get_db_url(),
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            self._session_factory = sessionmaker(
                self._engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._initialized = True
            logger.info("KnowledgeGraphService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphService: {e}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("KnowledgeGraphService closed")
    
    async def health_check(self) -> bool:
        """健康检查"""
        if not self._initialized:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def create_node(
        self,
        workspace_id: str,
        node_type: str,
        name: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        properties: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
        line_number: Optional[int] = None
    ) -> str:
        """创建知识节点"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        node_id = str(uuid4())
        embedding_str = f"[{','.join(map(str, embedding))}]" if embedding else None
        properties_json = None
        if properties:
            import json
            properties_json = json.dumps(properties)
        
        async with self._session_factory() as session:
            query = text("""
                INSERT INTO screenplay.knowledge_nodes (
                    id, workspace_id, node_type, name, content, embedding,
                    properties, source_file, line_number
                ) VALUES (
                    :id, :workspace_id, :node_type, :name, :content, :embedding,
                    :properties, :source_file, :line_number
                )
                RETURNING id
            """)
            
            result = await session.execute(query, {
                'id': node_id,
                'workspace_id': workspace_id,
                'node_type': node_type,
                'name': name,
                'content': content,
                'embedding': embedding_str,
                'properties': properties_json,
                'source_file': source_file,
                'line_number': line_number
            })
            await session.commit()
            
            return str(result.scalar())
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取知识节点"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT id, workspace_id, node_type, name, content, embedding,
                       properties, source_file, line_number, created_at, updated_at
                FROM screenplay.knowledge_nodes
                WHERE id = :id
            """)
            
            result = await session.execute(query, {'id': node_id})
            row = result.fetchone()
            
            if row:
                return {
                    'id': str(row[0]),
                    'workspace_id': row[1],
                    'node_type': row[2],
                    'name': row[3],
                    'content': row[4],
                    'embedding': row[5],
                    'properties': row[6],
                    'source_file': row[7],
                    'line_number': row[8],
                    'created_at': row[9].isoformat() if row[9] else None,
                    'updated_at': row[10].isoformat() if row[10] else None
                }
            return None
    
    async def list_nodes(
        self,
        workspace_id: str,
        node_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出知识节点"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            if node_type:
                query = text("""
                    SELECT id, workspace_id, node_type, name, content, embedding,
                           properties, source_file, line_number, created_at, updated_at
                    FROM screenplay.knowledge_nodes
                    WHERE workspace_id = :workspace_id AND node_type = :node_type
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                params = {'workspace_id': workspace_id, 'node_type': node_type, 'limit': limit, 'offset': offset}
            else:
                query = text("""
                    SELECT id, workspace_id, node_type, name, content, embedding,
                           properties, source_file, line_number, created_at, updated_at
                    FROM screenplay.knowledge_nodes
                    WHERE workspace_id = :workspace_id
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                params = {'workspace_id': workspace_id, 'limit': limit, 'offset': offset}
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            nodes = []
            for row in rows:
                nodes.append({
                    'id': str(row[0]),
                    'workspace_id': row[1],
                    'node_type': row[2],
                    'name': row[3],
                    'content': row[4],
                    'embedding': row[5],
                    'properties': row[6],
                    'source_file': row[7],
                    'line_number': row[8],
                    'created_at': row[9].isoformat() if row[9] else None,
                    'updated_at': row[10].isoformat() if row[10] else None
                })
            
            return nodes
    
    async def delete_node(self, node_id: str) -> bool:
        """删除知识节点（级联删除关联关系）"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                DELETE FROM screenplay.knowledge_nodes
                WHERE id = :id
                RETURNING id
            """)
            
            result = await session.execute(query, {'id': node_id})
            await session.commit()
            
            return result.scalar() is not None
    
    async def create_relation(
        self,
        workspace_id: str,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        strength: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建知识关系"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        relation_id = str(uuid4())
        properties_json = None
        if properties:
            import json
            properties_json = json.dumps(properties)
        
        async with self._session_factory() as session:
            query = text("""
                INSERT INTO screenplay.knowledge_relations (
                    id, workspace_id, source_node_id, target_node_id,
                    relation_type, strength, properties
                ) VALUES (
                    :id, :workspace_id, :source_node_id, :target_node_id,
                    :relation_type, :strength, :properties
                )
                ON CONFLICT (source_node_id, target_node_id, relation_type)
                DO UPDATE SET strength = EXCLUDED.strength, properties = EXCLUDED.properties
                RETURNING id
            """)
            
            result = await session.execute(query, {
                'id': relation_id,
                'workspace_id': workspace_id,
                'source_node_id': source_node_id,
                'target_node_id': target_node_id,
                'relation_type': relation_type,
                'strength': strength,
                'properties': properties_json
            })
            await session.commit()
            
            return str(result.scalar())
    
    async def get_relation(self, relation_id: str) -> Optional[Dict[str, Any]]:
        """获取知识关系"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT id, workspace_id, source_node_id, target_node_id,
                       relation_type, strength, properties, created_at
                FROM screenplay.knowledge_relations
                WHERE id = :id
            """)
            
            result = await session.execute(query, {'id': relation_id})
            row = result.fetchone()
            
            if row:
                return {
                    'id': str(row[0]),
                    'workspace_id': row[1],
                    'source_node_id': str(row[2]),
                    'target_node_id': str(row[3]),
                    'relation_type': row[4],
                    'strength': row[5],
                    'properties': row[6],
                    'created_at': row[7].isoformat() if row[7] else None
                }
            return None
    
    async def list_relations(
        self,
        workspace_id: str,
        source_node_id: Optional[str] = None,
        target_node_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出知识关系"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        conditions = ["workspace_id = :workspace_id"]
        params = {'workspace_id': workspace_id, 'limit': limit, 'offset': offset}
        
        if source_node_id:
            conditions.append("source_node_id = :source_node_id")
            params['source_node_id'] = source_node_id
        if target_node_id:
            conditions.append("target_node_id = :target_node_id")
            params['target_node_id'] = target_node_id
        if relation_type:
            conditions.append("relation_type = :relation_type")
            params['relation_type'] = relation_type
        
        where_clause = " AND ".join(conditions)
        
        async with self._session_factory() as session:
            query = text(f"""
                SELECT id, workspace_id, source_node_id, target_node_id,
                       relation_type, strength, properties, created_at
                FROM screenplay.knowledge_relations
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            relations = []
            for row in rows:
                relations.append({
                    'id': str(row[0]),
                    'workspace_id': row[1],
                    'source_node_id': str(row[2]),
                    'target_node_id': str(row[3]),
                    'relation_type': row[4],
                    'strength': row[5],
                    'properties': row[6],
                    'created_at': row[7].isoformat() if row[7] else None
                })
            
            return relations
    
    async def delete_relation(self, relation_id: str) -> bool:
        """删除知识关系"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                DELETE FROM screenplay.knowledge_relations
                WHERE id = :id
                RETURNING id
            """)
            
            result = await session.execute(query, {'id': relation_id})
            await session.commit()
            
            return result.scalar() is not None
    
    async def get_graph_stats(self, workspace_id: str) -> Dict[str, Any]:
        """获取知识图谱统计"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            nodes_query = text("""
                SELECT node_type, COUNT(*) as count
                FROM screenplay.knowledge_nodes
                WHERE workspace_id = :workspace_id
                GROUP BY node_type
            """)
            nodes_result = await session.execute(nodes_query, {'workspace_id': workspace_id})
            nodes_by_type = {row[0]: row[1] for row in nodes_result.fetchall()}
            
            relations_query = text("""
                SELECT relation_type, COUNT(*) as count
                FROM screenplay.knowledge_relations
                WHERE workspace_id = :workspace_id
                GROUP BY relation_type
            """)
            relations_result = await session.execute(relations_query, {'workspace_id': workspace_id})
            relations_by_type = {row[0]: row[1] for row in relations_result.fetchall()}
            
            total_nodes = sum(nodes_by_type.values())
            total_relations = sum(relations_by_type.values())
            
            return {
                'workspace_id': workspace_id,
                'nodes_by_type': nodes_by_type,
                'relations_by_type': relations_by_type,
                'total_nodes': total_nodes,
                'total_relations': total_relations
            }
