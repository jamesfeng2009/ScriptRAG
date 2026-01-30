"""
灰度切流管理器
逐步将读流量从 PostgreSQL 切换到 Milvus
"""

import asyncio
import random
from typing import List, Dict, Any, Optional, Literal
import logging
from datetime import datetime
import asyncpg
from pymilvus import Collection


logger = logging.getLogger(__name__)


class GradualCutoverManager:
    """灰度切流管理器"""
    
    def __init__(
        self,
        pg_pool: asyncpg.Pool,
        milvus_collection: Collection
    ):
        """
        初始化灰度切流管理器
        
        Args:
            pg_pool: PostgreSQL 连接池
            milvus_collection: Milvus 集合
        """
        self.pg_pool = pg_pool
        self.milvus_collection = milvus_collection
        self.milvus_traffic_percentage = 0  # Milvus 流量百分比（0-100）
        self.cutover_strategy: Literal["random", "workspace", "user"] = "random"
        self.workspace_whitelist: List[str] = []  # 白名单工作空间
        
    def set_traffic_percentage(self, percentage: int):
        """
        设置 Milvus 流量百分比
        
        Args:
            percentage: 0-100 的整数
        """
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        
        self.milvus_traffic_percentage = percentage
        logger.info(f"Milvus traffic percentage set to {percentage}%")
    
    def set_cutover_strategy(
        self,
        strategy: Literal["random", "workspace", "user"]
    ):
        """
        设置切流策略
        
        Args:
            strategy: 切流策略
                - random: 随机切流
                - workspace: 按工作空间切流
                - user: 按用户切流
        """
        self.cutover_strategy = strategy
        logger.info(f"Cutover strategy set to {strategy}")
    
    def add_workspace_to_whitelist(self, workspace_id: str):
        """添加工作空间到白名单（强制使用 Milvus）"""
        if workspace_id not in self.workspace_whitelist:
            self.workspace_whitelist.append(workspace_id)
            logger.info(f"Added workspace {workspace_id} to whitelist")
    
    def remove_workspace_from_whitelist(self, workspace_id: str):
        """从白名单移除工作空间"""
        if workspace_id in self.workspace_whitelist:
            self.workspace_whitelist.remove(workspace_id)
            logger.info(f"Removed workspace {workspace_id} from whitelist")
    
    def should_use_milvus(
        self,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        判断是否应该使用 Milvus
        
        Args:
            workspace_id: 工作空间 ID
            user_id: 用户 ID
            
        Returns:
            是否使用 Milvus
        """
        # 白名单工作空间强制使用 Milvus
        if workspace_id and workspace_id in self.workspace_whitelist:
            return True
        
        # 100% 流量使用 Milvus
        if self.milvus_traffic_percentage >= 100:
            return True
        
        # 0% 流量使用 PostgreSQL
        if self.milvus_traffic_percentage <= 0:
            return False
        
        # 根据策略决定
        if self.cutover_strategy == "random":
            return random.randint(1, 100) <= self.milvus_traffic_percentage
        
        elif self.cutover_strategy == "workspace" and workspace_id:
            # 基于工作空间 ID 的哈希值决定
            hash_value = hash(workspace_id) % 100
            return hash_value < self.milvus_traffic_percentage
        
        elif self.cutover_strategy == "user" and user_id:
            # 基于用户 ID 的哈希值决定
            hash_value = hash(user_id) % 100
            return hash_value < self.milvus_traffic_percentage
        
        else:
            # 默认随机
            return random.randint(1, 100) <= self.milvus_traffic_percentage
    
    async def search_with_cutover(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        带灰度切流的搜索
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询向量
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            user_id: 用户 ID（可选）
            
        Returns:
            搜索结果列表
        """
        use_milvus = self.should_use_milvus(workspace_id, user_id)
        
        if use_milvus:
            logger.debug(f"Using Milvus for search (workspace: {workspace_id})")
            return await self._search_milvus(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        else:
            logger.debug(f"Using PostgreSQL for search (workspace: {workspace_id})")
            return await self._search_postgres(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
    
    async def _search_postgres(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """使用 PostgreSQL 搜索"""
        async with self.pg_pool.acquire() as conn:
            query = """
                SELECT 
                    id::text,
                    file_path,
                    content,
                    1 - (embedding <=> $2) AS similarity,
                    has_deprecated,
                    has_fixme,
                    has_todo,
                    has_security
                FROM code_documents
                WHERE workspace_id = $1
                    AND 1 - (embedding <=> $2) >= $3
                ORDER BY embedding <=> $2
                LIMIT $4
            """
            rows = await conn.fetch(
                query,
                workspace_id,
                query_embedding,
                similarity_threshold,
                top_k
            )
            
            return [dict(row) for row in rows]
    
    async def _search_milvus(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """使用 Milvus 搜索"""
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}
        }
        
        # 构建过滤表达式
        expr = f'workspace_id == "{workspace_id}"'
        
        # 执行搜索
        results = self.milvus_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "id", "file_path", "content",
                "has_deprecated", "has_fixme", "has_todo", "has_security"
            ]
        )
        
        # 转换结果格式
        documents = []
        for hits in results:
            for hit in hits:
                if hit.score >= similarity_threshold:
                    documents.append({
                        "id": hit.entity.get("id"),
                        "file_path": hit.entity.get("file_path"),
                        "content": hit.entity.get("content"),
                        "similarity": hit.score,
                        "has_deprecated": hit.entity.get("has_deprecated"),
                        "has_fixme": hit.entity.get("has_fixme"),
                        "has_todo": hit.entity.get("has_todo"),
                        "has_security": hit.entity.get("has_security")
                    })
        
        return documents
    
    async def compare_results(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        对比 PostgreSQL 和 Milvus 的搜索结果
        用于验证迁移质量
        
        Returns:
            对比结果统计
        """
        # 同时查询两个数据库
        pg_results = await self._search_postgres(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=0.0
        )
        
        milvus_results = await self._search_milvus(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=0.0
        )
        
        # 提取 ID 集合
        pg_ids = {r["id"] for r in pg_results}
        milvus_ids = {r["id"] for r in milvus_results}
        
        # 计算重叠度
        intersection = pg_ids & milvus_ids
        union = pg_ids | milvus_ids
        
        overlap_rate = len(intersection) / len(union) if union else 0
        
        # 计算相似度差异
        similarity_diffs = []
        for pg_r in pg_results:
            for mv_r in milvus_results:
                if pg_r["id"] == mv_r["id"]:
                    diff = abs(pg_r["similarity"] - mv_r["similarity"])
                    similarity_diffs.append(diff)
        
        avg_similarity_diff = sum(similarity_diffs) / len(similarity_diffs) if similarity_diffs else 0
        
        comparison = {
            "pg_count": len(pg_results),
            "milvus_count": len(milvus_results),
            "overlap_count": len(intersection),
            "overlap_rate": overlap_rate,
            "avg_similarity_diff": avg_similarity_diff,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Comparison result: overlap_rate={overlap_rate:.2%}, avg_diff={avg_similarity_diff:.4f}")
        return comparison


class CutoverScheduler:
    """切流调度器 - 自动化灰度切流流程"""
    
    def __init__(self, cutover_manager: GradualCutoverManager):
        self.cutover_manager = cutover_manager
        self.schedule = [
            {"percentage": 1, "duration_hours": 24},    # 1% 流量持续 24 小时
            {"percentage": 5, "duration_hours": 24},    # 5% 流量持续 24 小时
            {"percentage": 10, "duration_hours": 24},   # 10% 流量持续 24 小时
            {"percentage": 25, "duration_hours": 48},   # 25% 流量持续 48 小时
            {"percentage": 50, "duration_hours": 48},   # 50% 流量持续 48 小时
            {"percentage": 75, "duration_hours": 24},   # 75% 流量持续 24 小时
            {"percentage": 100, "duration_hours": 0},   # 100% 流量
        ]
    
    async def execute_gradual_cutover(self):
        """执行渐进式切流"""
        logger.info("Starting gradual cutover process")
        
        for stage in self.schedule:
            percentage = stage["percentage"]
            duration_hours = stage["duration_hours"]
            
            # 设置流量百分比
            self.cutover_manager.set_traffic_percentage(percentage)
            logger.info(f"Set traffic to {percentage}%, will hold for {duration_hours} hours")
            
            # 等待指定时间
            if duration_hours > 0:
                await asyncio.sleep(duration_hours * 3600)
        
        logger.info("Gradual cutover completed - 100% traffic on Milvus")
