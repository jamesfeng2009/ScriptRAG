"""Skill Persistence Service - Skills data access layer"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncpg

from ..domain.skills import SkillConfig, SKILLS

logger = logging.getLogger(__name__)


def _parse_json_field(value, default):
    """解析 JSON 字段"""
    import json
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value or default


class SkillRecord:
    """技能记录类（内存表示）"""

    def __init__(
        self,
        workspace_id: str,
        skill_name: str,
        description: str,
        tone: str,
        compatible_with: Optional[List[str]] = None,
        prompt_config: Optional[Dict[str, Any]] = None,
        is_enabled: bool = True,
        is_default: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.workspace_id = workspace_id
        self.skill_name = skill_name
        self.description = description
        self.tone = tone
        self.compatible_with = compatible_with or []
        self.prompt_config = prompt_config or {}
        self.is_enabled = is_enabled
        self.is_default = is_default
        self.extra_data = extra_data or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()


class SkillDatabaseService:
    """技能数据库服务（使用 asyncpg 直接操作）"""

    _instance = None

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "screenplay_system",
        user: str = "postgres",
        password: str = "postgres",
        echo: bool = False
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.echo = echo
        self._pool: Optional[asyncpg.Pool] = None
        logger.info(f"SkillDatabaseService initialized for {host}:{port}/{database}")

    @classmethod
    def create_from_env(cls) -> "SkillDatabaseService":
        """从环境变量创建服务"""
        import os
        return cls(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'screenplay_system'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
            echo=os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        )

    async def connect(self):
        """建立数据库连接"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10
            )
            logger.info("Database connection pool created")

    async def disconnect(self):
        """断开数据库连接"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")

    async def create_table(self):
        """创建工作空间技能表"""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS screenplay.workspace_skills (
                    id SERIAL PRIMARY KEY,
                    workspace_id VARCHAR(100) NOT NULL,
                    skill_name VARCHAR(100) NOT NULL,
                    description TEXT NOT NULL,
                    tone VARCHAR(50) NOT NULL,
                    compatible_with JSONB DEFAULT '[]'::jsonb,
                    prompt_config JSONB DEFAULT '{}'::jsonb,
                    is_enabled BOOLEAN DEFAULT TRUE,
                    is_default BOOLEAN DEFAULT FALSE,
                    extra_data JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(workspace_id, skill_name)
                );
                
                CREATE INDEX IF NOT EXISTS idx_workspace_skills_workspace_id
                ON screenplay.workspace_skills(workspace_id);
            """)
            logger.info("Workspace skills table ensured")

    async def create(self, record: SkillRecord) -> SkillRecord:
        """创建技能记录"""
        import json
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO screenplay.workspace_skills 
                (workspace_id, skill_name, description, tone, compatible_with, 
                 prompt_config, is_enabled, is_default, extra_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, record.workspace_id, record.skill_name, record.description, record.tone,
                json.dumps(record.compatible_with), json.dumps(record.prompt_config), 
                record.is_enabled, record.is_default, json.dumps(record.extra_data))
            logger.info(f"Created skill '{record.skill_name}' for workspace '{record.workspace_id}'")
            return record

    async def get(self, workspace_id: str, skill_name: str) -> Optional[SkillRecord]:
        """获取指定工作空间的技能"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT workspace_id, skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.workspace_skills
                WHERE workspace_id = $1 AND skill_name = $2
            """, workspace_id, skill_name)
            if row:
                return SkillRecord(
                    workspace_id=row['workspace_id'],
                    skill_name=row['skill_name'],
                    description=row['description'],
                    tone=row['tone'],
                    compatible_with=_parse_json_field(row['compatible_with'], []),
                    prompt_config=_parse_json_field(row['prompt_config'], {}),
                    is_enabled=row['is_enabled'],
                    is_default=row['is_default'],
                    extra_data=_parse_json_field(row['extra_data'], {}),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None

    async def get_by_workspace(self, workspace_id: str) -> List[SkillRecord]:
        """获取工作空间的所有技能"""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT workspace_id, skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.workspace_skills
                WHERE workspace_id = $1
                ORDER BY skill_name
            """, workspace_id)
            return [
                SkillRecord(
                    workspace_id=row['workspace_id'],
                    skill_name=row['skill_name'],
                    description=row['description'],
                    tone=row['tone'],
                    compatible_with=_parse_json_field(row['compatible_with'], []),
                    prompt_config=_parse_json_field(row['prompt_config'], {}),
                    is_enabled=row['is_enabled'],
                    is_default=row['is_default'],
                    extra_data=_parse_json_field(row['extra_data'], {}),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]

    async def get_enabled_by_workspace(self, workspace_id: str) -> List[SkillRecord]:
        """获取工作空间的所有启用技能"""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT workspace_id, skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.workspace_skills
                WHERE workspace_id = $1 AND is_enabled = TRUE
                ORDER BY skill_name
            """, workspace_id)
            return [
                SkillRecord(
                    workspace_id=row['workspace_id'],
                    skill_name=row['skill_name'],
                    description=row['description'],
                    tone=row['tone'],
                    compatible_with=_parse_json_field(row['compatible_with'], []),
                    prompt_config=_parse_json_field(row['prompt_config'], {}),
                    is_enabled=row['is_enabled'],
                    is_default=row['is_default'],
                    extra_data=_parse_json_field(row['extra_data'], {}),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]

    async def get_default_skill(self, workspace_id: str) -> Optional[SkillRecord]:
        """获取工作空间的默认技能"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT workspace_id, skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.workspace_skills
                WHERE workspace_id = $1 AND is_default = TRUE
                LIMIT 1
            """, workspace_id)
            if row:
                return SkillRecord(
                    workspace_id=row['workspace_id'],
                    skill_name=row['skill_name'],
                    description=row['description'],
                    tone=row['tone'],
                    compatible_with=_parse_json_field(row['compatible_with'], []),
                    prompt_config=_parse_json_field(row['prompt_config'], {}),
                    is_enabled=row['is_enabled'],
                    is_default=row['is_default'],
                    extra_data=_parse_json_field(row['extra_data'], {}),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None

    async def update(self, workspace_id: str, skill_name: str, **kwargs) -> Optional[SkillRecord]:
        """更新技能记录"""
        if not kwargs:
            return await self.get(workspace_id, skill_name)

        set_clauses = []
        values = []
        param_idx = 1

        field_mapping = {
            'description': 'description',
            'tone': 'tone',
            'compatible_with': 'compatible_with',
            'prompt_config': 'prompt_config',
            'is_enabled': 'is_enabled',
            'is_default': 'is_default',
            'extra_data': 'extra_data'
        }

        for field, column in field_mapping.items():
            if field in kwargs:
                set_clauses.append(f"{column} = ${param_idx}")
                values.append(kwargs[field])
                param_idx += 1

        if not set_clauses:
            return await self.get(workspace_id, skill_name)

        values.append(datetime.now())
        values.append(workspace_id)
        values.append(skill_name)

        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE screenplay.workspace_skills
                SET {', '.join(set_clauses)}, updated_at = ${param_idx}
                WHERE workspace_id = ${param_idx + 1} AND skill_name = ${param_idx + 2}
            """, *values)

        logger.info(f"Updated skill '{skill_name}' for workspace '{workspace_id}'")
        return await self.get(workspace_id, skill_name)

    async def delete(self, workspace_id: str, skill_name: str) -> bool:
        """删除技能记录"""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM screenplay.workspace_skills
                WHERE workspace_id = $1 AND skill_name = $2
            """, workspace_id, skill_name)
            deleted = result == "DELETE 1"
            if deleted:
                logger.info(f"Deleted skill '{skill_name}' from workspace '{workspace_id}'")
            return deleted

    async def delete_by_workspace(self, workspace_id: str) -> int:
        """删除工作空间的所有技能"""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM screenplay.workspace_skills
                WHERE workspace_id = $1
            """, workspace_id)
            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} skills from workspace '{workspace_id}'")
            return count

    async def count(self, workspace_id: str) -> int:
        """统计工作空间的技能数量"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COUNT(*) as cnt
                FROM screenplay.workspace_skills
                WHERE workspace_id = $1 AND is_enabled = TRUE
            """, workspace_id)
            return row['cnt'] if row else 0


class SkillService:
    """技能服务层（带缓存）"""

    def __init__(self, db_service: SkillDatabaseService, enable_cache: bool = True):
        self.db_service = db_service
        self.enable_cache = enable_cache
        self._cache: Dict[str, List[SkillRecord]] = {}
        logger.info(f"SkillService initialized with cache={enable_cache}")

    def _get_cache_key(self, workspace_id: str) -> str:
        return f"workspace_skills:{workspace_id}"

    def _invalidate_cache(self, workspace_id: str):
        cache_key = self._get_cache_key(workspace_id)
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Cache invalidated for workspace '{workspace_id}'")

    async def create(self, record: SkillRecord) -> SkillRecord:
        result = await self.db_service.create(record)
        self._invalidate_cache(record.workspace_id)
        return result

    async def get(self, workspace_id: str, skill_name: str) -> Optional[SkillRecord]:
        return await self.db_service.get(workspace_id, skill_name)

    async def get_by_workspace(self, workspace_id: str, use_cache: bool = True) -> List[SkillRecord]:
        cache_key = self._get_cache_key(workspace_id)
        if use_cache and self.enable_cache:
            if cache_key in self._cache:
                logger.debug(f"Cache hit for workspace '{workspace_id}'")
                return self._cache[cache_key]

        result = await self.db_service.get_by_workspace(workspace_id)

        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    async def get_enabled_skills(self, workspace_id: str) -> List[SkillRecord]:
        cache_key = f"{self._get_cache_key(workspace_id)}:enabled"
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = await self.db_service.get_enabled_by_workspace(workspace_id)

        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    async def get_default(self, workspace_id: str) -> Optional[SkillRecord]:
        cache_key = f"{self._get_cache_key(workspace_id)}:default"
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = await self.db_service.get_default_skill(workspace_id)

        if self.enable_cache and result:
            self._cache[cache_key] = result

        return result

    async def update(self, workspace_id: str, skill_name: str, **kwargs) -> Optional[SkillRecord]:
        result = await self.db_service.update(workspace_id, skill_name, **kwargs)
        if result:
            self._invalidate_cache(workspace_id)
        return result

    async def delete(self, workspace_id: str, skill_name: str) -> bool:
        result = await self.db_service.delete(workspace_id, skill_name)
        if result:
            self._invalidate_cache(workspace_id)
        return result

    async def ensure_default_skills(self, workspace_id: str) -> List[SkillRecord]:
        existing = await self.get_by_workspace(workspace_id, use_cache=False)

        if existing:
            logger.debug(f"Workspace '{workspace_id}' already has {len(existing)} skills")
            return existing

        records = []
        for skill_name, skill_config in SKILLS.items():
            record = SkillRecord(
                workspace_id=workspace_id,
                skill_name=skill_name,
                description=skill_config.description,
                tone=skill_config.tone,
                compatible_with=skill_config.compatible_with,
                is_enabled=True,
                is_default=(skill_name == "standard_tutorial")
            )
            await self.db_service.create(record)
            records.append(record)

        logger.info(f"Initialized {len(records)} default skills for workspace '{workspace_id}'")
        self._invalidate_cache(workspace_id)
        return records

    async def get_skill_config(self, workspace_id: str, skill_name: str) -> Optional[SkillConfig]:
        record = await self.get(workspace_id, skill_name)
        if record:
            return SkillConfig(
                description=record.description,
                tone=record.tone,
                compatible_with=record.compatible_with
            )

        if skill_name in SKILLS:
            return SKILLS[skill_name]

        return None

    async def get_available_skills(self, workspace_id: str) -> Dict[str, SkillConfig]:
        records = await self.get_enabled_skills(workspace_id)

        skills = {}
        for record in records:
            skills[record.skill_name] = SkillConfig(
                description=record.description,
                tone=record.tone,
                compatible_with=record.compatible_with
            )

        if not skills:
            for skill_name, skill_config in SKILLS.items():
                skills[skill_name] = skill_config

        return skills
