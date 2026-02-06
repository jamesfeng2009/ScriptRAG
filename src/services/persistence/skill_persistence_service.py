"""技能持久化服务 - 技能数据访问层"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncpg

from src.domain.skills import SkillConfig, SKILLS

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
        port: int = 5433,
        database: str = "Screenplay",
        user: str = "postgres",
        password: str = "123456",
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
        from ...config import get_database_config
        db_config = get_database_config()
        return cls(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password,
            echo=db_config.echo
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
        """创建技能表"""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS screenplay.skills (
                    id SERIAL PRIMARY KEY,
                    skill_name VARCHAR(100) NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    tone VARCHAR(50) NOT NULL,
                    compatible_with JSONB DEFAULT '[]'::jsonb,
                    prompt_config JSONB DEFAULT '{}'::jsonb,
                    is_enabled BOOLEAN DEFAULT TRUE,
                    is_default BOOLEAN DEFAULT FALSE,
                    extra_data JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_skills_skill_name
                ON screenplay.skills(skill_name);
            """)
            logger.info("Skills table ensured")

    async def create(self, record: SkillRecord) -> SkillRecord:
        """创建技能记录（幂等操作）- 如果已存在则忽略"""
        import json
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO screenplay.skills 
                (skill_name, description, tone, compatible_with, 
                 prompt_config, is_enabled, is_default, extra_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (skill_name) DO NOTHING
            """, record.skill_name, record.description, record.tone,
                json.dumps(record.compatible_with), json.dumps(record.prompt_config), 
                record.is_enabled, record.is_default, json.dumps(record.extra_data))
            logger.info(f"Created skill '{record.skill_name}' (idempotent)")
            return record

    async def get(self, skill_name: str) -> Optional[SkillRecord]:
        """获取技能"""
        if self._pool is None:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.skills
                WHERE skill_name = $1
            """, skill_name)
            if row:
                return SkillRecord(
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

    async def get_all(self) -> List[SkillRecord]:
        """获取所有技能"""
        if self._pool is None:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.skills
                ORDER BY skill_name
            """)
            return [
                SkillRecord(
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

    async def get_enabled(self) -> List[SkillRecord]:
        """获取所有启用的技能"""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.skills
                WHERE is_enabled = TRUE
                ORDER BY skill_name
            """)
            return [
                SkillRecord(
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

    async def get_default(self) -> Optional[SkillRecord]:
        """获取默认技能"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT skill_name, description, tone, compatible_with,
                       prompt_config, is_enabled, is_default, extra_data, created_at, updated_at
                FROM screenplay.skills
                WHERE is_default = TRUE
                LIMIT 1
            """)
            if row:
                return SkillRecord(
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

    async def update(self, skill_name: str, **kwargs) -> Optional[SkillRecord]:
        """更新技能记录"""
        if not kwargs:
            return await self.get(skill_name)

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
                value = kwargs[field]
                # Convert dict fields to JSON strings
                if field in ('prompt_config', 'compatible_with', 'extra_data') and isinstance(value, dict):
                    import json
                    value = json.dumps(value)
                elif field == 'compatible_with' and isinstance(value, list):
                    import json
                    value = json.dumps(value)
                values.append(value)
                param_idx += 1

        if not set_clauses:
            return await self.get(skill_name)

        values.append(datetime.now())
        values.append(skill_name)

        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE screenplay.skills
                SET {', '.join(set_clauses)}, updated_at = ${param_idx}
                WHERE skill_name = ${param_idx + 1}
            """, *values)

        logger.info(f"Updated skill '{skill_name}'")
        return await self.get(skill_name)

    async def delete(self, skill_name: str) -> bool:
        """删除技能记录"""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM screenplay.skills
                WHERE skill_name = $1
            """, skill_name)
            deleted = result == "DELETE 1"
            if deleted:
                logger.info(f"Deleted skill '{skill_name}'")
            return deleted

    async def delete_all(self) -> int:
        """删除所有技能"""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM screenplay.skills
            """)
            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} skills")
            return count

    async def count(self) -> int:
        """统计启用的技能数量"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COUNT(*) as cnt
                FROM screenplay.skills
                WHERE is_enabled = TRUE
            """)
            return row['cnt'] if row else 0


class SkillService:
    """技能服务层（带缓存）"""

    def __init__(self, db_service: SkillDatabaseService, enable_cache: bool = True):
        self.db_service = db_service
        self.enable_cache = enable_cache
        self._cache: Dict[str, List[SkillRecord]] = {}
        logger.info(f"SkillService initialized with cache={enable_cache}")

    def _get_cache_key(self, key_type: str = "all") -> str:
        return f"skills:{key_type}"

    def _invalidate_all_cache(self):
        self._cache = {}
        logger.debug("All cache invalidated")

    async def create(self, record: SkillRecord) -> SkillRecord:
        result = await self.db_service.create(record)
        self._invalidate_all_cache()
        return result

    async def get(self, skill_name: str) -> Optional[SkillRecord]:
        return await self.db_service.get(skill_name)

    async def get_all(self, use_cache: bool = True) -> List[SkillRecord]:
        cache_key = self._get_cache_key("all")
        if use_cache and self.enable_cache and cache_key in self._cache:
            logger.debug("Cache hit for all skills")
            return self._cache[cache_key]

        result = await self.db_service.get_all()

        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    async def get_enabled_skills(self) -> List[SkillRecord]:
        cache_key = self._get_cache_key("enabled")
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = await self.db_service.get_enabled()

        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    async def get_default(self) -> Optional[SkillRecord]:
        cache_key = self._get_cache_key("default")
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = await self.db_service.get_default()

        if self.enable_cache and result:
            self._cache[cache_key] = result

        return result

    async def update(self, skill_name: str, **kwargs) -> Optional[SkillRecord]:
        result = await self.db_service.update(skill_name, **kwargs)
        if result:
            self._invalidate_all_cache()
        return result

    async def delete(self, skill_name: str) -> bool:
        result = await self.db_service.delete(skill_name)
        if result:
            self._invalidate_all_cache()
        return result

    async def ensure_default_skills(self) -> List[SkillRecord]:
        existing = await self.get_all(use_cache=False)

        if existing:
            logger.debug(f"Already has {len(existing)} skills")
            return existing

        records = []
        for skill_name, skill_config in SKILLS.items():
            record = SkillRecord(
                skill_name=skill_name,
                description=skill_config.description,
                tone=skill_config.tone,
                compatible_with=skill_config.compatible_with,
                is_enabled=True,
                is_default=(skill_name == "standard_tutorial")
            )
            try:
                await self.db_service.create(record)
                records.append(record)
            except Exception as e:
                if "duplicate key" in str(e).lower():
                    logger.debug(f"Skill '{skill_name}' already exists")
                else:
                    raise

        logger.info(f"Initialized {len(records)} default skills")
        self._invalidate_all_cache()
        return records

    async def get_skill_config(self, skill_name: str) -> Optional[SkillConfig]:
        record = await self.get(skill_name)
        if record:
            return SkillConfig(
                description=record.description,
                tone=record.tone,
                compatible_with=record.compatible_with,
                prompt_config=record.prompt_config
            )

        if skill_name in SKILLS:
            return SKILLS[skill_name]

        return None

    async def get_available_skills(self) -> Dict[str, SkillConfig]:
        records = await self.get_enabled_skills()

        skills = {}
        for record in records:
            skills[record.skill_name] = SkillConfig(
                description=record.description,
                tone=record.tone,
                compatible_with=record.compatible_with,
                prompt_config=record.prompt_config
            )

        if not skills:
            for skill_name, skill_config in SKILLS.items():
                skills[skill_name] = skill_config

        return skills

    async def __aenter__(self) -> "SkillService":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_service.disconnect()
