"""REST API Interface

This module implements the FastAPI REST API for the RAG screenplay generation system.
"""

import asyncio
import logging
import os
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
from dotenv import load_dotenv

from ..domain.models import SharedState
from ..application.orchestrator import WorkflowOrchestrator
from ..services.llm.service import LLMService
from ..services.retrieval_service import RetrievalService
from ..services.parser.tree_sitter_parser import TreeSitterParser
from ..services.summarization_service import SummarizationService
from ..services.database.postgres import PostgresService
from ..infrastructure.logging import setup_logging
from ..infrastructure.metrics import (
    get_metrics,
    record_workflow_execution,
    update_active_tasks,
    set_system_info
)


logger = logging.getLogger(__name__)


# ==================== Models ====================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """生成剧本请求"""
    topic: str = Field(..., description="User topic for screenplay generation")
    context: Optional[str] = Field("", description="Additional project context")
    skill: Optional[str] = Field("standard_tutorial", description="Initial skill mode")
    tone: Optional[str] = Field("professional", description="Global tone")
    max_retries: Optional[int] = Field(3, description="Maximum retries per step")
    config: Optional[Dict[str, Any]] = Field({}, description="Additional configuration options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Explain user authentication",
                "context": "Using JWT tokens",
                "skill": "standard_tutorial",
                "tone": "professional",
                "max_retries": 3,
                "config": {
                    "max_steps": 5,
                    "complexity": "simple",
                    "recursion_limit": 100
                }
            }
        }


class GenerateResponse(BaseModel):
    """生成剧本响应"""
    task_id: str = Field(..., description="Task ID for tracking")
    status: TaskStatus = Field(..., description="Task status")
    message: str = Field(..., description="Status message")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="Task ID")
    status: TaskStatus = Field(..., description="Task status")
    progress: Optional[Dict[str, Any]] = Field(None, description="Progress information")
    result: Optional[Dict[str, Any]] = Field(None, description="Result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Task creation time")
    updated_at: datetime = Field(..., description="Task last update time")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")
    services: Dict[str, str] = Field(..., description="Service statuses")


# ==================== Task Storage ====================

class TaskStore:
    """任务存储（内存存储，生产环境应使用 Redis）"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def create_task(self, task_id: str, request: GenerateRequest) -> Dict[str, Any]:
        """创建任务"""
        task = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "request": request.model_dump(),
            "progress": {},
            "result": None,
            "error": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务"""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            self.tasks[task_id]["updated_at"] = datetime.now()
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """列出所有任务"""
        return list(self.tasks.values())


# ==================== Global State ====================

task_store = TaskStore()
app_config = None
llm_service = None
retrieval_service = None
parser_service = None
summarization_service = None
persistence_service = None
orm_service = None


# ==================== FastAPI App ====================

app = FastAPI(
    title="RAG Screenplay Multi-Agent System API",
    description="REST API for generating screenplays based on code context using multi-agent system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Dependencies ====================

# 移除 API Key 认证，简化开发和测试
# 生产环境可以重新启用认证机制


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global app_config, llm_service, retrieval_service, parser_service, summarization_service, persistence_service, orm_service
    
    logger.info("Starting RAG Screenplay API...")
    
    # 加载环境变量
    load_dotenv()
    
    # 设置日志
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/api.log")
    setup_logging(level=log_level, log_file=log_file)
    
    # 设置系统信息
    set_system_info({
        'version': '1.0.0',
        'environment': os.getenv('ENVIRONMENT', 'production'),
        'api_host': os.getenv('API_HOST', '0.0.0.0'),
        'api_port': os.getenv('API_PORT', '8000')
    })
    
    # 加载配置
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            app_config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        app_config = {}
    
    # 初始化服务
    try:
        # LLM 服务
        llm_service = LLMService(app_config.get('llm', {}))
        logger.info("LLM service initialized")
        
        # 数据库服务
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'screenplay_system'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
        
        try:
            postgres_service = PostgresService(db_config)
            await postgres_service.connect()
            logger.info("Database service initialized")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, continuing without database")
            postgres_service = None
        
        # 向量数据库服务
        from ..services.database.vector_db import PostgresVectorDBService
        vector_db_service = PostgresVectorDBService(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'screenplay_system'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres')
        )
        
        # ORM 数据库服务
        from ..services.database.orm_service import DatabaseServiceFactory
        orm_service = DatabaseServiceFactory.create_from_env()
        
        # 持久化服务
        from ..services.persistence_service import PersistenceService
        persistence_service = PersistenceService(orm_service)
        
        # 检索服务
        from ..services.retrieval_service import RetrievalConfig
        retrieval_config = RetrievalConfig(**app_config.get('retrieval', {}))
        retrieval_service = RetrievalService(
            vector_db_service=vector_db_service,
            llm_service=llm_service,
            config=retrieval_config
        )
        logger.info("Retrieval service initialized")
        
        # 解析服务
        parser_service = TreeSitterParser()
        logger.info("Parser service initialized")
        
        # 摘要服务
        summarization_service = SummarizationService(
            llm_service=llm_service,
            config=app_config.get('retrieval', {}).get('summarization', {})
        )
        logger.info("Summarization service initialized")
        
        logger.info("All services initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("Shutting down RAG Screenplay API...")


# ==================== Background Task ====================

async def generate_screenplay_task(task_id: str, request: GenerateRequest):
    """
    后台任务：生成剧本
    
    Args:
        task_id: 任务 ID
        request: 生成请求
    """
    import time
    start_time = time.time()
    
    try:
        # 更新任务状态为运行中
        task_store.update_task(task_id, status=TaskStatus.RUNNING)
        update_active_tasks(len([t for t in task_store.list_tasks() if t['status'] == TaskStatus.RUNNING]))
        logger.info(f"Task {task_id} started")
        
        # 创建编排器
        orchestrator = WorkflowOrchestrator(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service
        )
        
        # 创建初始状态
        state = SharedState(
            user_topic=request.topic,
            project_context=request.context,
            current_skill=request.skill,
            global_tone=request.tone,
            max_retries=request.max_retries
        )
        
        # 执行工作流（增加递归限制以处理复杂工作流）
        recursion_limit = request.config.get('recursion_limit', 100) if request.config else 100
        result = await orchestrator.execute(state, recursion_limit=recursion_limit)
        
        # 计算执行时长
        duration = time.time() - start_time
        
        # 更新任务结果
        if result['success']:
            # 记录指标
            pivots_count = sum(1 for log in state.execution_log if log.get('action') == 'pivot_triggered')
            retries_count = sum(step.retry_count for step in state.outline)
            
            record_workflow_execution(
                duration=duration,
                status='success',
                steps_count=len(state.outline),
                pivots_count=pivots_count,
                retries_count=retries_count
            )
            
            task_store.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "screenplay": result.get('final_screenplay', ''),
                    "statistics": {
                        "total_steps": len(state.outline),
                        "fragments_generated": len(state.fragments),
                        "documents_retrieved": len(state.retrieved_docs),
                        "pivots_triggered": pivots_count,
                        "duration_seconds": duration
                    },
                    "execution_log": state.execution_log
                }
            )
            logger.info(f"Task {task_id} completed successfully")
        else:
            # 记录失败指标
            record_workflow_execution(
                duration=duration,
                status='failed',
                steps_count=len(state.outline)
            )
            
            task_store.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=result.get('error', 'Unknown error')
            )
            logger.error(f"Task {task_id} failed: {result.get('error')}")
    
    except Exception as e:
        duration = time.time() - start_time
        record_workflow_execution(
            duration=duration,
            status='failed',
            steps_count=0
        )
        
        task_store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e)
        )
        logger.error(f"Task {task_id} failed with exception: {e}")
    
    finally:
        # 更新活动任务数
        update_active_tasks(len([t for t in task_store.list_tasks() if t['status'] == TaskStatus.RUNNING]))


# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "RAG Screenplay Multi-Agent System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点
    
    Returns:
        服务健康状态
    """
    services = {
        "llm_service": "healthy" if llm_service else "unavailable",
        "retrieval_service": "healthy" if retrieval_service else "unavailable",
        "parser_service": "healthy" if parser_service else "unavailable",
        "summarization_service": "healthy" if summarization_service else "unavailable",
        "persistence_service": "healthy" if persistence_service else "unavailable",
        "database": "healthy" if orm_service and await orm_service.health_check() else "unavailable"
    }
    
    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now(),
        services=services
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_screenplay(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    生成剧本端点（异步）
    
    Args:
        request: 生成请求
        background_tasks: 后台任务管理器
        api_key: API Key（认证）
        
    Returns:
        任务 ID 和状态
    """
    # 生成任务 ID
    task_id = str(uuid.uuid4())
    
    # 创建任务
    task_store.create_task(task_id, request)
    
    # 添加后台任务
    background_tasks.add_task(generate_screenplay_task, task_id, request)
    
    logger.info(f"Created task {task_id} for topic: {request.topic}")
    
    return GenerateResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task created successfully. Use /status/{task_id} to check progress."
    )


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态端点
    
    Args:
        task_id: 任务 ID
        
    Returns:
        任务状态信息
    """
    task = task_store.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error"),
        created_at=task["created_at"],
        updated_at=task["updated_at"]
    )


@app.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks():
    """
    列出所有任务端点
    
    Returns:
        任务列表
    """
    tasks = task_store.list_tasks()
    
    return [
        TaskStatusResponse(
            task_id=task["task_id"],
            status=task["status"],
            progress=task.get("progress"),
            result=task.get("result"),
            error=task.get("error"),
            created_at=task["created_at"],
            updated_at=task["updated_at"]
        )
        for task in tasks
    ]


@app.get("/metrics")
async def metrics():
    """
    Prometheus 指标端点
    
    Returns:
        Prometheus 格式的指标
    """
    return get_metrics()


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        "presentation.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

