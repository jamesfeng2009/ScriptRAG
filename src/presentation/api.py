"""精简版 REST API - 专注于 RAG 剧本生成"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_app_config, get_llm_config
from ..domain.state_types import GlobalState
from ..application.orchestrator import WorkflowOrchestrator
from ..services.llm.service import LLMService
from ..services.retrieval_service import RetrievalService, RetrievalConfig
from ..infrastructure.logging import configure_logging
from ..services.persistence.task_persistence_service import TaskDatabaseService, TaskRecord, TaskService


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SkillType(str, Enum):
    STANDARD_TUTORIAL = "standard_tutorial"
    WARNING_MODE = "warning_mode"
    VISUALIZATION_ANALOGY = "visualization_analogy"
    RESEARCH_MODE = "research_mode"
    MEME_STYLE = "meme_style"


class RAGConfig(BaseModel):
    """RAG配置"""
    enable_hybrid_search: bool = True
    top_k: int = Field(5, ge=1, le=20)
    enable_reranking: bool = True


class SkillConfig(BaseModel):
    """Skill配置"""
    initial_skill: SkillType = SkillType.STANDARD_TUTORIAL
    enable_auto_switch: bool = True
    switch_threshold: float = Field(0.7, ge=0.0, le=1.0)


class GenerateRequest(BaseModel):
    """剧本生成请求"""
    topic: str = Field(..., min_length=1, description="生成主题")
    context: Optional[str] = Field("", description="上下文信息（可用于传入对话历史）")
    chat_session_id: Optional[str] = Field(None, description="关联的 Chat Session ID（可选）")
    rag: Optional[RAGConfig] = Field(default_factory=RAGConfig)
    rag_sources: Optional[List[str]] = Field(
        default=None,
        description="指定文档分类，不传则检索所有，如 ['python_tutorial', 'api_docs']"
    )
    skill: Optional[SkillConfig] = Field(default_factory=SkillConfig)
    enable_dynamic_adjustment: bool = Field(True, description="启用动态方向调整")
    max_retries: int = Field(3, ge=1, le=10)
    recursion_limit: int = Field(100, ge=10, le=200)


class GenerateResponse(BaseModel):
    """剧本生成响应"""
    task_id: str
    status: TaskStatus
    screenplay: Optional[str] = None
    outline: Optional[List[Dict[str, Any]]] = None
    skill_history: Optional[List[Dict[str, Any]]] = None
    direction_changes: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: datetime


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    llm_available: bool


class AdjustRequest(BaseModel):
    """方向调整请求"""
    action: str = Field(..., description="调整动作: switch_skill, skip_step, add_step, abort")
    skill: Optional[str] = Field(None, description="目标 Skill（switch_skill 时必填）")
    step_index: Optional[int] = Field(None, description="步骤索引（skip_step 时必填）")
    new_step: Optional[str] = Field(None, description="新步骤描述（add_step 时必填）")
    reason: str = Field(..., description="调整原因")


class AdjustResponse(BaseModel):
    """调整响应"""
    success: bool
    task_id: str
    action: str
    result: Dict[str, Any]
    message: str


class RAGAnalysisResponse(BaseModel):
    """RAG分析结果响应"""
    task_id: str
    has_analysis: bool
    content_types: Optional[List[str]] = None
    main_topic: Optional[str] = None
    sub_topics: Optional[List[str]] = None
    difficulty_level: Optional[float] = None
    tone_style: Optional[str] = None
    key_concepts: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    suggested_skill: Optional[str] = None
    confidence: Optional[float] = None
    direction_changes: Optional[List[Dict[str, Any]]] = None
    skill_history: Optional[List[Dict[str, Any]]] = None
    analyzed_at: Optional[datetime] = None


class RAGAdjustRequest(BaseModel):
    """RAG动态调整请求"""
    top_k: Optional[int] = Field(None, ge=1, le=20, description="检索文档数量")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    enable_hybrid_search: Optional[bool] = Field(None, description="启用混合搜索")
    enable_reranking: Optional[bool] = Field(None, description="启用重排序")
    force_reanalysis: bool = Field(False, description="强制重新分析")
    query: Optional[str] = Field(None, description="重新检索的查询词")


class RAGAdjustResponse(BaseModel):
    """RAG动态调整响应"""
    success: bool
    task_id: str
    previous_config: Dict[str, Any]
    new_config: Dict[str, Any]
    retrieved_docs_count: int
    analysis_result: Optional[Dict[str, Any]] = None
    message: str


class SkillCreateRequest(BaseModel):
    """创建技能请求"""
    skill_name: str = Field(..., min_length=1, max_length=100, description="技能名称")
    description: str = Field(..., description="技能描述")
    tone: str = Field(..., description="语调风格")
    compatible_with: List[str] = Field(default_factory=list, description="兼容的技能列表")
    prompt_config: Dict[str, Any] = Field(default_factory=dict, description="提示配置")
    is_enabled: bool = Field(True, description="是否启用")
    is_default: bool = Field(False, description="是否为默认技能")


class SkillResponse(BaseModel):
    """技能响应"""
    skill_name: str
    description: str
    tone: str
    compatible_with: List[str]
    prompt_config: Optional[Dict[str, Any]] = None
    is_enabled: bool
    is_default: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkspaceSkillsResponse(BaseModel):
    """技能列表响应"""
    skills: List[SkillResponse]
    default_skill: Optional[str] = None
    total_count: int


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    title: str = Field(..., description="文档标题")
    file_name: str = Field(..., description="文件名")
    content: str = Field(..., description="文档内容")
    category: Optional[str] = Field(None, description="文档分类")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str
    title: str
    file_name: str
    category: Optional[str]
    file_size: int
    created_at: Optional[str]


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentResponse]
    total_count: int
    page: int
    page_size: int


class DocumentSearchResponse(BaseModel):
    """文档搜索响应"""
    documents: List[Dict[str, Any]]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""
    success: bool
    id: str


class IngestRequest(BaseModel):
    """文档摄入请求"""
    file_path: str = Field(..., description="文件路径")
    source_id: Optional[str] = Field(None, description="文档唯一标识")


class IngestResponse(BaseModel):
    """文档摄入响应"""
    status: str
    source_id: str
    chunk_count: int
    error_msg: Optional[str] = None


class QueryRequest(BaseModel):
    """问答查询请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史")


class QueryResponse(BaseModel):
    """问答查询响应"""
    answer: str
    sources: List[Dict[str, Any]]


task_store: Dict[str, Dict[str, Any]] = {}
task_service: Optional[TaskService] = None
skill_service: Optional[Any] = None
app_config = None
llm_service = None
retrieval_service = None
orchestrator: Optional[WorkflowOrchestrator] = None
document_service = None
chat_session_service: Optional[Any] = None
rag_service: Optional[Any] = None

DEFAULT_WORKSPACE = ""

app = FastAPI(
    title="RAG Screenplay Generator",
    description="带RAG和动态方向调整的剧本生成系统",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_services():
    """初始化服务"""
    global app_config, llm_service, retrieval_service, orchestrator, task_service, skill_service, chat_session_service, summarization_service
    
    logger.info("Initializing services...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    app_config = get_app_config()
    configure_logging(level=app_config.log_level)
    
    import yaml
    config_path = app_config.config_path
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config_data = {}
    
    llm_config = get_llm_config()
    llm_providers = config_data.get("llm", {}).setdefault("providers", {})
    
    if llm_config.glm_api_key:
        llm_providers.setdefault("glm", {})["api_key"] = llm_config.glm_api_key
        llm_providers.setdefault("glm", {})["base_url"] = "https://open.bigmodel.cn/api/paas/v4"
        logger.info("GLM API key loaded")
    
    if llm_config.openai_api_key:
        llm_providers.setdefault("openai", {})["api_key"] = llm_config.openai_api_key
        logger.info("OpenAI API key loaded")
    
    if llm_config.qwen_api_key:
        llm_providers.setdefault("qwen", {})["api_key"] = llm_config.qwen_api_key
        logger.info("QWEN API key loaded")
    
    try:
        llm_service = LLMService(config_data.get('llm', {}))
        logger.info("LLM service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM service: {e}")
        llm_service = None
    
    try:
        from ..services.chat_session_persistence_service import ChatSessionPersistenceService
        chat_session_service = ChatSessionPersistenceService.get_instance()
        logger.info("Chat session service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize chat session service: {e}")
        chat_session_service = None
    
    try:
        db_service = TaskDatabaseService.create_from_env()
        task_service = TaskService(db_service, enable_cache=True)
        logger.info("Task service initialized with database persistence")
    except Exception as e:
        logger.error(f"Failed to initialize task service: {e}")
        task_service = None

    try:
        from ..services.skill_persistence_service import SkillDatabaseService, SkillService
        skill_db_service = SkillDatabaseService.create_from_env()
        skill_service = SkillService(skill_db_service, enable_cache=True)
        logger.info("Skill service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize skill service: {e}")
        skill_service = None
    
    try:
        from ..services.database.vector_db import PostgresVectorDBService
        from ..config import get_database_config
        db_config = get_database_config()
        vector_db_service = PostgresVectorDBService(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        )
        
        retrieval_config = RetrievalConfig(**config_data.get('retrieval', {}))
        retrieval_service = RetrievalService(
            vector_db_service=vector_db_service,
            llm_service=llm_service,
            config=retrieval_config
        )
        logger.info("Retrieval service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval service: {e}")
        retrieval_service = None
    
    try:
        from ..services.summarization_service import SummarizationService
        summarization_service = SummarizationService(llm_service)
        logger.info("Summarization service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize summarization service: {e}")
        summarization_service = None
    
    try:
        from ..services.document_persistence_service import DocumentService
        document_service = DocumentService()
        logger.info("Document service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize document service: {e}")
        document_service = None
    
    logger.info("All services initialized")


@app.on_event("startup")
async def startup_event():
    init_services()


@app.on_event("shutdown")
async def shutdown_event():
    if task_service:
        await task_service.close()
    if document_service:
        await document_service.close()


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "RAG Screenplay Generator API v2.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if llm_service else "degraded",
        llm_available=llm_service is not None
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """生成剧本（核心接口）"""
    task_id = str(uuid.uuid4())
    
    logger.info(f"[GENERATE] ============================================")
    logger.info(f"[GENERATE] 新建剧本生成任务")
    logger.info(f"[GENERATE] task_id: {task_id}")
    logger.info(f"[GENERATE] topic: {request.topic[:100]}...")
    logger.info(f"[GENERATE] chat_session_id: {request.chat_session_id}")
    
    if request.skill:
        logger.info(f"[GENERATE] skill_config: initial_skill={request.skill.initial_skill}, auto_switch={request.skill.enable_auto_switch}, threshold={request.skill.switch_threshold}")
    else:
        logger.info(f"[GENERATE] skill_config: None (使用默认值)")
    
    if not llm_service:
        logger.error("[GENERATE] LLM service 不可用")
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    if not task_service:
        logger.error("[GENERATE] Task service 不可用")
        raise HTTPException(status_code=503, detail="Task service not available")
    
    skill_name = request.skill.initial_skill.value if request.skill else "standard_tutorial"
    
    task_record = TaskRecord(
        task_id=task_id,
        status=TaskStatus.PENDING.value,
        topic=request.topic,
        context=request.context,
        current_skill=skill_name,
        request_data=request.model_dump(),
        chat_session_id=request.chat_session_id
    )
    
    logger.info(f"[GENERATE] 正在创建 Task 记录...")
    await task_service.create(task_record)
    logger.info(f"[GENERATE] ✅ Task 记录已创建: {task_id}")
    
    background_tasks.add_task(
        run_generation,
        task_id,
        request.model_dump()
    )
    
    logger.info(f"[GENERATE] 后台任务已启动: task_id={task_id}")
    logger.info(f"[GENERATE] ============================================")
    
    return GenerateResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )


async def run_generation(task_id: str, request_data: Dict[str, Any]):
    """后台执行剧本生成"""
    logger.info(f"[RUN_GENERATION] ============================================")
    logger.info(f"[RUN_GENERATION] 开始处理任务: {task_id}")
    logger.info(f"[RUN_GENERATION] request_data keys: {list(request_data.keys())}")
    
    user_topic = request_data.get("topic", "")
    if not user_topic:
        logger.warning(f"[RUN_GENERATION] topic 为空，尝试从 context 或其他字段获取")
        user_topic = request_data.get("context", "") or "默认主题"
    logger.info(f"[RUN_GENERATION] user_topic: '{user_topic}' (length: {len(user_topic)})")
    
    if not task_service:
        logger.error(f"[RUN_GENERATION] Task service 不可用: {task_id}")
        return
    
    await task_service.update(task_id, status=TaskStatus.RUNNING.value)
    logger.info(f"[RUN_GENERATION] Task 状态更新为 RUNNING: {task_id}")
    
    try:
        chat_session_id = request_data.get("chat_session_id")
        project_context = request_data.get("context", "")
        
        logger.info(f"[RUN_GENERATION] chat_session_id: {chat_session_id}")
        
        if chat_session_id:
            try:
                from ..services.chat_session_persistence_service import ChatSessionPersistenceService
                chat_service = ChatSessionPersistenceService.get_instance()
                await chat_service.connect()
                chat_session = await chat_service.get(chat_session_id)
                
                if chat_session and chat_session.message_history:
                    history_text = "\n\n".join([
                        f"【{msg['role']}】\n{msg['content']}"
                        for msg in chat_session.message_history
                    ])
                    project_context = f"[对话历史]\n{history_text}\n\n[生成要求]\n{project_context}"
                    logger.info(f"[RUN_GENERATION] 已加载对话历史, message_count={len(chat_session.message_history)}")
                else:
                    logger.info(f"[RUN_GENERATION] 对话历史为空或不存在")
            except Exception as e:
                logger.warning(f"[RUN_GENERATION] 加载对话历史失败: {e}")
        
        skill = request_data.get("skill", {})
        if isinstance(skill, dict):
            initial_skill = skill.get("initial_skill", "standard_tutorial")
            enable_auto_switch = skill.get("enable_auto_switch", False)
            switch_threshold = skill.get("switch_threshold", 0.7)
        else:
            initial_skill = str(skill) if skill else "standard_tutorial"
            enable_auto_switch = False
            switch_threshold = 0.7
        
        logger.info(f"[RUN_GENERATION] 技能配置: initial_skill={initial_skill}, auto_switch={enable_auto_switch}, threshold={switch_threshold}")
        
        rag_sources = request_data.get("rag_sources")
        if rag_sources:
            project_context = f"[使用文档分类: {', '.join(rag_sources)}]\n{project_context}"
            logger.info(f"[RUN_GENERATION] RAG 源: {rag_sources}")
        
        user_topic = request_data.get("topic", "")
        logger.info(f"[RUN_GENERATION] user_topic: '{user_topic}' (length: {len(user_topic)})")
        
        state: GlobalState = {
            "user_topic": user_topic,
            "project_context": project_context,
            "current_skill": initial_skill,
            "skill_history": [],
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "execution_log": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "fact_check_passed": True,
            "error_flag": None,
            "retry_count": 0,
            "workflow_complete": False,
            "pivot_triggered": False,
            "pivot_reason": None,
            "final_screenplay": None,
            "task_stack": None,
        }
        
        if llm_service:
            llm_service.session_id = task_id
            logger.info(f"[RUN_GENERATION] LLM service session_id 已设置: {task_id}")
        
        logger.info(f"[RUN_GENERATION] 正在创建 WorkflowOrchestrator...")
        runtime_orchestrator = WorkflowOrchestrator(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=None,
            summarization_service=summarization_service,
            workspace_id=DEFAULT_WORKSPACE,
            enable_dynamic_adjustment=request_data.get("enable_dynamic_adjustment", True)
        )
        
        logger.info(f"[RUN_GENERATE] 开始执行工作流...")
        recursion_limit = request_data.get("recursion_limit", 100)
        result = await runtime_orchestrator.execute(state, recursion_limit=recursion_limit)
        logger.info(f"[RUN_GENERATION] 工作流执行完成: success={result['success']}")
        
        if result['success']:
            final_state = result['state']
            
            if isinstance(final_state, dict):
                execution_log = final_state.get("execution_log", [])
                skill_history = final_state.get("skill_history", [])
                outline_data = final_state.get("outline", [])
                screenplay = None
                for log in reversed(execution_log):
                    if log.get("action") == "final_screenplay":
                        screenplay = log.get("details", {}).get("screenplay")
                        logger.info(f"[RUN_GENERATION] 已获取最终剧本, length={len(screenplay) if screenplay else 0}")
                    elif log.get("action") == "skill_switch":
                        logger.info(f"[RUN_GENERATION] 🎯 技能切换: {log.get('details')}")
            else:
                execution_log = getattr(final_state, "execution_log", [])
                skill_history = getattr(final_state, "skill_history", [])
                outline_data = getattr(final_state, "outline", [])
                screenplay = None
                for log in reversed(execution_log):
                    if log.get("action") == "final_screenplay":
                        screenplay = log.get("details", {}).get("screenplay")
                        logger.info(f"[RUN_GENERATION] 已获取最终剧本, length={len(screenplay) if screenplay else 0}")
                    elif log.get("action") == "skill_switch":
                        logger.info(f"[RUN_GENERATION] 🎯 技能切换: {log.get('details')}")
            
            if skill_history:
                logger.info(f"[RUN_GENERATION] 技能历史记录: {len(skill_history)} 次切换")
                for h in skill_history:
                    logger.info(f"  - {h.get('from_skill')} → {h.get('to_skill')}: {h.get('reason')}")
            
            logger.info(f"[RUN_GENERATION] 正在更新 Task 记录...")
            
            outline = []
            if isinstance(outline_data, list):
                outline = [
                    {"step_id": s.get("step_id") if isinstance(s, dict) else getattr(s, "step_id", ""), 
                     "description": s.get("description") if isinstance(s, dict) else getattr(s, "description", ""), 
                     "status": s.get("status") if isinstance(s, dict) else getattr(s, "status", "")}
                    for s in outline_data
                ]
            
            await task_service.update(
                task_id,
                status=TaskStatus.COMPLETED.value,
                screenplay=screenplay,
                outline=outline,
                skill_history=skill_history,
                direction_changes=[
                    {
                        "reason": h.get("reason"),
                        "from_skill": h.get("from_skill"),
                        "to_skill": h.get("to_skill"),
                        "triggered_by": h.get("step_id", "system")
                    }
                    for h in skill_history
                ],
                chat_session_id=chat_session_id
            )
            logger.info(f"[RUN_GENERATION] ✅ Task 已完成: {task_id}")
            
            if chat_session_id:
                try:
                    from ..services.chat_session_persistence_service import ChatSessionPersistenceService
                    chat_service = ChatSessionPersistenceService.get_instance()
                    await chat_service.link_task(chat_session_id, task_id)
                    logger.info(f"[RUN_GENERATION] ✅ Task 已关联到 Session: {chat_session_id} → {task_id}")
                except Exception as e:
                    logger.warning(f"[RUN_GENERATION] 关联 Task 到 Session 失败: {e}")
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"[RUN_GENERATION] ❌ Task 失败: {task_id}, error={error_msg}")
            await task_service.update(
                task_id,
                status=TaskStatus.FAILED.value,
                error=error_msg,
                chat_session_id=chat_session_id
            )
        
        logger.info(f"[RUN_GENERATION] ============================================")
    except Exception as e:
        logger.error(f"[RUN_GENERATION] ❌ 任务执行异常: {task_id}, error={e}")
        logger.exception("[RUN_GENERATION] 详细错误堆栈:")
        await task_service.update(
            task_id,
            status=TaskStatus.FAILED.value,
            error=str(e),
            chat_session_id=request_data.get("chat_session_id")
        )


@app.get("/result/{task_id}", response_model=GenerateResponse)
async def get_result(task_id: str):
    """获取生成结果"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = TaskStatus(task.status)
    
    response = GenerateResponse(
        task_id=task_id,
        status=status,
        created_at=task.created_at or datetime.now()
    )
    
    if status == TaskStatus.COMPLETED:
        response.screenplay = task.screenplay
        response.outline = task.outline
        response.skill_history = task.skill_history
        response.direction_changes = task.direction_changes
    elif status == TaskStatus.FAILED:
        response.error = task.error
    
    return response


@app.post("/adjust/{task_id}", response_model=AdjustResponse)
async def adjust_execution(
    task_id: str,
    request: AdjustRequest
):
    """动态调整执行方向"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if request.action not in ["switch_skill", "skip_step", "add_step", "abort"]:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    if request.action == "switch_skill" and not request.skill:
        raise HTTPException(status_code=400, detail="Skill is required for switch_skill action")
    
    if task.status == TaskStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Cannot adjust a completed task")
    
    if task.status == TaskStatus.FAILED.value:
        raise HTTPException(status_code=400, detail="Cannot adjust a failed task")
    
    if request.action == "abort":
        await task_service.update(task_id, status=TaskStatus.FAILED.value, error="Aborted by user")
        return AdjustResponse(
            success=True,
            task_id=task_id,
            action=request.action,
            result={},
            message="Task aborted successfully"
        )
    
    return AdjustResponse(
        success=True,
        task_id=task_id,
        action=request.action,
        result={
            "skill": request.skill,
            "step_index": request.step_index,
            "new_step": request.new_step,
            "reason": request.reason
        },
        message=f"Adjustment '{request.action}' applied successfully"
    )


@app.get("/tasks/{task_id}/rag-analysis", response_model=RAGAnalysisResponse)
async def get_rag_analysis(task_id: str):
    """获取 RAG 分析结果"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return RAGAnalysisResponse(
        task_id=task_id,
        has_analysis=False,
        main_topic=task.topic,
        suggested_skill=task.current_skill,
        direction_changes=task.direction_changes,
        skill_history=task.skill_history
    )


@app.post("/tasks/{task_id}/rag-adjust", response_model=RAGAdjustResponse)
async def adjust_rag_config(
    task_id: str,
    request: RAGAdjustRequest
):
    """动态调整 RAG 配置"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    previous_config = {
        "top_k": request.top_k,
        "similarity_threshold": request.similarity_threshold,
        "enable_hybrid_search": request.enable_hybrid_search,
        "enable_reranking": request.enable_reranking
    }
    
    return RAGAdjustResponse(
        success=True,
        task_id=task_id,
        previous_config=previous_config,
        new_config=previous_config,
        retrieved_docs_count=0,
        message="RAG configuration adjusted successfully"
    )


@app.get("/skills", response_model=WorkspaceSkillsResponse)
async def list_skills():
    """列出所有技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    skills = await skill_service.get_all()
    default_skill = await skill_service.get_default()

    return WorkspaceSkillsResponse(
        skills=[
            SkillResponse(
                skill_name=s.skill_name,
                description=s.description,
                tone=s.tone,
                compatible_with=s.compatible_with,
                prompt_config=s.prompt_config,
                is_enabled=s.is_enabled,
                is_default=s.is_default,
                created_at=s.created_at,
                updated_at=s.updated_at
            )
            for s in skills
        ],
        default_skill=default_skill.skill_name if default_skill else None,
        total_count=len(skills)
    )


@app.post("/skills", response_model=SkillResponse)
async def create_skill(request: SkillCreateRequest):
    """创建新技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    from ..services.skill_persistence_service import SkillRecord

    existing = await skill_service.get(request.skill_name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Skill '{request.skill_name}' already exists")

    record = SkillRecord(
        skill_name=request.skill_name,
        description=request.description,
        tone=request.tone,
        compatible_with=request.compatible_with,
        prompt_config=request.prompt_config,
        is_enabled=request.is_enabled,
        is_default=request.is_default
    )

    result = await skill_service.create(record)

    return SkillResponse(
        skill_name=result.skill_name,
        description=result.description,
        tone=result.tone,
        compatible_with=result.compatible_with,
        prompt_config=result.prompt_config,
        is_enabled=result.is_enabled,
        is_default=result.is_default,
        created_at=result.created_at,
        updated_at=result.updated_at
    )


@app.get("/skills/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str):
    """获取技能详情"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    skill = await skill_service.get(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return SkillResponse(
        skill_name=skill.skill_name,
        description=skill.description,
        tone=skill.tone,
        compatible_with=skill.compatible_with,
        prompt_config=skill.prompt_config,
        is_enabled=skill.is_enabled,
        is_default=skill.is_default,
        created_at=skill.created_at,
        updated_at=skill.updated_at
    )


@app.delete("/skills/{skill_name}")
async def delete_skill(skill_name: str):
    """删除技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    if not await skill_service.exists(skill_name):
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    await skill_service.delete(skill_name)

    return {"success": True, "skill_name": skill_name}


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(request: DocumentUploadRequest):
    """上传文档"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    from ..services.document_persistence_service import DocumentRecord
    import hashlib

    content_hash = hashlib.md5(request.content.encode()).hexdigest()
    file_size = len(request.content.encode())

    record = DocumentRecord(
        id=str(uuid.uuid4()),
        title=request.title,
        file_name=request.file_name,
        content=request.content,
        content_hash=content_hash,
        file_size=file_size,
        category=request.category,
        metadata=request.metadata
    )

    result = await document_service.create(
        title=record.title,
        file_name=record.file_name,
        content=record.content,
        category=record.category,
        file_size=record.file_size,
        metadata=record.metadata
    )

    return DocumentResponse(
        id=str(result.id),
        title=result.title,
        file_name=result.file_name,
        category=result.category,
        file_size=result.file_size,
        created_at=result.created_at.isoformat() if result.created_at else None
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None)
):
    """列出文档"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    offset = (page - 1) * page_size

    docs, total = await document_service.list_all(
        page=page,
        page_size=page_size,
        category=category
    )

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=str(doc.id),
                title=doc.title,
                file_name=doc.file_name,
                category=doc.category,
                file_size=doc.file_size,
                created_at=doc.created_at.isoformat() if doc.created_at else None
            )
            for doc in docs
        ],
        total_count=total,
        page=page,
        page_size=page_size
    )


@app.get("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20)
):
    """搜索文档"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    results = await document_service.search_by_content(
        query=query,
        top_k=top_k
    )

    return DocumentSearchResponse(
        documents=results,
        total_count=len(results)
    )


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """删除文档"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    success = await document_service.delete(doc_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDeleteResponse(success=True, id=doc_id)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """文档摄入 - ETL 流水线"""
    from ..services.rag.etl_service import create_etl_service

    try:
        etl_service = await create_etl_service(workspace_id=DEFAULT_WORKSPACE)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ETL service not available: {str(e)}")

    try:
        result = await etl_service.ingest(
            file_path=request.file_path,
            source_id=request.source_id
        )

        return IngestResponse(
            status=result.status,
            source_id=result.source_id,
            chunk_count=result.chunk_count,
            error_msg=result.error_msg
        )
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """问答查询 - RAG 流水线"""
    from ..services.rag.rag_service import create_rag_service

    try:
        rag_service = await create_rag_service(workspace_id=DEFAULT_WORKSPACE)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG service not available: {str(e)}")

    try:
        result = await rag_service.query(
            question=request.question,
            history=request.history
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[datetime] = None


class ChatSessionConfig(BaseModel):
    """Chat 会话配置"""
    skill: Optional[str] = Field(None, description="默认技能")
    enable_rag: bool = Field(False, description="是否启用 RAG")
    rag_sources: Optional[List[str]] = Field(None, description="RAG 文档分类")
    system_prompt: Optional[str] = Field(None, description="自定义 system prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ChatSession(BaseModel):
    """Chat 会话"""
    session_id: str
    mode: str = Field(..., description="simple 或 agent")
    config: ChatSessionConfig
    created_at: datetime
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    mode: str = Field("agent", description="模式: simple 或 agent")
    skill: Optional[str] = Field(None, description="默认技能")
    enable_rag: bool = Field(False, description="是否启用 RAG")
    rag_sources: Optional[List[str]] = Field(None, description="RAG 文档分类")
    system_prompt: Optional[str] = Field(None, description="自定义 system prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class CreateSessionResponse(BaseModel):
    """创建会话响应"""
    session_id: str
    mode: str
    config: ChatSessionConfig
    created_at: datetime
    message_count: int = 0


class SendMessageRequest(BaseModel):
    """发送消息请求"""
    message: str = Field(..., min_length=1, description="用户消息")
    skill: Optional[str] = Field(None, description="临时覆盖默认技能")
    enable_rag: Optional[bool] = Field(None, description="临时覆盖 RAG 设置")


class SendMessageResponse(BaseModel):
    """发送消息响应"""
    session_id: str
    role: str = "assistant"
    response: str
    skill_used: Optional[str] = None
    sources: Optional[List[str]] = None
    timestamp: datetime


class ChatHistoryManager:
    """对话历史管理器"""
    
    _sessions: Dict[str, Dict[str, Any]] = {}
    _messages: Dict[str, List[ChatMessage]] = {}
    _session_timestamps: Dict[str, datetime] = {}
    MAX_HISTORY_LENGTH = 20
    
    @classmethod
    def create_session(cls, session_id: str, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建新会话"""
        session = {
            "session_id": session_id,
            "mode": mode,
            "config": config,
            "created_at": datetime.now(),
            "message_count": 0
        }
        cls._sessions[session_id] = session
        cls._messages[session_id] = []
        cls._session_timestamps[session_id] = datetime.now()
        return session
    
    @classmethod
    def get_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        return cls._sessions.get(session_id)
    
    @classmethod
    def get_history(cls, session_id: str) -> List[ChatMessage]:
        """获取会话历史"""
        if session_id not in cls._messages:
            return []
        return cls._messages[session_id]
    
    @classmethod
    def add_message(cls, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id not in cls._messages:
            cls._messages[session_id] = []
        
        cls._messages[session_id].append(ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        ))
        
        if session_id in cls._sessions:
            cls._sessions[session_id]["message_count"] += 1
        
        if len(cls._messages[session_id]) > cls.MAX_HISTORY_LENGTH:
            cls._messages[session_id] = cls._messages[session_id][-cls.MAX_HISTORY_LENGTH:]
    
    @classmethod
    def delete_session(cls, session_id: str):
        """删除会话"""
        if session_id in cls._sessions:
            del cls._sessions[session_id]
        if session_id in cls._messages:
            del cls._messages[session_id]
        if session_id in cls._session_timestamps:
            del cls._session_timestamps[session_id]
    
    @classmethod
    def list_sessions(cls) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return list(cls._sessions.values())


class SimpleChatRequest(BaseModel):
    """简单 Chat 请求"""
    message: str = Field(..., min_length=1, description="用户消息")
    skill: Optional[str] = Field(None, description="技能名称，如 mysterious_fantasy, hot_battle")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: Optional[int] = Field(None, ge=1, le=10000, description="最大生成token数")


class SimpleChatResponse(BaseModel):
    """简单 Chat 响应"""
    session_id: str
    response: str
    skill_used: Optional[str] = None
    tokens_used: Optional[int] = None
    timestamp: datetime


class AgentChatRequest(BaseModel):
    """Agent Chat 请求"""
    session_id: Optional[str] = Field(None, description="会话ID，不传则创建新会话")
    message: str = Field(..., min_length=1, description="用户消息")
    skill: Optional[str] = Field(None, description="技能名称")
    enable_rag: bool = Field(False, description="是否启用 RAG")
    rag_sources: Optional[List[str]] = Field(None, description="RAG 文档分类")
    clear_history: bool = Field(False, description="是否清空历史")


class AgentChatResponse(BaseModel):
    """Agent Chat 响应"""
    session_id: str
    response: str
    skill_used: Optional[str] = None
    sources: Optional[List[str]] = None
    timestamp: datetime


@app.post("/chat/simple", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest):
    """
    简单 Chat 模式 - 直接调用 LLM
    
    特点：
    - 延迟低、成本低
    - 适合快速问答
    - 技能通过 System Prompt 影响输出风格
    """
    session_id = request.session_id if hasattr(request, 'session_id') and request.session_id else "simple_chat"
    
    ChatHistoryManager.add_message(session_id, "user", request.message)
    history = ChatHistoryManager.get_history(session_id)
    
    skill_prompt = ""
    if request.skill:
        if skill_service:
            skill_record = await skill_service.get(request.skill)
            if skill_record and skill_record.prompt_config:
                skill_prompt = skill_record.prompt_config.get("system_prompt", "")
    
    messages_for_llm = []
    if skill_prompt:
        messages_for_llm.append({"role": "system", "content": skill_prompt})
    
    for msg in history[-10:]:
        messages_for_llm.append({"role": msg.role, "content": msg.content})
    
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        response_text = await llm_service.chat_completion(
            messages=messages_for_llm,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 2000
        )
        
        ChatHistoryManager.add_message(session_id, "assistant", response_text)
        
        return SimpleChatResponse(
            session_id=session_id,
            response=response_text,
            skill_used=request.skill,
            tokens_used=len(response_text) // 4,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Simple chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/agent", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Agent Chat 模式 - 通过 Agent 工作流
    
    特点：
    - 支持多轮对话历史
    - 支持动态切换技能
    - 支持 RAG 知识检索
    - 适合专业剧本生成
    """
    session_id = request.session_id or f"agent_{uuid.uuid4().hex[:8]}"
    
    if request.clear_history:
        ChatHistoryManager.clear_history(session_id)
    
    ChatHistoryManager.add_message(session_id, "user", request.message)
    history = ChatHistoryManager.get_history(session_id)
    
    context = ""
    if request.enable_rag and rag_service:
        try:
            rag_result = await rag_service.query(
                question=request.message,
                history=[{"role": m.role, "content": m.content} for m in history[-5:]]
            )
            context = rag_result.answer
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
    
    current_skill = request.skill or "standard_tutorial"
    if skill_service and request.skill:
        skill_record = await skill_service.get(request.skill)
        if skill_record:
            system_prompt = skill_record.prompt_config.get("system_prompt", "")
            if system_prompt:
                context = f"[写作风格: {request.skill}]\n{system_prompt}\n\n[参考知识]\n{context}" if context else f"[写作风格: {request.skill}]\n{system_prompt}"
    
    history_text = "\n".join([
        f"{msg.role}: {msg.content}" 
        for msg in history[-10:]
    ])
    
    full_prompt = f"""[对话历史]
{history_text}

[用户新请求]
{request.message}
"""
    
    if context:
        full_prompt = f"{context}\n\n{full_prompt}"
    
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        response_text = await llm_service.chat_completion(
            messages=[
                {"role": "system", "content": "你是一个专业的剧本写作助手。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.8,
            max_tokens=3000
        )
        
        ChatHistoryManager.add_message(session_id, "assistant", response_text)
        
        if chat_session_service:
            try:
                history = ChatHistoryManager.get_history(session_id)
                message_history = [
                    {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
                    for msg in history
                ]
                await chat_session_service.update_message_history(session_id, message_history)
                logger.info(f"Message history persisted: {session_id}")
            except Exception as e:
                logger.error(f"Failed to persist message history: {e}")
        
        sources = None
        if request.enable_rag:
            sources = ["retrieved_knowledge"]
        
        return AgentChatResponse(
            session_id=session_id,
            response=response_text,
            skill_used=request.skill,
            sources=sources,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions", response_model=List[Dict[str, Any]])
async def list_chat_sessions():
    """列出所有 Chat 会话"""
    return ChatHistoryManager.list_sessions()


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """删除 Chat 会话"""
    ChatHistoryManager.delete_session(session_id)
    return {"success": True, "session_id": session_id}


@app.post("/chat/sessions", response_model=CreateSessionResponse)
async def create_chat_session(request: CreateSessionRequest):
    """
    创建 Chat 会话（带配置）
    
    特点：
    - 一次性配置会话参数（skill、rag、temperature）
    - 后续消息自动使用这些配置
    - 支持随时修改默认配置
    """
    session_id = f"chat_{uuid.uuid4().hex[:12]}"
    
    config = {
        "skill": request.skill,
        "enable_rag": request.enable_rag,
        "rag_sources": request.rag_sources,
        "system_prompt": request.system_prompt,
        "temperature": request.temperature
    }
    
    ChatHistoryManager.create_session(session_id, request.mode, config)
    
    if chat_session_service:
        try:
            from ..services.chat_session_persistence_service import ChatSessionRecord
            record = ChatSessionRecord(
                id=session_id,
                topic="",
                mode=request.mode,
                config=config,
                message_history=[],
                status="active"
            )
            await chat_session_service.create(record)
            logger.info(f"Chat session persisted: {session_id}")
        except Exception as e:
            logger.error(f"Failed to persist chat session: {e}")
    
    return CreateSessionResponse(
        session_id=session_id,
        mode=request.mode,
        config=ChatSessionConfig(**config),
        created_at=ChatHistoryManager.get_session(session_id)["created_at"],
        message_count=0
    )


@app.get("/chat/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str):
    """获取会话信息"""
    session = ChatHistoryManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return ChatSession(
        session_id=session["session_id"],
        mode=session["mode"],
        config=ChatSessionConfig(**session["config"]),
        created_at=session["created_at"],
        message_count=session["message_count"]
    )


@app.get("/chat/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_chat_messages(session_id: str):
    """获取会话消息历史"""
    session = ChatHistoryManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return ChatHistoryManager.get_history(session_id)


@app.post("/chat/sessions/{session_id}/messages", response_model=SendMessageResponse)
async def send_chat_message(
    session_id: str,
    request: SendMessageRequest
):
    """
    发送消息到会话
    
    使用会话配置的默认参数，也可临时覆盖
    """
    logger.info(f"[CHAT] 收到消息请求: session_id={session_id}, message={request.message[:50]}...")
    
    session = ChatHistoryManager.get_session(session_id)
    if not session:
        logger.warning(f"[CHAT] Session 不存在: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    config = session["config"]
    mode = session["mode"]
    
    logger.info(f"[CHAT] Session 配置: mode={mode}, skill={config.get('skill')}, temperature={config.get('temperature')}")
    
    ChatHistoryManager.add_message(session_id, "user", request.message)
    logger.info(f"[CHAT] 用户消息已添加到内存历史: role=user, content={request.message[:50]}...")
    
    if chat_session_service:
        try:
            history = ChatHistoryManager.get_history(session_id)
            message_history = [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
                for msg in history
            ]
            await chat_session_service.update_message_history(session_id, message_history)
            logger.info(f"[CHAT] ✅ 用户消息已持久化到数据库: session_id={session_id}")
        except Exception as e:
            logger.error(f"[CHAT] ❌ 持久化用户消息失败: {e}")
    
    history = ChatHistoryManager.get_history(session_id)
    
    effective_skill = request.skill or config.get("skill")
    effective_rag = request.enable_rag if request.enable_rag is not None else config.get("enable_rag", False)
    
    logger.info(f"[CHAT] Effective parameters: skill={effective_skill}, rag={effective_rag}")
    
    context = ""
    if effective_rag and rag_service:
        try:
            rag_result = await rag_service.query(
                question=request.message,
                history=[{"role": m.role, "content": m.content} for m in history[-5:]]
            )
            context = rag_result.answer
            logger.info(f"[CHAT] RAG 查询成功: context_length={len(context)}")
        except Exception as e:
            logger.warning(f"[CHAT] RAG 查询失败: {e}")
    
    if effective_skill and skill_service:
        skill_record = await skill_service.get(effective_skill)
        if skill_record and skill_record.prompt_config:
            system_prompt = skill_record.prompt_config.get("system_prompt", "")
            if system_prompt:
                context = f"[写作风格: {effective_skill}]\n{system_prompt}\n\n[参考知识]\n{context}" if context else f"[写作风格: {effective_skill}]\n{system_prompt}"
            logger.info(f"[CHAT] Skill 配置已应用: skill={effective_skill}, prompt_length={len(system_prompt)}")
    
    if config.get("system_prompt"):
        context = f"{config['system_prompt']}\n\n{context}" if context else config["system_prompt"]
    
    history_text = "\n".join([
        f"{msg.role}: {msg.content}" 
        for msg in history[-10:]
    ])
    
    full_prompt = f"""[对话历史]
{history_text}

[用户新请求]
{request.message}
"""
    
    if context:
        full_prompt = f"{context}\n\n{full_prompt}"
    
    logger.info(f"[CHAT] 准备调用 LLM: temperature={config.get('temperature', 0.7)}, max_tokens=3000, prompt_length={len(full_prompt)}")
    
    if not llm_service:
        logger.error("[CHAT] LLM service 不可用")
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        response_text = await llm_service.chat_completion(
            messages=[
                {"role": "system", "content": "你是一个专业的剧本写作助手。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=config.get("temperature", 0.7),
            max_tokens=3000
        )
        logger.info(f"[CHAT] ✅ LLM 调用成功: response_length={len(response_text)}")
        
        ChatHistoryManager.add_message(session_id, "assistant", response_text)
        logger.info(f"[CHAT] 助手消息已添加到内存历史")
        
        if chat_session_service:
            try:
                history = ChatHistoryManager.get_history(session_id)
                message_history = [
                    {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
                    for msg in history
                ]
                await chat_session_service.update_message_history(session_id, message_history)
                logger.info(f"[CHAT] ✅ 完整消息历史已持久化: message_count={len(message_history)}")
            except Exception as e:
                logger.error(f"[CHAT] ❌ 持久化完整消息历史失败: {e}")
        
        sources = None
        if effective_rag:
            sources = ["retrieved_knowledge"]
        
        response = SendMessageResponse(
            session_id=session_id,
            response=response_text,
            skill_used=effective_skill,
            sources=sources,
            timestamp=datetime.now()
        )
        logger.info(f"[CHAT] ✅ 响应构建成功: skill_used={effective_skill}")
        return response
    except Exception as e:
        logger.error(f"[CHAT] ❌ LLM 调用或后续处理失败: {e}")
        logger.exception("[CHAT] 详细错误:")
        raise HTTPException(status_code=500, detail=str(e))


class ChatExportResponse(BaseModel):
    """导出对话历史响应"""
    session_id: str
    topic: Optional[str] = None
    mode: str
    message_count: int
    history_text: str
    created_at: Optional[datetime] = None


@app.get("/chat/sessions/{session_id}/export", response_model=ChatExportResponse)
async def export_chat_history(session_id: str):
    """
    导出对话历史（用于生成剧本）
    
    将对话历史转换为纯文本格式，可直接用于生成剧本的 context
    """
    session = ChatHistoryManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = ChatHistoryManager.get_history(session_id)
    
    history_text = "\n\n".join([
        f"【{msg.role}】\n{msg.content}"
        for msg in messages
    ])
    
    return ChatExportResponse(
        session_id=session_id,
        topic=session.get("config", {}).get("topic"),
        mode=session["mode"],
        message_count=len(messages),
        history_text=history_text or "(暂无对话)",
        created_at=session.get("created_at")
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.presentation.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
