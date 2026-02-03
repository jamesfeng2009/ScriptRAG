"""简化版 REST API - 专注于剧本生成和动态方向调整"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
from dotenv import load_dotenv

from ..config import get_app_config, get_llm_config, get_database_config
from ..domain.models import SharedState
from ..application.enhanced_orchestrator import EnhancedWorkflowOrchestrator
from ..services.llm.service import LLMService
from ..services.retrieval_service import RetrievalService, RetrievalConfig
from ..services.parser.tree_sitter_parser import TreeSitterParser
from ..services.summarization_service import SummarizationService
from ..infrastructure.logging import setup_logging
from ..services.task_persistence_service import TaskDatabaseService, TaskRecord, TaskService


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
    context: Optional[str] = Field("", description="上下文信息")
    
    skill: Optional[SkillConfig] = Field(default_factory=SkillConfig)
    rag: Optional[RAGConfig] = Field(default_factory=RAGConfig)
    
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


task_store: Dict[str, Dict[str, Any]] = {}
task_service: Optional[TaskService] = None
skill_service: Optional[Any] = None
app_config = None
llm_service = None
retrieval_service = None
parser_service = None
summarization_service = None
orchestrator: Optional[EnhancedWorkflowOrchestrator] = None
document_service = None


app = FastAPI(
    title="RAG Screenplay Generator",
    description="带RAG和动态方向调整的剧本生成系统",
    version="1.0.0"
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
    global app_config, llm_service, retrieval_service, parser_service, summarization_service, orchestrator, task_service, skill_service
    
    logger.info("Initializing services...")
    
    load_dotenv()
    
    app_config = get_app_config()
    setup_logging(level=app_config.log_level)
    
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
        import traceback
        traceback.print_exc()
        skill_service = None
    
    try:
        from ..services.database.vector_db import PostgresVectorDBService
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
        parser_service = TreeSitterParser()
        logger.info("Parser service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize parser service: {e}")
        parser_service = None
    
    try:
        summarization_service = SummarizationService(
            llm_service=llm_service,
            config=config_data.get('retrieval', {}).get('summarization', {})
        )
        logger.info("Summarization service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize summarization service: {e}")
        summarization_service = None
    
    try:
        orchestrator = EnhancedWorkflowOrchestrator(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id=DEFAULT_WORKSPACE,
            enable_dynamic_adjustment=True
        )
        logger.info("Orchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        orchestrator = None
    
    logger.info("All services initialized")


@app.on_event("startup")
async def startup():
    init_services()
    
    if skill_service:
        try:
            await skill_service.db_service.connect()
            await skill_service.db_service.create_table()
            logger.info("Skill service database connected and tables created")
        except Exception as e:
            logger.error(f"Failed to connect skill service database: {e}")
    
    try:
        from ..services.document_persistence_service import DocumentService
        global document_service
        document_service = DocumentService()
        await document_service.init()
        logger.info("Document service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize document service: {e}")
        document_service = None


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "RAG Screenplay Generator", "version": "1.0.0"}


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
    
    logger.info(f"Generating screenplay for topic: {request.topic}")
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task_record = TaskRecord(
        task_id=task_id,
        status=TaskStatus.PENDING.value,
        topic=request.topic,
        context=request.context,
        current_skill=request.skill.initial_skill.value if request.skill else "standard_tutorial",
        request_data=request.model_dump()
    )
    
    await task_service.create(task_record)
    
    background_tasks.add_task(
        run_generation,
        task_id,
        request.model_dump()
    )
    
    return GenerateResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )


async def run_generation(task_id: str, request_data: Dict[str, Any]):
    """后台执行剧本生成"""
    if not task_service:
        logger.error(f"Task service not available for task {task_id}")
        return
    
    await task_service.update(task_id, status=TaskStatus.RUNNING.value)
    
    try:
        skill = request_data.get("skill", {})
        if isinstance(skill, dict):
            initial_skill = skill.get("initial_skill", "standard_tutorial")
        else:
            initial_skill = str(skill) if skill else "standard_tutorial"
        
        state = SharedState(
            user_topic=request_data.get("topic", ""),
            project_context=request_data.get("context", ""),
            current_skill=initial_skill,
            max_retries=request_data.get("max_retries", 3)
        )
        
        from ..application.enhanced_orchestrator import EnhancedWorkflowOrchestrator
        
        if llm_service:
            llm_service.session_id = task_id
        
        runtime_orchestrator = EnhancedWorkflowOrchestrator(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id=DEFAULT_WORKSPACE,
            enable_dynamic_adjustment=request_data.get("enable_dynamic_adjustment", True)
        )
        
        recursion_limit = request_data.get("recursion_limit", 100)
        result = await runtime_orchestrator.execute(state, recursion_limit=recursion_limit)
        
        if result['success']:
            final_state = result['state']
            
            screenplay = None
            for log in reversed(final_state.execution_log):
                if log.get("action") == "final_screenplay":
                    screenplay = log.get("details", {}).get("screenplay")
                    break
            
            await task_service.update(
                task_id,
                status=TaskStatus.COMPLETED.value,
                screenplay=screenplay,
                outline=[
                    {"step_id": s.step_id, "description": s.description, "status": s.status}
                    for s in final_state.outline
                ],
                skill_history=final_state.skill_history,
                direction_changes=[
                    {
                        "reason": h.get("reason"),
                        "from_skill": h.get("from_skill"),
                        "to_skill": h.get("to_skill"),
                        "triggered_by": h.get("step_id", "system")
                    }
                    for h in final_state.skill_history
                ]
            )
            logger.info(f"Task {task_id} completed")
        else:
            await task_service.update(
                task_id,
                status=TaskStatus.FAILED.value,
                error=result.get("error", "Unknown error")
            )
            logger.error(f"Task {task_id} failed: {result.get('error')}")
    
    except Exception as e:
        await task_service.update(
            task_id,
            status=TaskStatus.FAILED.value,
            error=str(e)
        )
        logger.error(f"Task {task_id} error: {e}")


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


@app.get("/result/{task_id}", response_model=GenerateResponse)
async def get_result(task_id: str):
    """获取生成结果"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return GenerateResponse(
        task_id=task_id,
        status=TaskStatus(task.status),
        screenplay=task.screenplay,
        outline=task.outline,
        skill_history=task.skill_history,
        direction_changes=task.direction_changes,
        error=task.error,
        created_at=task.created_at
    )


@app.post("/adjust/{task_id}", response_model=AdjustResponse)
async def adjust_task(task_id: str, request: AdjustRequest):
    """手动调整任务方向（手动干预 RAG/Skill）"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
        raise HTTPException(status_code=400, detail="Task already completed or failed")
    
    logger.info(f"Manual adjustment on task {task_id}: action={request.action}")
    
    result = {}
    
    if request.action == "switch_skill":
        if not request.skill:
            raise HTTPException(status_code=400, detail="Skill is required for switch_skill action")
        
        current_skill = task.current_skill
        task.current_skill = request.skill
        task.skill_history.append({
            "reason": f"manual_adjustment: {request.reason}",
            "from_skill": current_skill,
            "to_skill": request.skill,
            "triggered_by": "user"
        })
        result = {
            "from_skill": current_skill,
            "to_skill": request.skill
        }
        message = f"Skill switched from {current_skill} to {request.skill}"
    
    elif request.action == "skip_step":
        if request.step_index is None:
            raise HTTPException(status_code=400, detail="step_index is required for skip_step action")
        
        if request.step_index < 0 or request.step_index >= len(task.outline):
            raise HTTPException(status_code=400, detail="Invalid step_index")
        
        task.outline[request.step_index]["status"] = "skipped"
        result = {
            "step_index": request.step_index,
            "step_description": task.outline[request.step_index].get("description", "")[:100]
        }
        message = f"Step {request.step_index} skipped"
    
    elif request.action == "add_step":
        if not request.new_step:
            raise HTTPException(status_code=400, detail="new_step is required for add_step action")
        
        new_step = {
            "step_id": f"manual_{len(task.outline) + 1}",
            "description": request.new_step,
            "status": "pending"
        }
        task.outline.append(new_step)
        task.direction_changes.append({
            "reason": f"manual_adjustment: {request.reason}",
            "action": "add_step",
            "step_description": request.new_step[:100]
        })
        result = {
            "new_step": request.new_step[:100],
            "total_steps": len(task.outline)
        }
        message = f"New step added: {request.new_step[:50]}..."
    
    elif request.action == "abort":
        task.status = TaskStatus.FAILED.value
        task.error = f"Aborted by user: {request.reason}"
        result = {"aborted": True}
        message = "Task aborted"
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
    
    await task_service.update(
        task_id,
        current_skill=task.current_skill,
        skill_history=task.skill_history,
        outline=task.outline,
        direction_changes=task.direction_changes,
        status=task.status,
        error=task.error
    )
    
    return AdjustResponse(
        success=True,
        task_id=task_id,
        action=request.action,
        result=result,
        message=message
    )


@app.get("/tasks/{task_id}/rag-analysis", response_model=RAGAnalysisResponse)
async def get_rag_analysis(task_id: str):
    """获取任务的RAG分析结果"""
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    task = await task_service.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    rag_analysis = getattr(task, 'rag_analysis', None)
    
    return RAGAnalysisResponse(
        task_id=task_id,
        has_analysis=rag_analysis is not None,
        content_types=rag_analysis.get("content_types") if rag_analysis else None,
        main_topic=rag_analysis.get("main_topic") if rag_analysis else None,
        sub_topics=rag_analysis.get("sub_topics") if rag_analysis else None,
        difficulty_level=rag_analysis.get("difficulty_level") if rag_analysis else None,
        tone_style=rag_analysis.get("tone_style") if rag_analysis else None,
        key_concepts=rag_analysis.get("key_concepts") if rag_analysis else None,
        warnings=rag_analysis.get("warnings") if rag_analysis else None,
        prerequisites=rag_analysis.get("prerequisites") if rag_analysis else None,
        suggested_skill=rag_analysis.get("suggested_skill") if rag_analysis else None,
        confidence=rag_analysis.get("confidence") if rag_analysis else None,
        direction_changes=task.direction_changes,
        skill_history=task.skill_history,
        analyzed_at=datetime.now()
    )


@app.post("/tasks/{task_id}/rag-adjust", response_model=RAGAdjustResponse)
async def adjust_rag(task_id: str, request: RAGAdjustRequest):
    """动态调整RAG参数并重新分析
    
    支持调整：
    - top_k: 检索文档数量
    - similarity_threshold: 相似度阈值
    - enable_hybrid_search: 启用混合搜索
    - enable_reranking: 启用重排序
    - force_reanalysis: 强制重新分析
    - query: 重新检索的查询词
    """
    if not task_service:
        raise HTTPException(status_code=503, detail="Task service not available")
    
    if not retrieval_service:
        raise HTTPException(status_code=503, detail="Retrieval service not available")
    
    task = await task_service.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    previous_config = {
        "top_k": getattr(task, 'rag_top_k', None),
        "similarity_threshold": getattr(task, 'rag_similarity_threshold', None),
        "enable_hybrid_search": getattr(task, 'rag_enable_hybrid_search', None),
        "enable_reranking": getattr(task, 'rag_enable_reranking', None)
    }
    
    new_config = {}
    update_fields = {}
    
    if request.top_k is not None:
        new_config["top_k"] = request.top_k
        update_fields["rag_top_k"] = request.top_k
    
    if request.similarity_threshold is not None:
        new_config["similarity_threshold"] = request.similarity_threshold
        update_fields["rag_similarity_threshold"] = request.similarity_threshold
    
    if request.enable_hybrid_search is not None:
        new_config["enable_hybrid_search"] = request.enable_hybrid_search
        update_fields["rag_enable_hybrid_search"] = request.enable_hybrid_search
    
    if request.enable_reranking is not None:
        new_config["enable_reranking"] = request.enable_reranking
        update_fields["rag_enable_reranking"] = request.enable_reranking
    
    query = request.query or task.topic
    
    previous_config = {
        "top_k": getattr(task, 'rag_top_k', None),
        "similarity_threshold": getattr(task, 'rag_similarity_threshold', None),
        "enable_hybrid_search": getattr(task, 'rag_enable_hybrid_search', None),
        "enable_reranking": getattr(task, 'rag_enable_reranking', None)
    }
    
    new_config = {}
    update_fields = {}
    
    if request.top_k is not None:
        new_config["top_k"] = request.top_k
        update_fields["rag_top_k"] = request.top_k
    
    if request.similarity_threshold is not None:
        new_config["similarity_threshold"] = request.similarity_threshold
        update_fields["rag_similarity_threshold"] = request.similarity_threshold
    
    if request.enable_hybrid_search is not None:
        new_config["enable_hybrid_search"] = request.enable_hybrid_search
        update_fields["rag_enable_hybrid_search"] = request.enable_hybrid_search
    
    if request.enable_reranking is not None:
        new_config["enable_reranking"] = request.enable_reranking
        update_fields["rag_enable_reranking"] = request.enable_reranking
    
    retrieved_docs_count = 0
    analysis_result = None
    message_parts = []
    
    if new_config:
        message_parts.append(f"RAG参数已更新")
    
    try:
        if request.force_reanalysis or request.query:
            logger.info(f"Re-retrieving documents for task {task_id} with query: {query}")
            
            if retrieval_service:
                try:
                    search_results = await retrieval_service.hybrid_retrieve(
                        workspace_id=DEFAULT_WORKSPACE,
                        query=query,
                        top_k=request.top_k or 5
                    )
                    
                    retrieved_docs_count = len(search_results)
                    
                    analysis_result = {
                        "query": query,
                        "top_k": request.top_k or 5,
                        "results_count": retrieved_docs_count,
                        "docs_preview": [
                            {
                                "content": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                                "confidence": doc.confidence
                            }
                            for doc in search_results[:3]
                        ]
                    }
                    
                    if retrieved_docs_count > 0:
                        message_parts.append(f"检索到 {retrieved_docs_count} 个相关文档")
                    else:
                        message_parts.append("未检索到相关文档")
                        
                except Exception as retrieval_error:
                    logger.warning(f"Retrieval failed: {retrieval_error}")
                    message_parts.append("文档检索失败（LLM服务不可用），但参数已保存")
        
        if update_fields:
            await task_service.update(task_id, **update_fields)
        
        message = "; ".join(message_parts) if message_parts else "配置已更新"
        
        return RAGAdjustResponse(
            success=True,
            task_id=task_id,
            previous_config=previous_config,
            new_config=new_config if new_config else previous_config,
            retrieved_docs_count=retrieved_docs_count,
            analysis_result=analysis_result,
            message=message
        )
        
    except Exception as e:
        logger.error(f"RAG adjustment failed: {e}")
        return RAGAdjustResponse(
            success=False,
            task_id=task_id,
            previous_config=previous_config,
            new_config=new_config if new_config else {},
            retrieved_docs_count=retrieved_docs_count,
            analysis_result=analysis_result,
            message=f"RAG调整失败: {str(e)}"
        )


class SkillCreateRequest(BaseModel):
    """创建技能请求"""
    skill_name: str = Field(..., min_length=1, max_length=100, description="技能名称")
    description: str = Field(..., description="技能描述")
    tone: str = Field(..., description="语调风格")
    compatible_with: List[str] = Field(default_factory=list, description="兼容的技能列表")
    prompt_config: Dict[str, Any] = Field(default_factory=dict, description="提示配置")
    is_enabled: bool = Field(True, description="是否启用")
    is_default: bool = Field(False, description="是否为默认技能")


class SkillUpdateRequest(BaseModel):
    """更新技能请求"""
    description: Optional[str] = None
    tone: Optional[str] = None
    compatible_with: Optional[List[str]] = None
    prompt_config: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None
    is_default: Optional[bool] = None


class SkillResponse(BaseModel):
    """技能响应"""
    workspace_id: str
    skill_name: str
    description: str
    tone: str
    compatible_with: List[str]
    is_enabled: bool
    is_default: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkspaceSkillsResponse(BaseModel):
    """技能列表响应"""
    skills: List[SkillResponse]
    default_skill: Optional[str] = None
    total_count: int


DEFAULT_WORKSPACE = ""


@app.get("/skills", response_model=WorkspaceSkillsResponse)
async def list_skills():
    """列出所有技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    skills = await skill_service.get_by_workspace(DEFAULT_WORKSPACE)
    default_skill = await skill_service.get_default(DEFAULT_WORKSPACE)

    return WorkspaceSkillsResponse(
        skills=[
            SkillResponse(
                workspace_id=s.workspace_id,
                skill_name=s.skill_name,
                description=s.description,
                tone=s.tone,
                compatible_with=s.compatible_with,
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

    existing = await skill_service.get(DEFAULT_WORKSPACE, request.skill_name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Skill '{request.skill_name}' already exists")

    record = SkillRecord(
        workspace_id=DEFAULT_WORKSPACE,
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
        workspace_id=result.workspace_id,
        skill_name=result.skill_name,
        description=result.description,
        tone=result.tone,
        compatible_with=result.compatible_with,
        is_enabled=result.is_enabled,
        is_default=result.is_default,
        created_at=result.created_at,
        updated_at=result.updated_at
    )


@app.get("/skills/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str):
    """获取指定技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    record = await skill_service.get(DEFAULT_WORKSPACE, skill_name)
    if not record:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return SkillResponse(
        workspace_id=record.workspace_id,
        skill_name=record.skill_name,
        description=record.description,
        tone=record.tone,
        compatible_with=record.compatible_with,
        is_enabled=record.is_enabled,
        is_default=record.is_default,
        created_at=record.created_at,
        updated_at=record.updated_at
    )


@app.patch("/skills/{skill_name}", response_model=SkillResponse)
async def update_skill(skill_name: str, request: SkillUpdateRequest):
    """更新指定技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    result = await skill_service.update(DEFAULT_WORKSPACE, skill_name, **update_data)

    if not result:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return SkillResponse(
        workspace_id=result.workspace_id,
        skill_name=result.skill_name,
        description=result.description,
        tone=result.tone,
        compatible_with=result.compatible_with,
        is_enabled=result.is_enabled,
        is_default=result.is_default,
        created_at=result.created_at,
        updated_at=result.updated_at
    )


@app.delete("/skills/{skill_name}")
async def delete_skill(skill_name: str):
    """删除指定技能"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    if skill_name == "standard_tutorial":
        raise HTTPException(status_code=400, detail="Cannot delete default skill 'standard_tutorial'")

    deleted = await skill_service.delete(DEFAULT_WORKSPACE, skill_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return {"message": f"Skill '{skill_name}' deleted"}


@app.post("/skills/initialize")
async def initialize_skills():
    """初始化默认技能（从全局 SKILLS 复制）"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    skills = await skill_service.ensure_default_skills(DEFAULT_WORKSPACE)

    return {
        "message": f"Initialized {len(skills)} default skills",
        "skills": [s.skill_name for s in skills]
    }


@app.get("/skills/{skill_name}/config")
async def get_skill_config(skill_name: str):
    """获取技能配置（用于剧本生成）"""
    if not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    config = await skill_service.get_skill_config(DEFAULT_WORKSPACE, skill_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return {
        "skill_name": skill_name,
        "description": config.description,
        "tone": config.tone,
        "compatible_with": config.compatible_with
    }


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    title: Optional[str] = None
    category: Optional[str] = None


class DocumentResponse(BaseModel):
    """文档响应"""
    doc_id: str
    title: str
    file_name: str
    category: Optional[str] = None
    file_size: Optional[int] = None
    indexed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class DocumentDeleteResponse(BaseModel):
    """删除文档响应"""
    success: bool = True
    doc_id: str
    message: str = "Document deleted successfully"


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentSearchResponse(BaseModel):
    """文档搜索响应"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    retrieved_at: datetime


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None)
):
    """上传并索引文档（支持文件上传或直接输入内容）"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    
    if file is not None:
        file_content = await file.read()
        text_content = file_content.decode('utf-8', errors='ignore')
        actual_file_name = file.filename or "unknown"
        doc_title = title or actual_file_name
        file_size = len(file_content)
        content_type = file.content_type
    elif content is not None:
        text_content = content
        actual_file_name = file_name or "text_content.txt"
        doc_title = title or actual_file_name
        file_size = len(content.encode('utf-8'))
        content_type = "text/plain"
    else:
        raise HTTPException(status_code=400, detail="Either file or content is required")
    
    try:
        doc = await document_service.create(
            title=doc_title,
            file_name=actual_file_name,
            content=text_content,
            file_path=None,
            category=category,
            file_size=file_size,
            metadata={"content_type": content_type}
        )
        
        return DocumentResponse(
            doc_id=str(doc.id),
            title=doc.title,
            file_name=doc.file_name,
            category=doc.category,
            file_size=doc.file_size,
            indexed_at=doc.indexed_at,
            created_at=doc.created_at
        )
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None)
):
    """列出已索引文档"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    
    docs, total = await document_service.list_all(
        page=page,
        page_size=page_size,
        category=category
    )
    
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                doc_id=str(doc.id),
                title=doc.title,
                file_name=doc.file_name,
                category=doc.category,
                file_size=doc.file_size,
                indexed_at=doc.indexed_at,
                created_at=doc.created_at
            )
            for doc in docs
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@app.get("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(5, ge=1, le=20)
):
    """搜索文档（测试检索效果）"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    
    results = await document_service.search_by_content(query, top_k=top_k)
    
    return DocumentSearchResponse(
        query=query,
        results=results,
        total_results=len(results),
        retrieved_at=datetime.now()
    )


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """删除文档"""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    
    try:
        deleted = await document_service.delete(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentDeleteResponse(doc_id=doc_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


knowledge_graph_service: Optional[Any] = None
retrieval_logs_service: Optional[Any] = None
api_usage_stats_service: Optional[Any] = None
document_chunks_service: Optional[Any] = None


@app.on_event("startup")
async def init_test_services():
    """初始化测试服务"""
    global knowledge_graph_service, retrieval_logs_service, api_usage_stats_service, document_chunks_service
    
    try:
        from ..services.knowledge_graph_service import KnowledgeGraphService
        knowledge_graph_service = KnowledgeGraphService()
        await knowledge_graph_service.initialize()
        logger.info("Knowledge graph service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize knowledge graph service: {e}")
        knowledge_graph_service = None
    
    try:
        from ..services.retrieval_logs_service import RetrievalLogsService
        retrieval_logs_service = RetrievalLogsService()
        await retrieval_logs_service.initialize()
        logger.info("Retrieval logs service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval logs service: {e}")
        retrieval_logs_service = None
    
    try:
        from ..services.api_usage_stats_service import APIUsageStatsService
        api_usage_stats_service = APIUsageStatsService()
        await api_usage_stats_service.initialize()
        logger.info("API usage stats service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize API usage stats service: {e}")
        api_usage_stats_service = None
    
    try:
        from ..services.document_chunks_service import DocumentChunksService
        document_chunks_service = DocumentChunksService()
        await document_chunks_service.initialize()
        logger.info("Document chunks service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize document chunks service: {e}")
        document_chunks_service = None


@app.on_event("shutdown")
async def close_test_services():
    """关闭测试服务"""
    if knowledge_graph_service:
        await knowledge_graph_service.close()
    if retrieval_logs_service:
        await retrieval_logs_service.close()
    if api_usage_stats_service:
        await api_usage_stats_service.close()
    if document_chunks_service:
        await document_chunks_service.close()


class KnowledgeNodeCreate(BaseModel):
    """创建知识节点请求"""
    workspace_id: str = Field(default=DEFAULT_WORKSPACE)
    node_type: str = Field(..., description="节点类型: concept, entity, relation, function, class")
    name: str = Field(..., min_length=1, description="节点名称")
    content: Optional[str] = Field(None, description="节点内容")
    embedding: Optional[List[float]] = Field(None, description="嵌入向量")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source_file: Optional[str] = Field(None)
    line_number: Optional[int] = Field(None)


class KnowledgeNodeResponse(BaseModel):
    """知识节点响应"""
    id: str
    workspace_id: str
    node_type: str
    name: str
    content: Optional[str]
    embedding: Optional[str]
    properties: Dict[str, Any]
    source_file: Optional[str]
    line_number: Optional[int]
    created_at: Optional[str]
    updated_at: Optional[str]


class KnowledgeRelationCreate(BaseModel):
    """创建知识关系请求"""
    workspace_id: str = Field(default=DEFAULT_WORKSPACE)
    source_node_id: str = Field(..., description="源节点 ID")
    target_node_id: str = Field(..., description="目标节点 ID")
    relation_type: str = Field(..., description="关系类型: imports, inherits, calls, contains, depends_on")
    strength: float = Field(1.0, ge=0.0, le=1.0)
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)


class KnowledgeRelationResponse(BaseModel):
    """知识关系响应"""
    id: str
    workspace_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    strength: float
    properties: Dict[str, Any]
    created_at: Optional[str]


class KnowledgeGraphStatsResponse(BaseModel):
    """知识图谱统计响应"""
    workspace_id: str
    nodes_by_type: Dict[str, int]
    relations_by_type: Dict[str, int]
    total_nodes: int
    total_relations: int


class RetrievalLogResponse(BaseModel):
    """检索日志响应"""
    id: str
    workspace_id: str
    user_id: Optional[str]
    query: str
    top_k: int
    result_count: int
    total_score: float
    search_strategy: str
    duration_ms: int
    token_usage: Optional[Dict[str, Any]]
    cost_usd: float
    created_at: Optional[str]


class PopularQueryResponse(BaseModel):
    """热门查询响应"""
    query_preview: str
    query_count: int
    avg_results: float
    total_cost: float
    avg_duration_ms: float


class RetrievalStatsResponse(BaseModel):
    """检索统计响应"""
    workspace_id: str
    period_days: int
    total_retrievals: int
    avg_results_per_query: float
    avg_duration_ms: float
    total_cost_usd: float
    by_strategy: List[Dict[str, Any]]


class APIUsageStatsResponse(BaseModel):
    """API 使用统计响应"""
    period_days: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_requests: int
    active_days: int


class DailyCostResponse(BaseModel):
    """每日成本响应"""
    date: str
    daily_cost_usd: float
    daily_tokens: int
    daily_requests: int


class CleanupResponse(BaseModel):
    """清理响应"""
    deleted_count: int
    message: str


@app.post("/knowledge/nodes", response_model=Dict[str, str])
async def create_knowledge_node(request: KnowledgeNodeCreate):
    """创建知识节点"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        node_id = await knowledge_graph_service.create_node(
            workspace_id=request.workspace_id,
            node_type=request.node_type,
            name=request.name,
            content=request.content,
            embedding=request.embedding,
            properties=request.properties,
            source_file=request.source_file,
            line_number=request.line_number
        )
        return {"node_id": node_id}
    except Exception as e:
        logger.error(f"Failed to create knowledge node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/nodes/{node_id}", response_model=KnowledgeNodeResponse)
async def get_knowledge_node(node_id: str):
    """获取知识节点"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    node = await knowledge_graph_service.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return KnowledgeNodeResponse(**node)


@app.get("/knowledge/nodes", response_model=List[KnowledgeNodeResponse])
async def list_knowledge_nodes(
    workspace_id: str = Query(default=DEFAULT_WORKSPACE),
    node_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """列出知识节点"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    nodes = await knowledge_graph_service.list_nodes(
        workspace_id=workspace_id,
        node_type=node_type,
        limit=limit,
        offset=offset
    )
    return [KnowledgeNodeResponse(**node) for node in nodes]


@app.delete("/knowledge/nodes/{node_id}", response_model=Dict[str, str])
async def delete_knowledge_node(node_id: str):
    """删除知识节点"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    deleted = await knowledge_graph_service.delete_node(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return {"deleted": node_id}


@app.post("/knowledge/relations", response_model=Dict[str, str])
async def create_knowledge_relation(request: KnowledgeRelationCreate):
    """创建知识关系"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        relation_id = await knowledge_graph_service.create_relation(
            workspace_id=request.workspace_id,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            relation_type=request.relation_type,
            strength=request.strength,
            properties=request.properties
        )
        return {"relation_id": relation_id}
    except Exception as e:
        logger.error(f"Failed to create knowledge relation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/relations/{relation_id}", response_model=KnowledgeRelationResponse)
async def get_knowledge_relation(relation_id: str):
    """获取知识关系"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    relation = await knowledge_graph_service.get_relation(relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="Relation not found")
    
    return KnowledgeRelationResponse(**relation)


@app.get("/knowledge/relations", response_model=List[KnowledgeRelationResponse])
async def list_knowledge_relations(
    workspace_id: str = Query(default=DEFAULT_WORKSPACE),
    source_node_id: Optional[str] = Query(None),
    target_node_id: Optional[str] = Query(None),
    relation_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """列出知识关系"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    relations = await knowledge_graph_service.list_relations(
        workspace_id=workspace_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_type=relation_type,
        limit=limit,
        offset=offset
    )
    return [KnowledgeRelationResponse(**rel) for rel in relations]


@app.delete("/knowledge/relations/{relation_id}", response_model=Dict[str, str])
async def delete_knowledge_relation(relation_id: str):
    """删除知识关系"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    deleted = await knowledge_graph_service.delete_relation(relation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Relation not found")
    
    return {"deleted": relation_id}


@app.get("/knowledge/stats", response_model=KnowledgeGraphStatsResponse)
async def get_knowledge_graph_stats(workspace_id: str = Query(default=DEFAULT_WORKSPACE)):
    """获取知识图谱统计"""
    if not knowledge_graph_service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    stats = await knowledge_graph_service.get_graph_stats(workspace_id)
    return KnowledgeGraphStatsResponse(**stats)


@app.post("/retrieval-logs", response_model=Dict[str, str])
async def create_retrieval_log(
    workspace_id: str = Query(default=DEFAULT_WORKSPACE),
    query: str = Query(...),
    top_k: int = Query(5),
    result_count: int = Query(0),
    total_score: float = Query(0.0),
    search_strategy: str = Query("hybrid"),
    duration_ms: int = Query(0),
    cost_usd: float = Query(0.0),
    user_id: Optional[str] = Query(None)
):
    """创建检索日志"""
    if not retrieval_logs_service:
        raise HTTPException(status_code=503, detail="Retrieval logs service not available")
    
    try:
        log_id = await retrieval_logs_service.log_retrieval(
            workspace_id=workspace_id,
            query=query,
            top_k=top_k,
            result_count=result_count,
            total_score=total_score,
            search_strategy=search_strategy,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            user_id=user_id
        )
        return {"log_id": log_id}
    except Exception as e:
        logger.error(f"Failed to create retrieval log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieval-logs/{log_id}", response_model=RetrievalLogResponse)
async def get_retrieval_log(log_id: str):
    """获取检索日志"""
    if not retrieval_logs_service:
        raise HTTPException(status_code=503, detail="Retrieval logs service not available")
    
    log = await retrieval_logs_service.get_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    return RetrievalLogResponse(**log)


@app.get("/retrieval-logs", response_model=List[RetrievalLogResponse])
async def list_retrieval_logs(
    workspace_id: str = Query(default=DEFAULT_WORKSPACE),
    days: int = Query(7, ge=1, le=365),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """列出检索日志"""
    if not retrieval_logs_service:
        raise HTTPException(status_code=503, detail="Retrieval logs service not available")
    
    logs = await retrieval_logs_service.list_logs(
        workspace_id=workspace_id,
        days=days,
        limit=limit,
        offset=offset
    )
    return [RetrievalLogResponse(**log) for log in logs]


@app.get("/retrieval-logs/popular", response_model=List[PopularQueryResponse])
async def get_popular_queries(
    workspace_id: str = Query(default=DEFAULT_WORKSPACE),
    days: int = Query(7, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100)
):
    """获取热门查询"""
    if not retrieval_logs_service:
        raise HTTPException(status_code=503, detail="Retrieval logs service not available")
    
    queries = await retrieval_logs_service.get_popular_queries(
        workspace_id=workspace_id,
        days=days,
        limit=limit
    )
    return [PopularQueryResponse(**q) for q in queries]


@app.get("/retrieval-logs/stats", response_model=RetrievalStatsResponse)
async def get_retrieval_stats(
    workspace_id: str = Query(default=DEFAULT_WORKSPACE),
    days: int = Query(7, ge=1, le=365)
):
    """获取检索统计"""
    if not retrieval_logs_service:
        raise HTTPException(status_code=503, detail="Retrieval logs service not available")
    
    stats = await retrieval_logs_service.get_stats(
        workspace_id=workspace_id,
        days=days
    )
    return RetrievalStatsResponse(**stats)


@app.post("/retrieval-logs/cleanup", response_model=CleanupResponse)
async def cleanup_retrieval_logs(retention_days: int = Query(30, ge=1, le=365)):
    """清理旧检索日志"""
    if not retrieval_logs_service:
        raise HTTPException(status_code=503, detail="Retrieval logs service not available")
    
    deleted = await retrieval_logs_service.cleanup_old_logs(retention_days)
    return CleanupResponse(
        deleted_count=deleted,
        message=f"Cleaned up {deleted} old retrieval logs older than {retention_days} days"
    )


@app.post("/api-usage", response_model=Dict[str, str])
async def record_api_usage(
    provider: str = Query(...),
    model: str = Query(...),
    operation: str = Query(...),
    prompt_tokens: int = Query(0),
    completion_tokens: int = Query(0),
    cost_usd: float = Query(0.0),
    workspace_id: Optional[str] = Query(None)
):
    """记录 API 使用"""
    if not api_usage_stats_service:
        raise HTTPException(status_code=503, detail="API usage stats service not available")
    
    try:
        record_id = await api_usage_stats_service.record_usage(
            provider=provider,
            model=model,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            workspace_id=workspace_id
        )
        return {"record_id": record_id}
    except Exception as e:
        logger.error(f"Failed to record API usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api-usage/stats", response_model=APIUsageStatsResponse)
async def get_api_usage_stats(
    workspace_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
):
    """获取 API 使用统计"""
    if not api_usage_stats_service:
        raise HTTPException(status_code=503, detail="API usage stats service not available")
    
    stats = await api_usage_stats_service.get_total_stats(
        workspace_id=workspace_id,
        days=days
    )
    return APIUsageStatsResponse(**stats)


@app.get("/api-usage/daily-cost", response_model=List[DailyCostResponse])
async def get_daily_cost(
    workspace_id: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=365)
):
    """获取每日成本趋势"""
    if not api_usage_stats_service:
        raise HTTPException(status_code=503, detail="API usage stats service not available")
    
    trends = await api_usage_stats_service.get_cost_by_day(
        workspace_id=workspace_id,
        days=days
    )
    return [DailyCostResponse(**t) for t in trends]


@app.post("/api-usage/cleanup", response_model=CleanupResponse)
async def cleanup_api_usage(retention_days: int = Query(90, ge=1, le=365)):
    """清理旧 API 使用统计"""
    if not api_usage_stats_service:
        raise HTTPException(status_code=503, detail="API usage stats service not available")
    
    deleted = await api_usage_stats_service.cleanup_old_stats(retention_days)
    return CleanupResponse(
        deleted_count=deleted,
        message=f"Cleaned up {deleted} old API usage records older than {retention_days} days"
    )


class DocumentChunkCreate(BaseModel):
    """创建文档分块请求"""
    document_id: str
    chunk_index: int
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    embedding: Optional[List[float]] = None


class DocumentChunkResponse(BaseModel):
    """文档分块响应"""
    id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]
    char_count: Optional[int]
    created_at: Optional[str]


class DocumentChunksListResponse(BaseModel):
    """文档分块列表响应"""
    chunks: List[DocumentChunkResponse]
    total_count: int


@app.post("/document-chunks", response_model=Dict[str, str])
async def create_document_chunk(request: DocumentChunkCreate):
    """创建文档分块"""
    if not document_chunks_service:
        raise HTTPException(status_code=503, detail="Document chunks service not available")
    
    try:
        chunk_id = await document_chunks_service.create_chunk(
            document_id=request.document_id,
            chunk_index=request.chunk_index,
            content=request.content,
            start_line=request.start_line,
            end_line=request.end_line,
            embedding=request.embedding
        )
        return {"chunk_id": chunk_id}
    except Exception as e:
        logger.error(f"Failed to create document chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document-chunks", response_model=DocumentChunksListResponse)
async def list_document_chunks(
    document_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """列出文档分块"""
    if not document_chunks_service:
        raise HTTPException(status_code=503, detail="Document chunks service not available")
    
    chunks = await document_chunks_service.list_chunks(
        document_id=document_id,
        limit=limit,
        offset=offset
    )
    total_count = await document_chunks_service.get_chunks_count(document_id)
    
    return DocumentChunksListResponse(
        chunks=[DocumentChunkResponse(**chunk) for chunk in chunks],
        total_count=total_count
    )


@app.get("/document-chunks/stats", response_model=Dict[str, int])
async def get_document_chunks_stats(document_id: Optional[str] = Query(None)):
    """获取文档分块统计"""
    if not document_chunks_service:
        raise HTTPException(status_code=503, detail="Document chunks service not available")
    
    count = await document_chunks_service.get_chunks_count(document_id)
    return {"total_chunks": count}


@app.get("/document-chunks/{chunk_id}", response_model=DocumentChunkResponse)
async def get_document_chunk(chunk_id: str):
    """获取文档分块"""
    if not document_chunks_service:
        raise HTTPException(status_code=503, detail="Document chunks service not available")
    
    chunk = await document_chunks_service.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    return DocumentChunkResponse(**chunk)


@app.delete("/document-chunks/{chunk_id}", response_model=Dict[str, str])
async def delete_document_chunk(chunk_id: str):
    """删除文档分块"""
    if not document_chunks_service:
        raise HTTPException(status_code=503, detail="Document chunks service not available")
    
    deleted = await document_chunks_service.delete_chunk(chunk_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    return {"deleted": chunk_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.presentation.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


rag_etl_service = None
rag_service = None


class IngestRequest(BaseModel):
    """文档摄入请求"""
    file_path: str = Field(..., description="文件路径")
    source_id: Optional[str] = Field(None, description="文档唯一标识")


class IngestResponse(BaseModel):
    """文档摄入响应"""
    status: str = Field(..., description="状态: success, fast_upload, failed")
    source_id: str = Field(..., description="文档 ID")
    chunk_count: int = Field(..., description="分块数量")
    error_msg: Optional[str] = Field(None, description="错误信息")


class QueryRequest(BaseModel):
    """问答查询请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史")


class QueryResponse(BaseModel):
    """问答查询响应"""
    answer: str = Field(..., description="回答内容")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="信息来源")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """文档摄入 - ETL 流水线"""
    global rag_etl_service

    if not rag_etl_service:
        try:
            from src.services.rag.etl_service import create_etl_service
            rag_etl_service = await create_etl_service(workspace_id=DEFAULT_WORKSPACE)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ETL service not available: {str(e)}")

    try:
        result = await rag_etl_service.ingest(
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
    global rag_service

    if not rag_service:
        try:
            from src.services.rag.rag_service import create_rag_service
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


class OptimizationLevel(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    ADVANCED = "advanced"
    FULL = "full"


class EnhancedIngestRequest(BaseModel):
    """增强版文档摄入请求"""
    file_path: str = Field(..., description="文件路径")
    source_id: Optional[str] = Field(None, description="文档唯一标识")
    file_name: Optional[str] = Field(None, description="原始文件名")
    force_reindex: bool = Field(False, description="强制重新索引")
    optimization_level: OptimizationLevel = Field(OptimizationLevel.FULL, description="优化级别")
    embedding_model: str = Field("text-embedding-3-large", description="嵌入模型")
    enable_compression: bool = Field(True, description="是否启用压缩")
    enable_checkpoint: bool = Field(True, description="是否启用检查点")


class EnhancedIngestResponse(BaseModel):
    """增强版文档摄入响应"""
    status: str = Field(..., description="状态")
    source_id: str = Field(..., description="文档ID")
    chunk_count: int = Field(..., description="总块数")
    new_chunks: int = Field(0, description="新增块数")
    updated_chunks: int = Field(0, description="更新块数")
    deleted_chunks: int = Field(0, description="删除块数")
    skipped_chunks: int = Field(0, description="跳过块数")
    compression_ratio: float = Field(1.0, description="压缩比")
    processing_time_ms: float = Field(0.0, description="处理时间(ms)")
    error_msg: Optional[str] = Field(None, description="错误信息")
    checkpoint_id: Optional[str] = Field(None, description="检查点ID")
    consistency_report: Optional[Dict[str, Any]] = Field(None, description="一致性报告")


class ConsistencyCheckRequest(BaseModel):
    """一致性检查请求"""
    source_id: str = Field(..., description="文档ID")


class ConsistencyCheckResponse(BaseModel):
    """一致性检查响应"""
    document_id: str = Field(..., description="文档ID")
    total_chunks: int = Field(..., description="总块数")
    valid_embeddings: int = Field(..., description="有效嵌入数")
    missing_embeddings: int = Field(..., description="缺失嵌入数")
    orphaned_chunks: int = Field(..., description="孤立块数")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="问题列表")
    recommendation: str = Field(..., description="建议")


class RecoveryRequest(BaseModel):
    """恢复请求"""
    batch_id: str = Field(..., description="批次ID")


class RecoveryResponse(BaseModel):
    """恢复响应"""
    status: str = Field(..., description="状态")
    batch_id: str = Field(..., description="批次ID")
    start_index: int = Field(..., description="起始索引")
    remaining_chunks: int = Field(..., description="剩余块数")
    progress: str = Field(..., description="进度")


class OptimizationStatsResponse(BaseModel):
    """优化统计响应"""
    optimization_level: str = Field(..., description="优化级别")
    embedding_model: str = Field(..., description="嵌入模型")
    compression_enabled: bool = Field(..., description="是否启用压缩")
    checkpoint_enabled: bool = Field(..., description="是否启用检查点")
    dimension_manager: Optional[Dict[str, Any]] = Field(None, description="维度管理器统计")
    storage: Optional[Dict[str, Any]] = Field(None, description="存储优化报告")


enhanced_etl_service: Optional[Any] = None


@app.post("/ingest/enhanced", response_model=EnhancedIngestResponse)
async def ingest_document_enhanced(request: EnhancedIngestRequest):
    """增强版文档摄入 - 支持增量更新和存储优化"""
    global enhanced_etl_service

    if not enhanced_etl_service:
        try:
            from src.services.rag.enhanced_etl_service import (
                EnhancedETLService,
                create_enhanced_etl_service,
                OptimizationLevel as ETLOptimizationLevel
            )

            level_map = {
                OptimizationLevel.NONE: ETLOptimizationLevel.NONE,
                OptimizationLevel.STANDARD: ETLOptimizationLevel.STANDARD,
                OptimizationLevel.ADVANCED: ETLOptimizationLevel.ADVANCED,
                OptimizationLevel.FULL: ETLOptimizationLevel.FULL,
            }

            enhanced_etl_service = await create_enhanced_etl_service(
                workspace_id=DEFAULT_WORKSPACE,
                optimization_level=level_map[request.optimization_level],
                embedding_model=request.embedding_model,
                enable_compression=request.enable_compression,
                enable_checkpoint=request.enable_checkpoint
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Enhanced ETL service not available: {str(e)}")

    try:
        result = await enhanced_etl_service.ingest(
            file_path=request.file_path,
            source_id=request.source_id,
            file_name=request.file_name,
            force_reindex=request.force_reindex
        )

        return EnhancedIngestResponse(
            status=result.status,
            source_id=result.source_id,
            chunk_count=result.chunk_count,
            new_chunks=result.new_chunks,
            updated_chunks=result.updated_chunks,
            deleted_chunks=result.deleted_chunks,
            skipped_chunks=result.skipped_chunks,
            compression_ratio=result.compression_ratio,
            processing_time_ms=result.processing_time_ms,
            error_msg=result.error_msg,
            checkpoint_id=result.checkpoint_id,
            consistency_report=result.consistency_report
        )
    except Exception as e:
        logger.error(f"Enhanced ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consistency/check", response_model=ConsistencyCheckResponse)
async def check_consistency(request: ConsistencyCheckRequest):
    """检查数据一致性"""
    global enhanced_etl_service

    if not enhanced_etl_service:
        raise HTTPException(status_code=503, detail="Enhanced ETL service not initialized")

    try:
        result = await enhanced_etl_service.check_consistency(request.source_id)

        if result.get("status") == "not_enabled":
            raise HTTPException(status_code=400, detail=result.get("message"))

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("message"))

        return ConsistencyCheckResponse(
            document_id=result["document_id"],
            total_chunks=result["total_chunks"],
            valid_embeddings=result["valid_embeddings"],
            missing_embeddings=result["missing_embeddings"],
            orphaned_chunks=result["orphaned_chunks"],
            issues=result["issues"],
            recommendation=result["recommendation"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consistency/recover", response_model=RecoveryResponse)
async def recover_from_checkpoint(request: RecoveryRequest):
    """从检查点恢复"""
    global enhanced_etl_service

    if not enhanced_etl_service:
        raise HTTPException(status_code=503, detail="Enhanced ETL service not initialized")

    try:
        result = await enhanced_etl_service.recover_from_checkpoint(request.batch_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("message"))

        return RecoveryResponse(
            status=result["status"],
            batch_id=result["batch_id"],
            start_index=result["start_index"],
            remaining_chunks=result["remaining_chunks"],
            progress=result["progress"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization/stats", response_model=OptimizationStatsResponse)
async def get_optimization_stats():
    """获取优化统计信息"""
    global enhanced_etl_service

    if not enhanced_etl_service:
        raise HTTPException(status_code=503, detail="Enhanced ETL service not initialized")

    try:
        stats = enhanced_etl_service.get_optimization_stats()

        return OptimizationStatsResponse(
            optimization_level=stats["optimization_level"],
            embedding_model=stats["embedding_model"],
            compression_enabled=stats["compression_enabled"],
            checkpoint_enabled=stats["checkpoint_enabled"],
            dimension_manager=stats.get("dimension_manager"),
            storage=stats.get("storage")
        )
    except Exception as e:
        logger.error(f"Get optimization stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ReconciliationResponse(BaseModel):
    """对账响应"""
    scanned_files: int
    fixed_missing_vectors: int
    fixed_orphaned_vectors: int
    fixed_failed_status: int
    duration_seconds: float
    errors: List[str] = []


class ReconciliationStatus(BaseModel):
    """对账状态"""
    status: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    total_fixed: int = 0


@app.post("/admin/reconciliation/run", response_model=ReconciliationResponse)
async def run_reconciliation(background_tasks: BackgroundTasks):
    """手动触发数据对账（Source of Truth 架构）"""
    async def run_job():
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'src.scripts.reconciliation'],
            cwd='/Users/fengyu/Downloads/myproject/workspace/agent-skills-demo',
            capture_output=True,
            text=True
        )
        logger.info(f"Reconciliation output: {result.stdout}")
        if result.stderr:
            logger.error(f"Reconciliation error: {result.stderr}")

    background_tasks.add_task(run_job)

    return {
        "status": "started",
        "message": "对账任务已在后台启动，请稍后查看日志"
    }


@app.get("/admin/reconciliation/status", response_model=ReconciliationStatus)
async def get_reconciliation_status():
    """获取对账状态"""
    return {
        "status": "available",
        "message": "对账脚本已就绪，可通过 POST /admin/reconciliation/run 触发"
    }


@app.post("/admin/reconciliation/cleanup-orphaned")
async def cleanup_orphaned_vectors():
    """清理孤立向量"""
    import subprocess

    result = subprocess.run(
        ['python', '-m', 'src.scripts.reconciliation', '--cleanup-orphaned'],
        cwd='/Users/fengyu/Downloads/myproject/workspace/agent-skills-demo',
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return {
            "status": "success",
            "message": result.stdout.strip()
        }
    else:
        raise HTTPException(status_code=500, detail=result.stderr)
