"""Command Line Interface

This module implements the CLI for the RAG screenplay generation system.

验证需求: 12.8
"""

import asyncio
import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional
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


logger = logging.getLogger(__name__)


class ScreenplayCLI:
    """
    命令行接口
    
    职责：
    1. 解析命令行参数
    2. 加载配置文件
    3. 初始化服务
    4. 执行剧本生成工作流
    5. 输出结果到文件或 stdout
    
    验证需求: 12.8
    """
    
    def __init__(self):
        """初始化 CLI"""
        self.config = None
        self.llm_service = None
        self.retrieval_service = None
        self.parser_service = None
        self.summarization_service = None
        self.orchestrator = None
    
    def parse_args(self) -> argparse.Namespace:
        """
        解析命令行参数
        
        Returns:
            解析后的参数
        """
        parser = argparse.ArgumentParser(
            description="RAG Screenplay Multi-Agent System - Generate screenplays based on code context",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Generate screenplay with topic
  %(prog)s --topic "Explain user authentication" --workspace-id abc123
  
  # Generate with project context
  %(prog)s --topic "Database migration" --context "PostgreSQL to MongoDB" --workspace-id abc123
  
  # Output to file
  %(prog)s --topic "API design" --output screenplay.md --workspace-id abc123
  
  # Use custom config
  %(prog)s --topic "Testing strategy" --config custom_config.yaml --workspace-id abc123
  
  # Set log level
  %(prog)s --topic "CI/CD pipeline" --log-level DEBUG --workspace-id abc123
            """
        )
        
        # Required arguments
        parser.add_argument(
            "--topic",
            type=str,
            required=True,
            help="User topic for screenplay generation (required)"
        )
        
        parser.add_argument(
            "--workspace-id",
            type=str,
            required=True,
            help="Workspace ID for code retrieval (required)"
        )
        
        # Optional arguments
        parser.add_argument(
            "--context",
            type=str,
            default="",
            help="Additional project context (optional)"
        )
        
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output file path (default: stdout)"
        )
        
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            default="config.yaml",
            help="Configuration file path (default: config.yaml)"
        )
        
        parser.add_argument(
            "--env",
            type=str,
            default=".env",
            help="Environment file path (default: .env)"
        )
        
        parser.add_argument(
            "--log-level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Logging level (default: INFO)"
        )
        
        parser.add_argument(
            "--log-file",
            type=str,
            default=None,
            help="Log file path (default: logs/app.log)"
        )
        
        parser.add_argument(
            "--skill",
            type=str,
            default="standard_tutorial",
            choices=[
                "standard_tutorial",
                "warning_mode",
                "visualization_analogy",
                "research_mode",
                "meme_style",
                "fallback_summary"
            ],
            help="Initial skill mode (default: standard_tutorial)"
        )
        
        parser.add_argument(
            "--tone",
            type=str,
            default="professional",
            choices=["professional", "cautionary", "engaging", "exploratory", "casual", "neutral"],
            help="Global tone (default: professional)"
        )
        
        parser.add_argument(
            "--max-retries",
            type=int,
            default=3,
            help="Maximum retries per step (default: 3)"
        )
        
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s 1.0.0"
        )
        
        return parser.parse_args()
    
    def load_config(self, config_path: str) -> dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse configuration file: {e}")
            sys.exit(1)
    
    def load_env(self, env_path: str):
        """
        加载环境变量
        
        Args:
            env_path: 环境文件路径
        """
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Environment variables loaded from {env_path}")
        else:
            logger.warning(f"Environment file not found: {env_path}, using system environment")
    
    async def initialize_services(self, config: dict, workspace_id: str):
        """
        初始化所有服务
        
        Args:
            config: 配置字典
            workspace_id: 工作空间 ID
        """
        logger.info("Initializing services...")
        
        # 初始化 LLM 服务
        self.llm_service = LLMService(config['llm'])
        logger.info("LLM service initialized")
        
        # 初始化数据库服务
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
            logger.error(f"Failed to connect to database: {e}")
            logger.warning("Continuing without database connection (retrieval will be limited)")
            postgres_service = None
        
        # 初始化检索服务
        self.retrieval_service = RetrievalService(
            llm_service=self.llm_service,
            postgres_service=postgres_service,
            config=config['retrieval']
        )
        logger.info("Retrieval service initialized")
        
        # 初始化解析服务
        self.parser_service = TreeSitterParser()
        logger.info("Parser service initialized")
        
        # 初始化摘要服务
        self.summarization_service = SummarizationService(
            llm_service=self.llm_service,
            config=config['retrieval']['summarization']
        )
        logger.info("Summarization service initialized")
        
        # 初始化编排器
        self.orchestrator = WorkflowOrchestrator(
            llm_service=self.llm_service,
            retrieval_service=self.retrieval_service,
            parser_service=self.parser_service,
            summarization_service=self.summarization_service,
            workspace_id=workspace_id
        )
        logger.info("Workflow orchestrator initialized")
    
    async def generate_screenplay(
        self,
        topic: str,
        context: str,
        workspace_id: str,
        skill: str,
        tone: str,
        max_retries: int
    ) -> dict:
        """
        生成剧本
        
        Args:
            topic: 用户主题
            context: 项目上下文
            workspace_id: 工作空间 ID
            skill: 初始 Skill
            tone: 全局语调
            max_retries: 最大重试次数
            
        Returns:
            生成结果字典
        """
        logger.info(f"Starting screenplay generation for topic: {topic}")
        
        # 创建初始状态
        state = SharedState(
            user_topic=topic,
            project_context=context,
            current_skill=skill,
            global_tone=tone,
            max_retries=max_retries
        )
        
        # 执行工作流
        result = await self.orchestrator.execute(state)
        
        return result
    
    def write_output(self, screenplay: str, output_path: Optional[str]):
        """
        写入输出
        
        Args:
            screenplay: 生成的剧本
            output_path: 输出文件路径（None 表示 stdout）
        """
        if output_path:
            try:
                # 确保输出目录存在
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # 写入文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(screenplay)
                
                logger.info(f"Screenplay written to {output_path}")
                print(f"\n✓ Screenplay successfully generated and saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to write output file: {e}")
                print(f"\n✗ Error writing output file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # 输出到 stdout
            print("\n" + "="*80)
            print("GENERATED SCREENPLAY")
            print("="*80 + "\n")
            print(screenplay)
            print("\n" + "="*80)
    
    async def run(self):
        """
        运行 CLI 主流程
        """
        # 解析参数
        args = self.parse_args()
        
        # 加载环境变量
        self.load_env(args.env)
        
        # 设置日志
        log_file = args.log_file or os.getenv('LOG_FILE', 'logs/app.log')
        setup_logging(level=args.log_level, log_file=log_file)
        
        logger.info("="*80)
        logger.info("RAG Screenplay Multi-Agent System - CLI")
        logger.info("="*80)
        
        # 加载配置
        self.config = self.load_config(args.config)
        
        # 初始化服务
        await self.initialize_services(self.config, args.workspace_id)
        
        # 生成剧本
        print(f"\n🎬 Generating screenplay for topic: {args.topic}")
        print(f"📁 Workspace ID: {args.workspace_id}")
        print(f"🎨 Skill: {args.skill}")
        print(f"🎭 Tone: {args.tone}")
        print(f"⚙️  Max retries: {args.max_retries}")
        print("\nProcessing...\n")
        
        result = await self.generate_screenplay(
            topic=args.topic,
            context=args.context,
            workspace_id=args.workspace_id,
            skill=args.skill,
            tone=args.tone,
            max_retries=args.max_retries
        )
        
        # 处理结果
        if result['success']:
            screenplay = result.get('final_screenplay', '')
            
            if screenplay:
                self.write_output(screenplay, args.output)
                
                # 显示统计信息
                state = result['state']
                print(f"\n📊 Statistics:")
                print(f"   - Total steps: {len(state.outline)}")
                print(f"   - Fragments generated: {len(state.fragments)}")
                print(f"   - Documents retrieved: {len(state.retrieved_docs)}")
                print(f"   - Pivots triggered: {sum(1 for log in state.execution_log if log.get('action') == 'pivot_triggered')}")
                
                logger.info("Screenplay generation completed successfully")
            else:
                print("\n✗ Error: No screenplay generated", file=sys.stderr)
                logger.error("No screenplay generated")
                sys.exit(1)
        else:
            error = result.get('error', 'Unknown error')
            print(f"\n✗ Error: {error}", file=sys.stderr)
            logger.error(f"Screenplay generation failed: {error}")
            sys.exit(1)


def main():
    """CLI 入口点"""
    cli = ScreenplayCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
