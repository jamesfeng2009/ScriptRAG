# Project Structure

## Overview

This document describes the complete structure of the RAG Screenplay Multi-Agent System project.

## Directory Layout

```
rag-screenplay-multi-agent/
├── .kiro/                          # Kiro specifications
│   └── specs/
│       └── rag-screenplay-multi-agent/
│           ├── requirements.md     # System requirements
│           ├── design.md          # Design document
│           └── tasks.md           # Implementation tasks
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── presentation/              # Presentation Layer
│   │   ├── __init__.py
│   │   └── api.py                # REST API interface
│   │
│   ├── application/               # Application Layer
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # LangGraph workflow orchestration
│   │   └── coordinator.py        # Agent lifecycle management
│   │
│   ├── domain/                    # Domain Layer
│   │   ├── __init__.py
│   │   ├── models.py             # Pydantic data models
│   │   ├── skills.py             # Skill manager
│   │   └── agents/               # Agent implementations
│   │       ├── __init__.py
│   │       ├── planner.py        # Planner agent
│   │       ├── navigator.py      # Navigator agent (RAG)
│   │       ├── director.py       # Director agent (decision maker)
│   │       ├── pivot_manager.py  # Pivot manager agent
│   │       ├── writer.py         # Writer agent
│   │       └── compiler.py       # Compiler agent
│   │
│   ├── services/                  # Service Layer
│   │   ├── __init__.py
│   │   ├── llm/                  # LLM services
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py        # Abstract LLM adapter
│   │   │   ├── openai_adapter.py # OpenAI implementation
│   │   │   ├── qwen_adapter.py   # Qwen implementation
│   │   │   ├── minimax_adapter.py# MiniMax implementation
│   │   │   ├── glm_adapter.py    # GLM implementation
│   │   │   └── service.py        # Unified LLM service
│   │   │
│   │   ├── database/             # Database services
│   │   │   ├── __init__.py
│   │   │   ├── vector_db.py      # Vector database (pgvector)
│   │   │   ├── postgres.py       # PostgreSQL operations
│   │   │   └── redis_cache.py    # Redis caching
│   │   │
│   │   └── parser/               # Code parsing services
│   │       ├── __init__.py
│   │       └── tree_sitter_parser.py  # Tree-sitter integration
│   │
│   └── infrastructure/            # Infrastructure Layer
│       ├── __init__.py
│       ├── logging.py            # Logging configuration
│       ├── monitoring.py         # Performance monitoring
│       └── error_handler.py      # Error handling
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest configuration
│   ├── unit/                     # Unit tests
│   │   └── __init__.py
│   ├── property/                 # Property-based tests (Hypothesis)
│   │   └── __init__.py
│   └── integration/              # Integration tests
│       └── __init__.py
│
├── scripts/                       # Utility scripts
│   ├── init_db.sql               # Database initialization
│   ├── verify_setup.py           # Setup verification
│   └── quick_start.sh            # Quick start script
│
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── config.yaml                    # Application configuration
├── docker-compose.yml             # Docker deployment
├── Dockerfile                     # Docker image definition
├── Makefile                       # Common tasks
├── pyproject.toml                 # Poetry configuration
├── requirements.txt               # Python dependencies
├── setup.py                       # Setup script
├── README.md                      # Project overview
├── SETUP.md                       # Setup instructions
└── PROJECT_STRUCTURE.md           # This file
```

## Layer Responsibilities

### Presentation Layer (`src/presentation/`)
- **Purpose**: User interfaces and external APIs
- **Components**:
  - CLI: Command-line interface for running the system
  - API: REST API for programmatic access
- **Dependencies**: Application layer

### Application Layer (`src/application/`)
- **Purpose**: Workflow orchestration and coordination
- **Components**:
  - Orchestrator: LangGraph state machine management
  - Coordinator: Agent lifecycle and execution management
- **Dependencies**: Domain layer, Service layer

### Domain Layer (`src/domain/`)
- **Purpose**: Core business logic and agents
- **Components**:
  - Models: Pydantic data models (SharedState, etc.)
  - Skills: Generation style management
  - Agents: Six specialized agents for screenplay generation
- **Dependencies**: Service layer (through dependency injection)

### Service Layer (`src/services/`)
- **Purpose**: External service integrations
- **Components**:
  - LLM: Multi-provider LLM adapters (OpenAI, Qwen, MiniMax, GLM)
  - Database: PostgreSQL, pgvector, Redis
  - Parser: Tree-sitter code parsing
- **Dependencies**: Infrastructure layer

### Infrastructure Layer (`src/infrastructure/`)
- **Purpose**: Cross-cutting concerns
- **Components**:
  - Logging: Structured logging
  - Monitoring: Performance metrics
  - Error Handler: Error handling and recovery
- **Dependencies**: None (foundation layer)

## Key Files

### Configuration Files

- **pyproject.toml**: Poetry project configuration and dependencies
- **requirements.txt**: Pip-compatible dependency list
- **config.yaml**: Application settings (Skills, retrieval, LLM models)
- **.env**: Environment variables (API keys, database credentials)
- **.env.example**: Template for environment variables

### Docker Files

- **Dockerfile**: Application container definition
- **docker-compose.yml**: Multi-container deployment (app, PostgreSQL, Redis)

### Development Files

- **Makefile**: Common development tasks (test, lint, format)
- **setup.py**: Python package setup
- **.gitignore**: Git ignore patterns

### Documentation

- **README.md**: Project overview and quick start
- **SETUP.md**: Detailed setup instructions
- **PROJECT_STRUCTURE.md**: This file

## Agent Architecture

The system uses six specialized agents:

1. **Planner**: Generates initial screenplay outline
2. **Navigator**: Performs RAG retrieval (hybrid vector + keyword search)
3. **Director**: Evaluates content and makes decisions
4. **Pivot Manager**: Corrects outline when conflicts occur
5. **Writer**: Generates screenplay fragments using active Skill
6. **Compiler**: Integrates fragments into final screenplay

## Data Flow

```
User Input → Planner → Outline
           ↓
Loop for each step:
  Outline → Navigator → Retrieved Content
         → Director → Decision
         → [Pivot Manager] (if conflict)
         → Writer → Fragment
         → Fact Checker → Validated Fragment
           ↓
All Fragments → Compiler → Final Screenplay
```

## Testing Strategy

- **Unit Tests** (`tests/unit/`): Test individual components
- **Property Tests** (`tests/property/`): Test universal properties with Hypothesis
- **Integration Tests** (`tests/integration/`): Test complete workflows

## Development Workflow

1. **Setup**: Run `scripts/quick_start.sh` or follow `SETUP.md`
2. **Configure**: Edit `.env` with API keys
3. **Develop**: Implement tasks from `.kiro/specs/rag-screenplay-multi-agent/tasks.md`
4. **Test**: Run `make test` or `pytest`
5. **Format**: Run `make format`
6. **Lint**: Run `make lint`

## Deployment

### Local Development
```bash
python -m src.presentation.cli
```

### Docker
```bash
docker-compose up -d
```

### Production
- Use environment-specific `.env` files
- Configure PostgreSQL with proper security
- Set up Redis for caching
- Enable monitoring and logging
- Use reverse proxy (nginx) for API

## Next Steps

1. Review requirements: `.kiro/specs/rag-screenplay-multi-agent/requirements.md`
2. Review design: `.kiro/specs/rag-screenplay-multi-agent/design.md`
3. Follow implementation tasks: `.kiro/specs/rag-screenplay-multi-agent/tasks.md`
