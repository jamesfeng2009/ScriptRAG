# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with layered architecture
- Core dependencies configuration (Poetry and pip)
- Multi-LLM provider support (OpenAI, Qwen, MiniMax, GLM)
- Database integration setup (PostgreSQL + pgvector, Redis)
- Tree-sitter code parsing setup
- Comprehensive logging configuration
- Docker deployment configuration
- Test framework setup (pytest, Hypothesis)
- Development tools (black, ruff, mypy)
- Documentation (README, SETUP, PROJECT_STRUCTURE)
- Utility scripts (verify_setup.py, quick_start.sh)

### Directory Structure
- `src/presentation/`: CLI and API interfaces (placeholders)
- `src/application/`: Workflow orchestration (placeholders)
- `src/domain/`: Agents, models, skills (placeholders)
- `src/services/`: LLM, database, parser services (placeholders)
- `src/infrastructure/`: Logging, monitoring, error handling
- `tests/`: Unit, property, and integration test directories

## [0.1.0] - 2024-01-29

### Initial Release
- Project scaffolding complete
- Ready for implementation of core features
- All dependencies configured
- Development environment ready

---

## Implementation Progress

Track implementation progress in `.kiro/specs/rag-screenplay-multi-agent/tasks.md`

### Completed Tasks
- [x] Task 1: Project structure and core dependencies

### Pending Tasks
- [ ] Task 2: Core data models
- [ ] Task 3: Skills system
- [ ] Task 4: Service layer abstractions
- [ ] Task 5: Database schema
- [ ] Task 6-26: Agent implementations and features
