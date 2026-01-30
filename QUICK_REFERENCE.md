# Quick Reference Guide

## Common Commands

### Setup
```bash
# Quick start (automated)
./scripts/quick_start.sh

# Manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Development
```bash
# Run tests
make test                 # All tests
make test-unit           # Unit tests only
make test-property       # Property-based tests
make test-integration    # Integration tests
make test-cov            # With coverage report

# Code quality
make format              # Format code with black
make lint                # Run linters (ruff, mypy)

# Clean up
make clean               # Remove build artifacts
```

### Docker
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Rebuild
docker-compose build --no-cache
```

### Database
```bash
# Initialize database
psql -U postgres -c "CREATE DATABASE screenplay_system;"
psql -U postgres -d screenplay_system -f scripts/init_db.sql

# Check connection
psql -U postgres -d screenplay_system -c "SELECT version();"
```

## Project Structure Quick View

```
src/
├── presentation/    # CLI, API
├── application/     # Orchestration
├── domain/         # Agents, Models
├── services/       # LLM, DB, Parser
└── infrastructure/ # Logging, Monitoring
```

## Key Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API keys, database credentials |
| `config.yaml` | Application settings, Skills, models |
| `pyproject.toml` | Python dependencies (Poetry) |
| `requirements.txt` | Python dependencies (pip) |
| `docker-compose.yml` | Docker deployment |

## Environment Variables

### Required
```bash
# At least one LLM provider
OPENAI_API_KEY=sk-xxx
# OR
QWEN_API_KEY=sk-xxx
# OR
MINIMAX_API_KEY=xxx
# OR
GLM_API_KEY=xxx

# Database
POSTGRES_HOST=localhost
POSTGRES_DB=screenplay_system
POSTGRES_USER=postgres
POSTGRES_PASSWORD=xxx
```

### Optional
```bash
REDIS_HOST=localhost
LOG_LEVEL=INFO
MAX_RETRIES=3
```

## Six Agents

1. **Planner**: Generates outline
2. **Navigator**: RAG retrieval
3. **Director**: Makes decisions
4. **Pivot Manager**: Corrects conflicts
5. **Writer**: Generates fragments
6. **Compiler**: Integrates final screenplay

## Six Skills

1. **standard_tutorial**: Professional tutorial format
2. **warning_mode**: Highlights deprecated/risky content
3. **visualization_analogy**: Uses analogies for complex concepts
4. **research_mode**: Acknowledges information gaps
5. **meme_style**: Light-hearted, humorous
6. **fallback_summary**: High-level overview

## Testing

### Run Specific Test
```bash
pytest tests/unit/test_specific.py -v
pytest tests/unit/test_specific.py::test_function -v
```

### Property-Based Testing
```bash
# Run with more examples
pytest tests/property/ --hypothesis-show-statistics
```

### Coverage
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Troubleshooting

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Database Connection
```bash
# Test PostgreSQL connection
psql -U postgres -h localhost -p 5432 -d screenplay_system
```

### Redis Connection
```bash
# Test Redis connection
redis-cli ping
```

### Verify Setup
```bash
python scripts/verify_setup.py
```

## Documentation

- **README.md**: Project overview
- **SETUP.md**: Detailed setup instructions
- **PROJECT_STRUCTURE.md**: Complete structure documentation
- **CHANGELOG.md**: Version history
- **.kiro/specs/**: Requirements, design, tasks

## Next Steps

1. ✓ Complete Task 1: Project structure (DONE)
2. → Task 2: Implement core data models
3. → Task 3: Implement Skills system
4. → Task 4-26: Continue with remaining tasks

See `.kiro/specs/rag-screenplay-multi-agent/tasks.md` for full task list.

## Support

For detailed information:
- Setup: `SETUP.md`
- Structure: `PROJECT_STRUCTURE.md`
- Requirements: `.kiro/specs/rag-screenplay-multi-agent/requirements.md`
- Design: `.kiro/specs/rag-screenplay-multi-agent/design.md`
- Tasks: `.kiro/specs/rag-screenplay-multi-agent/tasks.md`
