# Setup Guide

## Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- PostgreSQL 17 with pgvector extension
- Redis (optional, for caching)
- Git

### 2. Installation

#### Option A: Using Poetry (Recommended)

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option C: Using Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f app
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and settings
nano .env  # or use your preferred editor
```

Required configuration:
- At least one LLM provider API key (OpenAI, Qwen, MiniMax, or GLM)
- PostgreSQL connection details
- Redis connection details (optional)

### 4. Database Setup

```bash
# Create database and enable extensions
psql -U postgres -c "CREATE DATABASE screenplay_system;"
psql -U postgres -d screenplay_system -c "CREATE EXTENSION vector;"

# Run initialization script
psql -U postgres -d screenplay_system -f scripts/init_db.sql
```

### 5. Verify Installation

```bash
# Run tests
make test

# Or with pytest directly
pytest tests/ -v
```

## Development Setup

### Code Quality Tools

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy src/
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Property-based tests
make test-property

# Integration tests
make test-integration

# With coverage report
make test-cov
```

## Project Structure

```
rag-screenplay-multi-agent/
├── src/
│   ├── presentation/      # CLI, API interfaces
│   ├── application/       # Workflow orchestration
│   ├── domain/           # Agents, models, skills
│   │   └── agents/       # Six specialized agents
│   ├── services/         # LLM, database, parser services
│   │   ├── llm/         # Multi-provider LLM adapters
│   │   ├── database/    # PostgreSQL, Redis services
│   │   └── parser/      # Tree-sitter code parsing
│   └── infrastructure/   # Logging, monitoring
├── tests/
│   ├── unit/            # Unit tests
│   ├── property/        # Property-based tests (Hypothesis)
│   └── integration/     # Integration tests
├── scripts/             # Database and utility scripts
├── config.yaml          # Application configuration
├── .env                 # Environment variables (create from .env.example)
└── docker-compose.yml   # Docker deployment configuration
```

## Next Steps

After setup is complete, you can:

1. Review the requirements document: `.kiro/specs/rag-screenplay-multi-agent/requirements.md`
2. Check the design document: `.kiro/specs/rag-screenplay-multi-agent/design.md`
3. Follow the implementation tasks: `.kiro/specs/rag-screenplay-multi-agent/tasks.md`

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify pgvector extension
psql -U postgres -d screenplay_system -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Python Package Issues

```bash
# Clear cache and reinstall
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

### Docker Issues

```bash
# Rebuild containers
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## Support

For issues and questions, please refer to:
- Requirements: `.kiro/specs/rag-screenplay-multi-agent/requirements.md`
- Design: `.kiro/specs/rag-screenplay-multi-agent/design.md`
- Tasks: `.kiro/specs/rag-screenplay-multi-agent/tasks.md`
