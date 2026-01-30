# Monitoring Setup

This directory contains the monitoring configuration for the RAG Screenplay Multi-Agent System.

## Components

### Prometheus
- **Port**: 9090
- **Purpose**: Metrics collection and storage
- **Configuration**: `prometheus.yml`
- **Alert Rules**: `alerts.yml`

### Grafana
- **Port**: 3000
- **Purpose**: Metrics visualization and dashboards
- **Default Credentials**: admin/admin
- **Dashboard**: `grafana-dashboard.json`

### Alertmanager
- **Port**: 9093
- **Purpose**: Alert routing and notification
- **Configuration**: `alertmanager.yml`

### Exporters
- **Node Exporter** (Port 9100): System metrics
- **PostgreSQL Exporter** (Port 9187): Database metrics
- **Redis Exporter** (Port 9121): Cache metrics

## Quick Start

### 1. Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### 2. Access Services

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

### 3. Import Dashboard

The Grafana dashboard is automatically provisioned from `grafana-dashboard.json`.

Alternatively, manually import:
1. Open Grafana (http://localhost:3000)
2. Go to Dashboards → Import
3. Upload `grafana-dashboard.json`

### 4. Configure Data Source

1. Go to Configuration → Data Sources
2. Add Prometheus data source
3. URL: `http://prometheus:9090`
4. Save & Test

## Metrics

### Workflow Metrics
- `workflow_executions_total`: Total workflow executions by status
- `workflow_duration_seconds`: Workflow execution duration
- `workflow_steps_total`: Number of steps per workflow
- `workflow_pivots_total`: Number of pivots triggered
- `workflow_retries_total`: Number of retries

### Agent Metrics
- `agent_executions_total`: Agent executions by name and status
- `agent_execution_duration_seconds`: Agent execution duration

### LLM Metrics
- `llm_calls_total`: LLM calls by provider, model, and status
- `llm_call_duration_seconds`: LLM call duration
- `llm_tokens_total`: Token usage by provider and type

### Retrieval Metrics
- `retrieval_operations_total`: Retrieval operations by type and status
- `retrieval_duration_seconds`: Retrieval operation duration
- `documents_retrieved_total`: Documents retrieved by workspace

### System Metrics
- `active_tasks`: Number of active tasks
- `task_queue_size`: Task queue size
- `errors_total`: Errors by component and type

## Alerts

### Critical Alerts
- **ServiceDown**: Service is unreachable
- **HighWorkflowFailureRate**: >20% workflow failure rate

### Warning Alerts
- **HighErrorRate**: >0.1 errors/sec
- **HighLLMFailureRate**: >10% LLM call failure rate
- **SlowWorkflowExecution**: P95 duration >300s
- **SlowLLMCalls**: P95 duration >30s
- **TooManyActiveTasks**: >100 active tasks
- **HighRetrievalFailureRate**: >20% retrieval failure rate

### Info Alerts
- **HighTokenUsage**: >1M tokens/hour
- **ExcessivePivots**: >0.5 pivots/sec

## Alert Configuration

### Email Notifications

Edit `alertmanager.yml`:

```yaml
email_configs:
  - to: 'your-email@example.com'
    from: 'alertmanager@example.com'
    smarthost: 'smtp.example.com:587'
    auth_username: 'alertmanager@example.com'
    auth_password: 'your-password'
```

### Slack Notifications

Edit `alertmanager.yml`:

```yaml
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#alerts'
```

## Custom Metrics

To add custom metrics in your code:

```python
from infrastructure.metrics import (
    record_agent_execution,
    record_llm_call,
    record_retrieval_operation,
    record_workflow_execution,
    record_error
)

# Record agent execution
record_agent_execution(
    agent_name="planner",
    duration=2.5,
    status="success"
)

# Record LLM call
record_llm_call(
    provider="openai",
    model="gpt-4o",
    task_type="high_performance",
    duration=1.2,
    status="success",
    tokens={"prompt_tokens": 100, "completion_tokens": 50}
)

# Record retrieval operation
record_retrieval_operation(
    operation_type="hybrid",
    duration=0.5,
    status="success",
    workspace_id="abc123",
    documents_count=5
)

# Record workflow execution
record_workflow_execution(
    duration=120.0,
    status="success",
    steps_count=7,
    pivots_count=2,
    retries_count=1
)

# Record error
record_error(
    component="navigator",
    error_type="retrieval_failed"
)
```

## Troubleshooting

### Prometheus Not Scraping Metrics

1. Check if API is exposing metrics: `curl http://localhost:8000/metrics`
2. Check Prometheus targets: http://localhost:9090/targets
3. Verify `prometheus.yml` configuration

### Grafana Dashboard Not Loading

1. Check Prometheus data source connection
2. Verify dashboard JSON is valid
3. Check Grafana logs: `docker logs screenplay-grafana`

### Alerts Not Firing

1. Check alert rules: http://localhost:9090/alerts
2. Verify Alertmanager configuration
3. Check Alertmanager logs: `docker logs screenplay-alertmanager`

## Production Considerations

### Security
- Enable authentication for Prometheus and Grafana
- Use HTTPS for all services
- Restrict network access with firewall rules
- Rotate credentials regularly

### Scalability
- Use Prometheus federation for multi-cluster setup
- Configure retention policies for metrics
- Use remote storage for long-term retention
- Scale Alertmanager with clustering

### High Availability
- Run multiple Prometheus instances
- Use Thanos or Cortex for HA setup
- Configure Alertmanager clustering
- Use external storage for Grafana

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
