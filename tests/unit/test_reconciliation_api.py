"""
Reconciliation API 端点测试

测试覆盖:
- POST /admin/reconciliation/run - 手动触发对账
- GET /admin/reconciliation/status - 获取对账状态
- POST /admin/reconciliation/cleanup-orphaned - 清理孤立向量
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ReconciliationResponse(BaseModel):
    """对账响应"""
    status: str
    message: str


class ReconciliationStatus(BaseModel):
    """对账状态"""
    status: str
    message: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    total_fixed: int = 0


def create_test_app():
    """创建测试用的 FastAPI 应用"""
    from fastapi import FastAPI
    import logging

    logger = logging.getLogger(__name__)

    app = FastAPI()

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

    return app


@pytest.fixture
def client():
    """创建测试客户端"""
    app = create_test_app()
    return TestClient(app)


class TestReconciliationAPIEndpoints:
    """对账 API 端点测试"""

    def test_get_status_endpoint(self, client):
        """测试获取对账状态端点"""
        response = client.get("/admin/reconciliation/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "available"
        assert "message" in data

    def test_get_status_response_model(self, client):
        """测试状态端点响应模型"""
        response = client.get("/admin/reconciliation/status")
        data = response.json()

        assert "status" in data
        assert "message" in data
        assert "last_run" in data
        assert "next_run" in data
        assert "total_fixed" in data

    def test_run_reconciliation_endpoint(self, client):
        """测试触发对账端点"""
        response = client.post("/admin/reconciliation/run")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "message" in data

    def test_run_reconciliation_response_model(self, client):
        """测试触发对账响应模型"""
        response = client.post("/admin/reconciliation/run")
        data = response.json()

        assert "status" in data
        assert "message" in data
        assert data["status"] == "started"

    @patch('subprocess.run')
    def test_cleanup_orphaned_endpoint_success(self, mock_run, client):
        """测试清理孤立向量成功"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='Cleaned 10 orphaned vectors',
            stderr=''
        )

        response = client.post("/admin/reconciliation/cleanup-orphaned")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch('subprocess.run')
    def test_cleanup_orphaned_endpoint_failure(self, mock_run, client):
        """测试清理孤立向量失败"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr='Database connection failed'
        )

        response = client.post("/admin/reconciliation/cleanup-orphaned")

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    @patch('subprocess.run')
    def test_cleanup_orphaned_calls_correct_script(self, mock_run, client):
        """测试清理孤立向量调用正确脚本"""
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')

        client.post("/admin/reconciliation/cleanup-orphaned")

        mock_run.assert_called_once()
        call_args = mock_run.call_args

        assert '--cleanup-orphaned' in call_args[0][0]
        assert 'python' in call_args[0][0][0]


class TestReconciliationAPIResponseValidation:
    """响应验证测试"""

    def test_reconciliation_response_validation(self):
        """测试响应模型验证"""
        response_data = {
            "status": "started",
            "message": "对账任务已在后台启动，请稍后查看日志"
        }

        response = ReconciliationResponse(**response_data)
        assert response.status == "started"
        assert "后台" in response.message

    def test_reconciliation_status_validation(self):
        """测试状态响应模型验证"""
        status_data = {
            "status": "available",
            "message": "对账脚本已就绪",
            "last_run": "2024-01-15T10:00:00",
            "next_run": "2024-01-16T02:00:00",
            "total_fixed": 100
        }

        status = ReconciliationStatus(**status_data)
        assert status.status == "available"
        assert status.total_fixed == 100

    def test_reconciliation_status_optional_fields(self):
        """测试状态响应可选字段"""
        status_data = {
            "status": "unavailable",
            "message": "对账脚本未就绪"
        }

        status = ReconciliationStatus(**status_data)
        assert status.status == "unavailable"
        assert status.last_run is None
        assert status.next_run is None
        assert status.total_fixed == 0


class TestReconciliationAPIErrorHandling:
    """错误处理测试"""

    def test_run_endpoint_returns_json(self, client):
        """测试触发端点返回 JSON"""
        response = client.post("/admin/reconciliation/run")

        assert response.headers.get("content-type") == "application/json"

    def test_status_endpoint_returns_json(self, client):
        """测试状态端点返回 JSON"""
        response = client.get("/admin/reconciliation/status")

        assert response.headers.get("content-type") == "application/json"

    @patch('subprocess.run')
    def test_cleanup_endpoint_handles_empty_output(self, mock_run, client):
        """测试清理端点处理空输出"""
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')

        response = client.post("/admin/reconciliation/cleanup-orphaned")

        assert response.status_code == 200

    @patch('subprocess.run')
    def test_cleanup_endpoint_handles_special_chars(self, mock_run, client):
        """测试清理端点处理特殊字符"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='Cleaned 5 orphaned vectors (包含中文和特殊字符!)',
            stderr=''
        )

        response = client.post("/admin/reconciliation/cleanup-orphaned")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
