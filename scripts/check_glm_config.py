#!/usr/bin/env python3
"""
GLM 配置检查脚本

用途：验证 GLM API 配置是否正确设置
使用：python scripts/check_glm_config.py
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv


def print_header(text):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_success(text):
    """打印成功信息"""
    print(f"✓ {text}")


def print_error(text):
    """打印错误信息"""
    print(f"✗ {text}")


def print_warning(text):
    """打印警告信息"""
    print(f"⚠ {text}")


def print_info(text):
    """打印信息"""
    print(f"ℹ {text}")


def check_env_file():
    """检查 .env 文件"""
    print_header("1. 检查 .env 文件")
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print_warning(".env 文件不存在")
        print_info("创建 .env 文件: cp .env.example .env")
        return False
    
    print_success(".env 文件存在")
    
    # 加载环境变量
    load_dotenv()
    
    return True


def check_glm_api_key():
    """检查 GLM API 密钥"""
    print_header("2. 检查 GLM API 密钥")
    
    api_key = os.getenv("GLM_API_KEY")
    
    if not api_key:
        print_error("GLM_API_KEY 未设置")
        print_info("请在 .env 文件中添加: GLM_API_KEY=your-glm-api-key")
        return False
    
    if api_key.startswith("your-"):
        print_error("GLM_API_KEY 仍为示例值")
        print_info("请替换为实际的 API 密钥")
        return False
    
    # 隐藏大部分密钥用于安全显示
    masked_key = api_key[:10] + "..." + api_key[-5:]
    print_success(f"GLM_API_KEY 已设置: {masked_key}")
    
    return True


def check_glm_base_url():
    """检查 GLM Base URL"""
    print_header("3. 检查 GLM Base URL")
    
    base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    
    if not base_url:
        print_warning("GLM_BASE_URL 未设置，使用默认值")
        base_url = "https://open.bigmodel.cn/api/paas/v4"
    
    print_success(f"GLM_BASE_URL: {base_url}")
    
    return True


def check_active_provider():
    """检查活跃提供商"""
    print_header("4. 检查活跃提供商配置")
    
    active_provider = os.getenv("ACTIVE_LLM_PROVIDER", "openai")
    fallback_providers = os.getenv("FALLBACK_LLM_PROVIDERS", "qwen,glm")
    
    print_info(f"活跃提供商: {active_provider}")
    print_info(f"回退提供商: {fallback_providers}")
    
    if active_provider == "glm":
        print_success("GLM 已设置为活跃提供商")
    else:
        print_info(f"当前使用 {active_provider} 作为主提供商")
        if "glm" in fallback_providers:
            print_success("GLM 已配置为回退提供商")
        else:
            print_warning("GLM 未配置为回退提供商")
    
    return True


def check_config_yaml():
    """检查 config.yaml 文件"""
    print_header("5. 检查 config.yaml 文件")
    
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        print_warning("config.yaml 文件不存在")
        return False
    
    print_success("config.yaml 文件存在")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config or 'llm' not in config:
            print_warning("config.yaml 中未找到 llm 配置")
            return False
        
        llm_config = config['llm']
        
        if 'providers' in llm_config and 'glm' in llm_config['providers']:
            print_success("config.yaml 中已配置 GLM 提供商")
            
            glm_config = llm_config['providers']['glm']
            if 'models' in glm_config:
                print_info(f"  - 高性能模型: {glm_config['models'].get('high_performance', 'N/A')}")
                print_info(f"  - 轻量级模型: {glm_config['models'].get('lightweight', 'N/A')}")
                print_info(f"  - 嵌入模型: {glm_config['models'].get('embedding', 'N/A')}")
        else:
            print_warning("config.yaml 中未配置 GLM 提供商")
        
        return True
        
    except Exception as e:
        print_error(f"读取 config.yaml 失败: {e}")
        return False


def check_llm_providers_yaml():
    """检查 config/llm_providers.yaml 文件"""
    print_header("6. 检查 config/llm_providers.yaml 文件")
    
    config_path = Path("config/llm_providers.yaml")
    example_path = Path("config/llm_providers.example.yaml")
    
    if not config_path.exists():
        if example_path.exists():
            print_warning("config/llm_providers.yaml 不存在")
            print_info("创建文件: cp config/llm_providers.example.yaml config/llm_providers.yaml")
        else:
            print_warning("config/llm_providers.yaml 和示例文件都不存在")
        return False
    
    print_success("config/llm_providers.yaml 文件存在")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config or 'providers' not in config:
            print_warning("config/llm_providers.yaml 中未找到 providers 配置")
            return False
        
        if 'glm' in config['providers']:
            print_success("config/llm_providers.yaml 中已配置 GLM")
            
            glm_config = config['providers']['glm']
            api_key = glm_config.get('api_key', '')
            
            if api_key.startswith("your-"):
                print_warning("GLM API 密钥仍为示例值")
            else:
                masked_key = api_key[:10] + "..." + api_key[-5:]
                print_success(f"GLM API 密钥已设置: {masked_key}")
        else:
            print_warning("config/llm_providers.yaml 中未配置 GLM")
        
        return True
        
    except Exception as e:
        print_error(f"读取 config/llm_providers.yaml 失败: {e}")
        return False


def check_python_imports():
    """检查 Python 依赖"""
    print_header("7. 检查 Python 依赖")
    
    required_packages = [
        ('openai', 'OpenAI SDK'),
        ('pydantic', 'Pydantic'),
        ('yaml', 'PyYAML'),
        ('dotenv', 'python-dotenv'),
    ]
    
    all_ok = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print_success(f"{name} 已安装")
        except ImportError:
            print_error(f"{name} 未安装")
            all_ok = False
    
    return all_ok


def print_summary(results):
    """打印总结"""
    print_header("检查总结")
    
    total = len(results)
    passed = sum(1 for r in results if r)
    failed = total - passed
    
    print(f"总检查项: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print_success("所有检查都通过了！")
        print_info("你可以开始使用 GLM 了")
        return True
    else:
        print_error(f"有 {failed} 项检查失败")
        print_info("请根据上面的提示修复问题")
        return False


def print_next_steps():
    """打印后续步骤"""
    print_header("后续步骤")
    
    print("1. 启动应用:")
    print("   python -m src.presentation.api")
    print()
    print("2. 访问 API:")
    print("   http://localhost:8000")
    print()
    print("3. 查看 API 文档:")
    print("   http://localhost:8000/docs")
    print()
    print("4. 生成剧本:")
    print("   curl -X POST http://localhost:8000/generate \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -H 'X-API-Key: your-secret-api-key' \\")
    print("     -d '{\"topic\": \"Test\", \"workspace_id\": \"test\"}'")
    print()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  GLM 配置检查工具")
    print("="*60)
    
    results = []
    
    # 执行所有检查
    results.append(check_env_file())
    results.append(check_glm_api_key())
    results.append(check_glm_base_url())
    results.append(check_active_provider())
    results.append(check_config_yaml())
    results.append(check_llm_providers_yaml())
    results.append(check_python_imports())
    
    # 打印总结
    success = print_summary(results)
    
    # 打印后续步骤
    if success:
        print_next_steps()
    
    # 返回退出码
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
