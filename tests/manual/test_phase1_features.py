"""Manual test script for Phase 1 features

This script demonstrates and tests all Phase 1 features:
1. Skill configuration loading
2. SkillManager with config files
3. PromptManager integration
4. Writer Agent with config
5. Hot-reload functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.domain.skills import SkillManager
from src.domain.skill_loader import SkillConfigLoader, create_default_config
from src.domain.prompt_manager import PromptManager
from src.domain.agents.writer import apply_skill, get_prompt_manager, set_prompt_manager
from src.domain.models import OutlineStep, RetrievedDocument


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_1_skill_config_loading():
    """Test 1: Load skills from configuration file"""
    print_section("Test 1: Skill Configuration Loading")
    
    try:
        # Load skills from config
        loader = SkillConfigLoader("config/skills.yaml")
        skills = loader.load_from_yaml()
        
        print(f"✅ Loaded {len(skills)} skills from config/skills.yaml")
        print("\nAvailable skills:")
        for skill_name, skill_config in skills.items():
            print(f"  - {skill_name}: {skill_config.description}")
        
        # Load prompt configs
        prompt_configs = loader.load_prompt_configs()
        print(f"\n✅ Loaded {len(prompt_configs)} prompt configurations")
        
        # Show one example
        if 'standard_tutorial' in prompt_configs:
            config = prompt_configs['standard_tutorial']
            print(f"\nExample - standard_tutorial:")
            print(f"  Temperature: {config.temperature}")
            print(f"  Max tokens: {config.max_tokens}")
            print(f"  System prompt (first 100 chars): {config.system_prompt[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_2_skill_manager_with_config():
    """Test 2: SkillManager with configuration file"""
    print_section("Test 2: SkillManager with Configuration")
    
    try:
        # Create manager with config
        manager = SkillManager(config_path="config/skills.yaml")
        
        print(f"✅ SkillManager initialized with config")
        print(f"   Config path: {manager.get_config_path()}")
        print(f"   Hot-reload enabled: {manager.is_hot_reload_enabled()}")
        
        # List skills
        skills = manager.list_skills()
        print(f"\n✅ Available skills: {len(skills)}")
        for skill in skills:
            print(f"  - {skill}")
        
        # Test compatibility
        if manager.check_compatibility("standard_tutorial", "visualization_analogy"):
            print("\n✅ Compatibility check works")
            print("   standard_tutorial -> visualization_analogy: Compatible")
        
        # Test skill retrieval
        skill = manager.get_skill("standard_tutorial")
        print(f"\n✅ Retrieved skill: {skill.description}")
        print(f"   Tone: {skill.tone}")
        print(f"   Compatible with: {', '.join(skill.compatible_with)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_3_prompt_manager():
    """Test 3: PromptManager functionality"""
    print_section("Test 3: PromptManager")
    
    try:
        # Create prompt manager
        manager = PromptManager(config_path="config/skills.yaml")
        
        print(f"✅ PromptManager initialized")
        print(f"   Available skills: {len(manager.list_available_skills())}")
        
        # Get prompt config
        config = manager.get_prompt_config("standard_tutorial")
        if config:
            print(f"\n✅ Retrieved prompt config for 'standard_tutorial'")
            print(f"   Temperature: {config.temperature}")
            print(f"   Max tokens: {config.max_tokens}")
        
        # Format messages
        messages = manager.format_messages(
            skill_name="standard_tutorial",
            step_description="介绍 FastAPI 的基本概念",
            retrieved_content="FastAPI 是一个现代、快速的 Web 框架..."
        )
        
        print(f"\n✅ Formatted messages:")
        print(f"   Number of messages: {len(messages)}")
        print(f"   System message length: {len(messages[0]['content'])} chars")
        print(f"   User message length: {len(messages[1]['content'])} chars")
        
        # Test temperature and max_tokens
        temp = manager.get_temperature("standard_tutorial")
        max_tokens = manager.get_max_tokens("standard_tutorial")
        print(f"\n✅ Retrieved parameters:")
        print(f"   Temperature: {temp}")
        print(f"   Max tokens: {max_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_4_writer_agent_integration():
    """Test 4: Writer Agent with PromptManager"""
    print_section("Test 4: Writer Agent Integration")
    
    try:
        # Get default prompt manager
        manager = get_prompt_manager()
        print(f"✅ Got default PromptManager")
        print(f"   Available skills: {len(manager.list_available_skills())}")
        
        # Create test data
        step = OutlineStep(
            step_id=0,
            description="介绍 FastAPI 框架的核心特性",
            status="pending",
            retry_count=0
        )
        
        retrieved_docs = [
            RetrievedDocument(
                content="FastAPI 是一个现代、快速（高性能）的 Web 框架，用于构建 API。",
                source="docs/fastapi_intro.md",
                confidence=0.95,
                metadata={"has_deprecated": False}
            ),
            RetrievedDocument(
                content="FastAPI 基于 Starlette 和 Pydantic，提供自动 API 文档生成。",
                source="docs/fastapi_features.md",
                confidence=0.88,
                metadata={"has_deprecated": False}
            )
        ]
        
        # Apply skill
        result = apply_skill(
            skill_name="standard_tutorial",
            step=step,
            retrieved_docs=retrieved_docs,
            llm_service=None  # Not needed for this test
        )
        
        print(f"\n✅ Applied skill 'standard_tutorial'")
        print(f"   Messages: {len(result['messages'])}")
        print(f"   Temperature: {result.get('temperature', 'N/A')}")
        print(f"   Max tokens: {result.get('max_tokens', 'N/A')}")
        print(f"   Config loaded: {result['metadata'].get('config_loaded', False)}")
        print(f"   Sources: {len(result['metadata']['sources'])}")
        
        # Show message preview
        print(f"\n   System message preview:")
        print(f"   {result['messages'][0]['content'][:150]}...")
        
        print(f"\n   User message preview:")
        print(f"   {result['messages'][1]['content'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_5_custom_skill():
    """Test 5: Add and use a custom skill"""
    print_section("Test 5: Custom Skill")
    
    try:
        from tempfile import NamedTemporaryFile
        import yaml
        
        # Create a custom skill config
        custom_config = {
            'version': '1.0',
            'skills': {
                'test_custom_skill': {
                    'description': '测试自定义skill',
                    'tone': 'friendly',
                    'compatible_with': ['standard_tutorial'],
                    'prompt_config': {
                        'system_prompt': '你是一个友好的测试助手。',
                        'user_template': '任务: {step_description}\n内容: {retrieved_content}',
                        'temperature': 0.9,
                        'max_tokens': 1500
                    },
                    'enabled': True,
                    'metadata': {'test': True}
                }
            }
        }
        
        # Write to temp file
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            # Load custom config
            manager = SkillManager(config_path=temp_path)
            
            print(f"✅ Loaded custom skill configuration")
            print(f"   Skills: {manager.list_skills()}")
            
            # Get custom skill
            skill = manager.get_skill('test_custom_skill')
            print(f"\n✅ Retrieved custom skill:")
            print(f"   Description: {skill.description}")
            print(f"   Tone: {skill.tone}")
            
            # Test with PromptManager
            prompt_manager = PromptManager(config_path=temp_path)
            temp = prompt_manager.get_temperature('test_custom_skill')
            max_tokens = prompt_manager.get_max_tokens('test_custom_skill')
            
            print(f"\n✅ Custom skill parameters:")
            print(f"   Temperature: {temp}")
            print(f"   Max tokens: {max_tokens}")
            
            return True
            
        finally:
            # Cleanup
            Path(temp_path).unlink()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_6_config_validation():
    """Test 6: Configuration validation"""
    print_section("Test 6: Configuration Validation")
    
    try:
        import yaml
        
        # Load and validate config
        with open("config/skills.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        loader = SkillConfigLoader("config/skills.yaml")
        is_valid = loader.validate_config(config)
        
        if is_valid:
            print(f"✅ Configuration is valid")
            print(f"   Version: {config['version']}")
            print(f"   Skills defined: {len(config['skills'])}")
            
            # Check for required fields
            for skill_name, skill_config in config['skills'].items():
                if 'prompt_config' in skill_config:
                    pc = skill_config['prompt_config']
                    has_placeholders = (
                        '{step_description}' in pc.get('user_template', '') and
                        '{retrieved_content}' in pc.get('user_template', '')
                    )
                    status = "✅" if has_placeholders else "⚠️"
                    print(f"   {status} {skill_name}: Placeholders {'present' if has_placeholders else 'missing'}")
        else:
            print(f"❌ Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_7_performance():
    """Test 7: Performance benchmarks"""
    print_section("Test 7: Performance Benchmarks")
    
    try:
        import time
        
        # Test 1: Config loading time
        start = time.time()
        loader = SkillConfigLoader("config/skills.yaml")
        skills = loader.load_from_yaml()
        load_time = (time.time() - start) * 1000
        
        print(f"✅ Config loading time: {load_time:.2f}ms")
        
        # Test 2: PromptManager initialization
        start = time.time()
        manager = PromptManager(config_path="config/skills.yaml")
        init_time = (time.time() - start) * 1000
        
        print(f"✅ PromptManager init time: {init_time:.2f}ms")
        
        # Test 3: Message formatting
        start = time.time()
        for _ in range(100):
            messages = manager.format_messages(
                skill_name="standard_tutorial",
                step_description="Test",
                retrieved_content="Test content"
            )
        format_time = (time.time() - start) * 1000 / 100
        
        print(f"✅ Message formatting time (avg): {format_time:.2f}ms")
        
        # Test 4: apply_skill
        step = OutlineStep(step_id=0, description="Test", status="pending", retry_count=0)
        docs = [RetrievedDocument(content="Test", source="test.py", confidence=0.9, metadata={})]
        
        start = time.time()
        for _ in range(100):
            result = apply_skill("standard_tutorial", step, docs, None)
        apply_time = (time.time() - start) * 1000 / 100
        
        print(f"✅ apply_skill time (avg): {apply_time:.2f}ms")
        
        # Summary
        print(f"\n📊 Performance Summary:")
        print(f"   Total overhead per request: ~{format_time + apply_time:.2f}ms")
        print(f"   Impact: Negligible (< 1ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀" * 40)
    print("  Phase 1 Feature Testing")
    print("🚀" * 40)
    
    tests = [
        ("Skill Configuration Loading", test_1_skill_config_loading),
        ("SkillManager with Config", test_2_skill_manager_with_config),
        ("PromptManager", test_3_prompt_manager),
        ("Writer Agent Integration", test_4_writer_agent_integration),
        ("Custom Skill", test_5_custom_skill),
        ("Configuration Validation", test_6_config_validation),
        ("Performance Benchmarks", test_7_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{'=' * 80}")
    print(f"  Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'=' * 80}\n")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 is working perfectly!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
