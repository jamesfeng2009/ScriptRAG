#!/usr/bin/env python
"""圣斗士星矢剧本生成演示 - 基于技能的模拟模式

本演示展示圣斗士星矢剧本生成的不同技能风格：
1. 热血战斗模式 (heated_battle) - 硬刚十二宫，爆发小宇宙
2. 策略智取模式 (strategic_approach) - 以智取胜，找到对手弱点
3. 感情羁绊模式 (emotional_bond) - 伙伴情深，人性深度

使用方式：
    python demo_saint_seiya_skill_switching.py --compare  # 对比所有技能风格
    python demo_saint_seiya_skill_switching.py --workflow # 演示技能切换
    python demo_saint_seiya_skill_switching.py --interactive # 交互式模式
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.domain.skill_loader import SkillConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_theme_skills(theme_id: str = "saint_seiya") -> Dict[str, Any]:
    """加载主题的技能配置"""
    themes_dir = Path("config/themes")
    loader = SkillConfigLoader(themes_dir=themes_dir)
    
    theme = loader.load_theme(theme_id)
    if not theme:
        raise ValueError(f"Theme not found: {theme_id}")
    
    return theme.get("skills", {})


def get_skill_info(skills: Dict[str, Any]) -> List[Dict[str, str]]:
    """获取技能选项列表"""
    loader = SkillConfigLoader(themes_dir=Path("config/themes"))
    return loader.get_theme_skill_options("saint_seiya")


def simulate_script_generation(
    skill_id: str,
    skill_config: Dict[str, Any],
    step_description: str
) -> str:
    """
    模拟剧本生成（基于技能配置的模板）
    
    Args:
        skill_id: 技能 ID
        skill_config: 技能配置
        step_description: 步骤描述
        
    Returns:
        模拟生成的剧本片段
    """
    tone = skill_config.get("tone", "normal")
    description = skill_config.get("description", "")
    
    templates = {
        "heated_battle": f"""【热血战斗模式】
场景：{step_description}

艾欧里亚的拳头如同闪电般落下，每一击都携带着足以撕裂空间的力量。
"这就是黄金圣斗士的实力吗？"星矢艰难地站起来，嘴角渗出鲜血。

"你已经很强了，但还远远不够！"艾欧里亚冷笑一声，"接受现实吧凡人！"

星矢紧握双拳，心中燃烧着不屈的意志。
"雅典娜...我绝对不能在这里倒下！"

突然，星矢的小宇宙开始剧烈燃烧！
"燃烧吧！我的小宇宙！"

天马流星拳的星光划破狮子宫的黑暗，与艾欧里亚的闪电光速拳碰撞在一起！
整座宫殿都在这股力量下颤抖！

这不仅仅是战斗，更是意志的较量！
星矢用行动证明了：即使面对神明般的对手，
人类的小宇宙也能爆发出无限的可能！""",

        "strategic_approach": f"""【策略智取模式】
场景：{step_description}

"不能在这样硬拼下去了..."星矢在战斗中思考着。

通过仔细观察，星矢发现了艾欧里亚招式的规律：
"他的闪电光速拳虽然威力无穷，但每次出招前都有一个微小的蓄势动作..."

更重要的是，艾欧里亚作为狮子宫的守护者，他内心深处有着不可告人的愧疚。
撒加控制下的艾欧里亚，其实一直承受着良心的煎熬。

"破绽不在他的招式..."星矢暗想，"而在他的心里！"

神话时代的故事浮现在脑海：
雅典娜之所以能战胜强大的泰坦，
不是因为力量，而是因为爱与正义的力量能够感化一切。

"艾欧里亚！你真的想伤害雅典娜吗？"
"你内心的声音是什么？"

这一问，直击艾欧里亚的心灵防线！
战斗的胜负，往往不在于谁的拳头更硬，
而在于谁能够看穿对手的内心！""",

        "emotional_bond": f"""【感情羁绊模式】
场景：{step_description}

在战斗的间隙，星矢的意识开始模糊...

他想起了离开天马星座前的那个夜晚。
"星矢，记住，你不是一个人在战斗。"老师临终前的话仿佛还在耳边。

他想起了雅典娜那温柔的眼神。
"星矢，我相信你..."
那份信任，是支撑他走到现在的力量源泉。

"紫龙...冰河...瞬...一辉..."
战友们的声音在心中响起。

"我们约定过，要一起保护雅典娜！"
"无论前方有多少困难，我们都不是一个人在战斗！"

泪水与汗水交织，但星矢的眼神却越来越坚定。
"对不起让大家担心了..."

"但是！我绝对不能在这里倒下！"
"因为...我不是一个人！"

星矢的小宇宙在这一刻达到了前所未有的高度！
这不是为了自己，而是为了所有信任他、等待他的人！
这就是圣斗士的力量源泉——爱与羁绊的力量！"""
    }
    
    return templates.get(skill_id, f"【{skill_id}模式】\n场景：{step_description}\n\n（模拟内容）")


def demo_skill_comparison():
    """演示不同技能生成风格的对比"""
    print(f"\n{'='*70}")
    print("🎭 圣斗士星矢 - 技能风格对比演示")
    print("="*70)
    
    skills = load_theme_skills()
    skill_options = get_skill_info(skills)
    
    step_desc = "星矢面对黄金圣斗士艾欧里亚的绝招闪电光速拳"
    
    for skill_info in skill_options:
        skill_id = skill_info.get("id", "")
        name = skill_info.get("name", "")
        icon = skill_info.get("icon", "")
        
        if skill_id not in skills:
            continue
        
        skill_config = skills[skill_id]
        
        print(f"\n{icon} {name} ({skill_id})")
        print("-" * 50)
        print(f"📝 技能描述: {skill_config.get('description', '')}")
        print(f"🎭 语气风格: {skill_config.get('tone', '')}")
        print(f"\n📖 场景: {step_desc}")
        print("="*50)
        
        screenplay = simulate_script_generation(skill_id, skill_config, step_desc)
        print(screenplay)
        print()


def demo_skill_switching():
    """演示在工作流中切换技能"""
    print(f"\n{'='*70}")
    print("🔄 圣斗士星矢 - 技能切换剧本演示")
    print("="*70)
    
    skills = load_theme_skills()
    skill_options = get_skill_info(skills)
    
    print(f"\n📋 加载到的技能: {list(skills.keys())}")
    print(f"📋 技能选项: {[s.get('id') for s in skill_options]}")
    
    skill_switches = [
        ("第一幕", "heated_battle", "星矢闯入狮子宫，面对黄金圣斗士艾欧里亚"),
        ("第二幕", "emotional_bond", "星矢回忆起与紫龙的友情和约定"),
        ("第三幕", "strategic_approach", "星矢分析艾欧里亚的弱点和心理防线"),
        ("第四幕", "heated_battle", "星矢小宇宙爆发，使出天马流星拳的真正力量"),
        ("第五幕", "emotional_bond", "星矢用友情和信任感化了艾欧里亚")
    ]
    
    print(f"\n🎬 剧本：圣斗士星矢 - 狮子宫篇")
    print(f"📊 计划技能切换次数: {len(skill_switches)}")
    print("="*70)
    
    full_script = []
    actual_switches = 0
    
    for act, skill_id, description in skill_switches:
        skill_info = next((s for s in skill_options if s.get("id") == skill_id), {})
        icon = skill_info.get("icon", "📝")
        skill_name = skill_info.get("name", skill_id)
        
        if skill_id not in skills:
            print(f"⚠️ 跳过技能: {skill_id} (不在 skills 字典中)")
            continue
        
        skill_config = skills[skill_id]
        actual_switches += 1
        
        print(f"\n{'='*70}")
        print(f"{act}: {description}")
        print(f"🎭 当前技能: {icon} {skill_name}")
        print("="*70)
        
        screenplay = simulate_script_generation(skill_id, skill_config, description)
        print(screenplay)
        
        full_script.append(f"\n{'='*50}\n")
        full_script.append(f"【{act}】技能: {skill_name}\n")
        full_script.append(screenplay)
    
    print(f"\n{'='*70}")
    print(f"📊 完整剧本长度: {sum(len(s) for s in full_script)} 字符")
    print(f"🔄 实际技能切换次数: {actual_switches}")
    print(f"📋 使用的技能: {list(dict.fromkeys([s[1] for s in skill_switches]))}")
    print("="*70)


def interactive_mode():
    """交互式模式"""
    print(f"\n{'='*70}")
    print("🎮 交互式圣斗士星矢剧本生成")
    print("="*70)
    
    skills = load_theme_skills()
    skill_options = get_skill_info(skills)
    current_skill_id = skill_options[0].get("id") if skill_options else "heated_battle"
    
    print("\n命令说明:")
    print("  /skill <id>   - 切换技能 (heated_battle / strategic_approach / emotional_bond)")
    print("  /list         - 列出所有可用技能")
    print("  /info <skill> - 查看技能详情")
    print("  /quit         - 退出")
    print("  直接输入场景描述即可生成剧本\n")
    
    print(f"🎯 当前技能: {current_skill_id}")
    print("\n可用技能:")
    for s in skill_options:
        icon = s.get("icon", "📝")
        name = s.get("name", s.get("id"))
        desc = s.get("description", "")
        print(f"  {icon} {name}: {desc}")
    
    print("\n" + "="*70)
    print("💡 场景示例:")
    print("  - 星矢面对艾欧里亚的闪电光速拳")
    print("  - 紫龙为救星矢牺牲自己的生命")
    print("  - 星矢分析撒加的弱点")
    print("  - 一辉登场救援瞬")
    print("="*70)
    
    while True:
        try:
            user_input = input(f"\n🎤 请输入场景描述 (或输入命令: /skill /list /quit): ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit':
                    print("\n👋 再见！圣斗士之旅到此结束！")
                    break
                
                elif command == '/list':
                    print("\n📋 可用技能:")
                    for s in skill_options:
                        icon = s.get("icon", "📝")
                        name = s.get("name", s.get("id"))
                        desc = s.get("description", "")
                        print(f"  {icon} {name}: {desc}")
                
                elif command.startswith('/skill '):
                    new_skill = user_input.split()[1]
                    if new_skill in skills:
                        current_skill_id = new_skill
                        print(f"\n✅ 已切换到技能: {current_skill_id}")
                    else:
                        print(f"\n❌ 未知技能: {new_skill}")
                        print(f"可用技能: {list(skills.keys())}")
                
                elif command.startswith('/info '):
                    skill_id = user_input.split()[1]
                    if skill_id in skills:
                        skill_config = skills[skill_id]
                        print(f"\n📝 技能详情 - {skill_id}:")
                        print(f"  描述: {skill_config.get('description', '')}")
                        print(f"  语气: {skill_config.get('tone', '')}")
                    else:
                        print(f"\n❌ 未知技能: {skill_id}")
                
                else:
                    print(f"\n❓ 未知命令: {user_input}")
                
                continue
            
            skill_config = skills.get(current_skill_id, skills.get("heated_battle"))
            
            print(f"\n🎭 使用技能: {current_skill_id}")
            print(f"📖 场景: {user_input}")
            print("="*70)
            
            screenplay = simulate_script_generation(current_skill_id, skill_config, user_input)
            print("\n📖 生成的剧本片段:")
            print(screenplay)
        
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break


def demo_skill_info():
    """展示技能配置信息"""
    print(f"\n{'='*70}")
    print("📋 圣斗士星矢 - 技能配置信息")
    print("="*70)
    
    skills = load_theme_skills()
    skill_options = get_skill_info(skills)
    
    print("\n🎯 主题: 圣斗士星矢")
    print(f"📝 可用技能数量: {len(skills)}")
    
    print("\n" + "-"*70)
    print("技能列表:")
    print("-"*70)
    
    for s in skill_options:
        skill_id = s.get("id", "")
        name = s.get("name", "")
        icon = s.get("icon", "")
        desc = s.get("description", "")
        triggers = s.get("trigger_keywords", [])
        
        print(f"\n{icon} {name} ({skill_id})")
        print(f"   描述: {desc}")
        print(f"   触发词: {', '.join(triggers[:5])}")
        
        if skill_id in skills:
            config = skills[skill_id]
            print(f"   语气: {config.get('tone', 'N/A')}")
            prompt_config = config.get("prompt_config", {})
            print(f"   Temperature: {prompt_config.get('temperature', 'N/A')}")
    
    print("\n" + "="*70)
    print("💡 技能选择建议:")
    print("-"*70)
    print("  🔥 热血战斗 - 适合激烈的战斗场面，强调意志力和成长")
    print("  🧠 策略智取 - 适合分析对手弱点，利用神话典故")
    print("  💕 感情羁绊 - 适合描写友情、牺牲、感动的场景")
    print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="圣斗士星矢剧本生成演示 - 基于技能的方向切换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_saint_seiya_skill_switching.py --compare     # 对比所有技能风格
  python demo_saint_seiya_skill_switching.py --workflow    # 演示技能切换
  python demo_saint_seiya_skill_switching.py --interactive # 交互式模式
  python demo_saint_seiya_skill_switching.py --info        # 查看技能配置
        """
    )
    
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='对比所有技能的生成风格'
    )
    parser.add_argument(
        '--workflow', 
        action='store_true',
        help='演示工作流中的技能切换'
    )
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='交互式模式'
    )
    parser.add_argument(
        '--info', 
        action='store_true',
        help='查看技能配置信息'
    )
    
    args = parser.parse_args()
    
    print("\n🔧 加载圣斗士星矢主题配置...")
    skills = load_theme_skills()
    
    if args.info:
        demo_skill_info()
    elif args.compare:
        demo_skill_comparison()
    elif args.workflow:
        demo_skill_switching()
    elif args.interactive:
        interactive_mode()
    else:
        parser.print_help()
        
        print("\n" + "="*70)
        print("📋 可用命令选项:")
        print("="*70)
        print("\n  --compare     - 对比所有技能的生成风格")
        print("  --workflow    - 演示工作流中的技能切换")
        print("  --interactive - 交互式模式")
        print("  --info        - 查看技能配置信息")
        
        print("\n" + "="*70)
        print("📋 可用技能选项:")
        print("="*70)
        
        skill_options = get_skill_info(skills)
        
        for s in skill_options:
            skill_id = s.get("id", "")
            name = s.get("name", "")
            icon = s.get("icon", "")
            desc = s.get("description", "")
            
            print(f"\n  {icon} {name} ({skill_id})")
            print(f"     {desc}")
        
        print("\n💡 使用 --compare 查看不同技能的生成效果")
        print("="*70)


if __name__ == "__main__":
    main()
