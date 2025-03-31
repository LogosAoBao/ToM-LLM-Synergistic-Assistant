# main.py
import argparse
import json
import os
import time
from typing import List, Dict

from tom_llm_system import (
    ToMCollaborationSystem,
    ExperimentModule,
    VirtualEnvironmentInterface
)


# 基线系统实现
class BaselineSystem:
    """基线系统实现(无ToM)"""

    def __init__(self, config=None):
        """初始化基线系统"""
        # 使用相同的环境和感知模块，但不使用ToM
        self.full_system = ToMCollaborationSystem(config)
        self.name = "NoToM-Baseline"

    def run_complete_demo(self, demo_data):
        """运行完整流程，但不使用ToM推理"""
        results = {}

        try:
            # 重置系统
            self.full_system.reset()

            # 观察演示并推断任务
            task_goal = self.full_system.observe_and_infer_task(demo_data)
            results["task_goal"] = task_goal

            # 空信念模型
            empty_beliefs = {}
            for agent_id in self.full_system.environment.current_state.agents:
                if agent_id != self.full_system.config["robot_id"]:
                    # 创建一个空的默认信念
                    empty_beliefs[agent_id] = {
                        "knowledge": {},
                        "goals": [],
                        "attention": {},
                        "confidence": 0.5
                    }

            self.full_system.agent_beliefs = empty_beliefs
            results["agent_beliefs"] = empty_beliefs

            # 生成协作计划(不使用ToM)
            action_plan = self.full_system.planning_module.generate_collaboration_plan(
                task_goal=task_goal,
                env_state=self.full_system.environment.current_state,
                agent_beliefs=empty_beliefs,
                robot_id=self.full_system.config["robot_id"]
            )
            results["action_plan"] = action_plan

            # 执行计划
            execution_status = self.full_system.execute_plan(action_plan)
            results["execution_status"] = execution_status

            # 收集最终状态
            final_observation = self.full_system.environment.get_observation()
            results["final_state"] = final_observation

            return results

        except Exception as e:
            print(f"基线系统运行出错: {str(e)}")
            results["error"] = str(e)
            return results


def load_demo_data(data_path: str) -> List:
    """加载演示数据"""
    if os.path.isdir(data_path):
        # 加载目录中的所有演示
        demos = []
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                with open(os.path.join(data_path, filename), 'r') as f:
                    demos.append(json.load(f))
        return demos
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        # 加载单个演示文件
        with open(data_path, 'r') as f:
            return [json.load(f)]
    else:
        # 创建一个简单的演示序列
        return [
            [
                {"agent": "human", "action": "pick", "object": "cup_1", "time": 0.5},
                {"agent": "human", "action": "move", "object": "cup_1", "target": "sink", "time": 1.2},
                {"agent": "human", "action": "place", "object": "cup_1", "target": "sink", "time": 2.0}
            ]
        ]


def run_comparative_experiment(config_path: str = None, demo_path: str = None, output_dir: str = 'results'):
    """运行对比实验"""
    # 加载配置
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载演示数据
    demos = load_demo_data(demo_path or 'demo_data')

    # 创建实验模块
    experiment = ExperimentModule()

    # 创建系统
    print("初始化ToM-LLM协作系统...")
    tom_system = ToMCollaborationSystem(config)

    print("初始化基线系统...")
    baseline_system = BaselineSystem(config)

    # 运行ToM系统实验
    print("\n开始ToM系统实验...")
    tom_results = experiment.run_experiment(
        system=tom_system,
        demos=demos,
        experiment_name="ToM-LLM-System"
    )

    # 运行基线系统实验
    print("\n开始基线系统实验...")
    baseline_results = experiment.run_experiment(
        system=baseline_system,
        demos=demos,
        experiment_name="Baseline-System"
    )

    # 对比结果
    comparison = experiment.compare_results(tom_results, baseline_results)

    # 保存结果
    timestamp = int(time.time())
    with open(os.path.join(output_dir, f'tom_results_{timestamp}.json'), 'w') as f:
        json.dump(tom_results, f, indent=2)

    with open(os.path.join(output_dir, f'baseline_results_{timestamp}.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    with open(os.path.join(output_dir, f'comparison_{timestamp}.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # 生成结果报告
    report = experiment.generate_report(comparison)
    with open(os.path.join(output_dir, f'report_{timestamp}.md'), 'w') as f:
        f.write(report)

    print(f"\n实验完成，结果保存在 {output_dir} 目录")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行ToM-LLM协作系统实验')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--demo', type=str, help='演示数据路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['full', 'single', 'tom_only', 'baseline_only'],
                        default='full', help='运行模式')

    args = parser.parse_args()

    if args.mode == 'full':
        # 运行完整对比实验
        run_comparative_experiment(args.config, args.demo, args.output)
    elif args.mode == 'single':
        # 运行单个演示
        config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)

        system = ToMCollaborationSystem(config)
        demo = load_demo_data(args.demo or 'demo_data')[0]

        results = system.run_complete_demo(demo)
        print(json.dumps(results, indent=2))