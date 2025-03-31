"""
ToM-LLM协作系统 - 主要软件模块实现示例

该文件展示了系统各核心模块的实现方案，包括:
1. 感知与目标推理模块
2. 心理理论(ToM)模块
3. 协作规划模块
4. 环境接口
5. 系统集成
"""

import os
import json
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import random
from transformers import AutoModel, AutoTokenizer
#from mock_models import nn, MockModel, MockTokenizer

# =====================================================================
# 1. 配置与数据结构定义
# =====================================================================

@dataclass
class EnvironmentState:
    """环境状态表示"""
    objects: Dict[str, Dict]  # 对象ID到属性的映射
    agents: Dict[str, Dict]   # 代理ID到状态的映射
    relations: List[Dict]     # 关系列表(主体,关系,客体)
    timestamp: float          # 时间戳

    def to_text_representation(self) -> str:
        """将环境状态转换为文本表示，用于LLM处理"""
        # 实现环境到文本的转换逻辑
        text = "环境状态描述:\n"

        # 描述物体
        text += "物体:\n"
        for obj_id, attrs in self.objects.items():
            text += f"- {obj_id}: {', '.join([f'{k}={v}' for k, v in attrs.items()])}\n"

        # 描述代理
        text += "代理:\n"
        for agent_id, state in self.agents.items():
            text += f"- {agent_id}: {', '.join([f'{k}={v}' for k, v in state.items()])}\n"

        # 描述关系
        text += "关系:\n"
        for rel in self.relations:
            text += f"- {rel['subject']} {rel['relation']} {rel['object']}\n"

        return text


@dataclass
class TaskGoal:
    """任务目标表示"""
    name: str                      # 任务名称
    target_state: Dict[str, Any]   # 目标状态描述
    constraints: List[str]         # 约束条件
    decomposition: List[str]       # 任务分解步骤
    priority: float = 1.0          # 任务优先级

    def to_text_representation(self) -> str:
        """将任务目标转换为文本表示"""
        text = f"任务: {self.name}\n"
        text += "目标状态:\n"
        for k, v in self.target_state.items():
            text += f"- {k}: {v}\n"

        text += "约束条件:\n"
        for c in self.constraints:
            text += f"- {c}\n"

        text += "任务分解:\n"
        for i, step in enumerate(self.decomposition):
            text += f"{i+1}. {step}\n"

        return text


@dataclass
class AgentBelief:
    """心理理论中的代理信念模型"""
    agent_id: str                  # 代理ID
    knowledge: Dict[str, float]    # 知识项及置信度
    goals: List[Dict[str, float]]  # 目标及优先级
    attention: Dict[str, float]    # 注意力分配(对象ID到注意力权重)
    confidence: float = 0.8        # 对该信念模型的整体置信度

    def to_text_representation(self) -> str:
        """将代理信念转换为文本表示"""
        text = f"代理 {self.agent_id} 的信念模型 (置信度: {self.confidence:.2f}):\n"

        # 知识状态
        text += "知识状态:\n"
        for k, v in self.knowledge.items():
            text += f"- {k} (确信度: {v:.2f})\n"

        # 目标
        text += "可能的目标:\n"
        for g in self.goals:
            for goal, priority in g.items():
                text += f"- {goal} (优先级: {priority:.2f})\n"

        # 注意力
        text += "注意力分配:\n"
        for obj, attn in self.attention.items():
            text += f"- {obj}: {attn:.2f}\n"

        return text


@dataclass
class Action:
    """动作表示"""
    agent_id: str           # 执行代理
    action_type: str        # 动作类型
    targets: List[str]      # 目标对象
    parameters: Dict        # 参数
    preconditions: List     # 前提条件
    effects: List           # 效果
    duration: float         # 持续时间
    confidence: float = 1.0 # 置信度


# =====================================================================
# 2. 感知与目标推理模块
# =====================================================================

class PerceptionModule:
    """通过视频或动作序列理解任务目标"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """初始化感知模块

        Args:
            model_path: 预训练视觉语言模型路径
            device: 计算设备
        """
        self.device = device
        # 加载视觉语言模型(如CLIP或多模态LLM)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 动作识别器
        self.action_recognizer = self._initialize_action_recognizer()
        # 目标提取器
        self.goal_extractor = self._initialize_goal_extractor()

    def _initialize_action_recognizer(self):
        """初始化动作识别组件"""
        # 在实际实现中，这可能是一个专门的动作识别模型
        # 这里简化为示例
        return lambda x: {"actions": ["pick", "move", "place"], "confidence": 0.9}

    def _initialize_goal_extractor(self):
        """初始化目标提取组件"""
        # 在实际实现中，这可能是一个基于LLM的推理组件
        return lambda x, y: {"goal": "清理桌子", "steps": ["收集物品", "擦拭表面", "整理物品"]}

    def process_demonstration(self,
                             demo_data: Union[str, List],
                             env_context: EnvironmentState) -> TaskGoal:
        """处理演示数据，提取任务目标

        Args:
            demo_data: 演示数据(视频路径或动作序列)
            env_context: 环境上下文

        Returns:
            提取的任务目标
        """
        if isinstance(demo_data, str) and os.path.exists(demo_data):
            # 处理视频文件
            return self._process_video(demo_data, env_context)
        else:
            # 处理动作序列
            return self._process_action_sequence(demo_data, env_context)

    def _process_video(self, video_path: str, env_context: EnvironmentState) -> TaskGoal:
        """从视频中提取任务目标

        在实际实现中，这将使用视觉模型处理视频帧
        """
        # 模拟视频处理和帧提取
        print(f"处理视频: {video_path}")

        # 1. 提取关键帧
        # 2. 识别每帧中的物体和动作
        # 3. 将视频转换为动作序列
        simulated_action_sequence = [
            {"agent": "human", "action": "pick", "object": "cup", "time": 0.5},
            {"agent": "human", "action": "move", "object": "cup", "target": "sink", "time": 1.2},
            {"agent": "human", "action": "place", "object": "cup", "target": "sink", "time": 2.0},
        ]

        # 使用与动作序列相同的处理逻辑
        return self._process_action_sequence(simulated_action_sequence, env_context)

    def _process_action_sequence(self,
                                actions: List[Dict],
                                env_context: EnvironmentState) -> TaskGoal:
        """从动作序列中提取任务目标"""
        # 1. 分析动作序列
        action_analysis = self._analyze_actions(actions)

        # 2. 结合环境上下文推断目标
        goal_inference = self._infer_goal(action_analysis, env_context)

        # 3. 构建结构化的任务目标
        task_goal = TaskGoal(
            name=goal_inference["goal"],
            target_state=self._construct_target_state(actions, env_context),
            constraints=self._extract_constraints(actions, env_context),
            decomposition=goal_inference["steps"],
            priority=1.0
        )

        return task_goal

    def _analyze_actions(self, actions: List[Dict]) -> Dict:
        """分析动作序列中的模式"""
        # 提取动作类型、操作的对象和时序信息
        action_types = [a["action"] for a in actions]
        objects = [a.get("object", "") for a in actions]
        targets = [a.get("target", "") for a in actions if "target" in a]

        result = {
            "action_types": action_types,
            "objects": objects,
            "targets": targets,
            "frequency": {a: action_types.count(a) for a in set(action_types)},
            "duration": actions[-1]["time"] - actions[0]["time"] if "time" in actions[0] else 0
        }

        return result

    def _infer_goal(self, action_analysis: Dict, env_context: EnvironmentState) -> Dict:
        """推断任务目标"""
        # 构建提示文本
        prompt = f"""
        根据以下动作序列和环境信息，推断人类可能的任务目标:
        
        动作序列:
        {json.dumps(action_analysis, indent=2, ensure_ascii=False)}
        
        环境信息:
        {env_context.to_text_representation()}
        
        推断任务目标和分解步骤:
        """

        # 在实际实现中，这里会使用LLM进行推理
        # 这里简化为示例返回值
        if "cup" in str(action_analysis) and "sink" in str(action_analysis):
            return {
                "goal": "清理餐具",
                "steps": [
                    "收集餐桌上的脏餐具",
                    "将脏餐具放入水槽",
                    "清洗餐具",
                    "将干净餐具放回架子"
                ]
            }
        else:
            return {
                "goal": "整理房间",
                "steps": [
                    "收集散落的物品",
                    "将物品归类",
                    "放回原位"
                ]
            }

    def _construct_target_state(self,
                               actions: List[Dict],
                               env_context: EnvironmentState) -> Dict[str, Any]:
        """构建目标状态描述"""
        # 基于最后一个动作和环境状态推断目标状态
        # 在实际实现中，这需要更复杂的逻辑
        target_state = {}

        # 提取最后动作的目标位置或状态
        if actions and "target" in actions[-1]:
            target_obj = actions[-1]["object"]
            target_loc = actions[-1]["target"]
            target_state[target_obj] = {"location": target_loc}

        # 添加环境中可能的目标状态
        for obj_id, attrs in env_context.objects.items():
            if "clean" in attrs and not attrs["clean"]:
                target_state[obj_id] = {"clean": True}

        return target_state

    def _extract_constraints(self,
                            actions: List[Dict],
                            env_context: EnvironmentState) -> List[str]:
        """提取任务约束条件"""
        constraints = []

        # 分析环境中的特殊对象属性
        for obj_id, attrs in env_context.objects.items():
            if "fragile" in attrs and attrs["fragile"]:
                constraints.append(f"小心处理易碎物品 {obj_id}")
            if "heavy" in attrs and attrs["heavy"]:
                constraints.append(f"可能需要协助搬运重物 {obj_id}")

        # 分析动作序列中的时间约束
        if len(actions) >= 2 and "time" in actions[0] and "time" in actions[-1]:
            duration = actions[-1]["time"] - actions[0]["time"]
            if duration < 10:  # 假设10秒是快速任务的阈值
                constraints.append("任务需要快速完成")

        return constraints


# =====================================================================
# 3. 心理理论(ToM)模块
# =====================================================================

class TheoryOfMindModule:
    """基于LLM的心理理论推理模块"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """初始化ToM模块

        Args:
            model_path: 预训练LLM模型路径
            device: 计算设备
        """
        self.device = device
        # 实际实现应使用专门为ToM任务微调的LLM
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 历史信念跟踪器
        self.belief_history = {}

        # 置信度调整参数
        self.learning_rate = 0.05  # 信念更新学习率

    def infer_agent_belief(self,
                          agent_id: str,
                          env_state: EnvironmentState,
                          actions_history: List[Dict],
                          task_context: Optional[TaskGoal] = None) -> AgentBelief:
        """推断代理的信念、意图和目标

        Args:
            agent_id: 目标代理ID
            env_state: 当前环境状态
            actions_history: 历史动作序列
            task_context: 可选的任务上下文

        Returns:
            代理的信念模型
        """
        # 1. 准备LLM推理的提示文本
        prompt = self._prepare_tom_prompt(agent_id, env_state, actions_history, task_context)

        # 2. 使用LLM进行推理(真实实现中调用LLM API)
        # 此处简化为模拟输出
        tom_output = self._simulate_llm_tom_inference(prompt, agent_id, actions_history)

        # 3. 解析LLM输出为结构化信念
        belief = self._parse_belief_from_llm(tom_output, agent_id)

        # 4. 更新历史信念记录
        self._update_belief_history(agent_id, belief)

        return belief

    def _prepare_tom_prompt(self,
                           agent_id: str,
                           env_state: EnvironmentState,
                           actions_history: List[Dict],
                           task_context: Optional[TaskGoal] = None) -> str:
        """准备ToM推理的提示文本"""
        prompt = f"""
        基于心理理论(Theory of Mind)分析，推断代理 '{agent_id}' 的信念、知识状态和目标。
        
        环境状态:
        {env_state.to_text_representation()}
        
        最近的动作历史:
        """

        # 添加动作历史
        for i, action in enumerate(actions_history[-5:]):  # 只使用最近5个动作
            prompt += f"{i+1}. {action['agent']} 执行 {action['action']} "
            if "object" in action:
                prompt += f"对象: {action['object']} "
            if "target" in action:
                prompt += f"目标: {action['target']} "
            prompt += "\n"

        # 添加任务上下文(如果有)
        if task_context:
            prompt += f"\n任务上下文:\n{task_context.to_text_representation()}\n"

        # 添加历史信念(如果有)
        if agent_id in self.belief_history:
            prompt += f"\n上一时刻的信念评估:\n{self.belief_history[agent_id].to_text_representation()}\n"

        # 提示具体输出格式
        prompt += """
        请分析并输出:
        1. 代理可能的知识状态(列出关键知识项和确信度)
        2. 代理可能的目标(按优先级排序)
        3. 代理当前的注意力焦点
        4. 整体信念模型的置信度评估
        
        输出格式:
        知识状态:
        - [知识项]: [确信度]
        
        目标:
        - [目标描述]: [优先级]
        
        注意力:
        - [对象]: [注意力权重]
        
        整体置信度: [0-1之间的数值]
        """

        return prompt

    def _simulate_llm_tom_inference(self,
                                   prompt: str,
                                   agent_id: str,
                                   actions_history: List[Dict]) -> str:
        """模拟LLM的ToM推理输出

        在实际系统中，这里会调用LLM API
        """
        # 根据动作历史简单推断一些模式
        target_objects = [a.get("object", "") for a in actions_history if a.get("agent") == agent_id]
        frequent_objects = set([obj for obj in target_objects if target_objects.count(obj) > 1])

        # 模拟输出
        if "human" in agent_id:
            return f"""
            知识状态:
            - 房间布局: 0.95
            - 物品位置: 0.85
            - 协作机器人能力: 0.70
            - 任务完成标准: 0.90
            
            目标:
            - 清理房间: 0.85
            - 准备晚餐: 0.35
            - 帮助其他代理: 0.50
            
            注意力:
            - 厨房区域: 0.70
            - 机器人: 0.20
            - {"、".join(frequent_objects) if frequent_objects else "桌子"}: 0.90
            
            整体置信度: 0.82
            """
        else:
            return f"""
            知识状态:
            - 任务指令: 0.95
            - 物品功能: 0.80
            - 环境状态变化: 0.75
            
            目标:
            - 执行指定任务: 0.90
            - 避免碰撞: 0.85
            - 高效完成: 0.70
            
            注意力:
            - 当前操作物体: 0.85
            - 人类位置: 0.75
            - 下一步目标: 0.60
            
            整体置信度: 0.78
            """

    def _parse_belief_from_llm(self, llm_output: str, agent_id: str) -> AgentBelief:
        """解析LLM输出的信念模型"""
        # 实际系统中需要更复杂的解析逻辑
        # 此处简化实现

        knowledge = {}
        goals = []
        attention = {}
        confidence = 0.8

        lines = llm_output.strip().split('\n')
        section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('知识状态:'):
                section = 'knowledge'
            elif line.startswith('目标:'):
                section = 'goals'
            elif line.startswith('注意力:'):
                section = 'attention'
            elif line.startswith('整体置信度:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('-') and section:
                parts = line[1:].split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())

                        if section == 'knowledge':
                            knowledge[key] = value
                        elif section == 'goals':
                            goals.append({key: value})
                        elif section == 'attention':
                            attention[key] = value
                    except:
                        pass

        return AgentBelief(
            agent_id=agent_id,
            knowledge=knowledge,
            goals=goals,
            attention=attention,
            confidence=confidence
        )

    def _update_belief_history(self, agent_id: str, belief: AgentBelief):
        """更新代理的信念历史"""
        # 存储当前信念
        self.belief_history[agent_id] = belief

        # 在实际系统中，可能还需要持久化存储以分析信念随时间的演变

    def update_belief_confidence(self,
                               agent_id: str,
                               feedback: Dict[str, Dict[str, float]]):
        """基于环境反馈更新信念置信度（Meta-Rewarding实现）

        Args:
            agent_id: 代理ID
            feedback: 反馈信息(类别:{预测项:正确率})
        """
        if agent_id not in self.belief_history:
            return

        belief = self.belief_history[agent_id]

        # 更新知识项置信度
        for knowledge_item, correctness in feedback.get('knowledge', {}).items():
            if knowledge_item in belief.knowledge:
                belief.knowledge[knowledge_item] = belief.knowledge[knowledge_item] * (1 - self.learning_rate) + correctness * self.learning_rate

        # 更新目标预测置信度
        for goal_feedback in feedback.get('goals', []):
            for goal, correctness in goal_feedback.items():
                for i, goal_dict in enumerate(belief.goals):
                    if goal in goal_dict:
                        belief.goals[i][goal] = belief.goals[i][goal] * (1 - self.learning_rate) + correctness * self.learning_rate

        # 更新整体置信度
        if 'overall' in feedback:
            belief.confidence = belief.confidence * (1 - self.learning_rate) + feedback['overall'] * self.learning_rate

        # 保存更新后的信念
        self.belief_history[agent_id] = belief


# =====================================================================
# 4. 协作规划模块
# =====================================================================

class CollaborationPlanningModule:
    """结合LLM和搜索算法的协作规划模块"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """初始化协作规划模块

        Args:
            model_path: 预训练LLM模型路径
            device: 计算设备
        """
        self.device = device
        # 实际实现应使用针对规划任务微调的LLM
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 动作库
        self.action_library = self._initialize_action_library()

        # 搜索算法参数
        self.max_search_depth = 5
        self.max_search_iterations = 100
        self.exploration_constant = 1.41  # UCB1算法常数

    def _initialize_action_library(self) -> Dict[str, Dict]:
        """初始化机器人可执行的动作库"""
        # 在实际系统中，这应该从配置文件加载
        return {
            "pick": {
                "parameters": ["object"],
                "preconditions": ["reachable(object)", "graspable(object)", "hand_empty()"],
                "effects": ["holding(object)", "not hand_empty()"]
            },
            "place": {
                "parameters": ["object", "target"],
                "preconditions": ["holding(object)", "reachable(target)"],
                "effects": ["on(object, target)", "hand_empty()", "not holding(object)"]
            },
            "move": {
                "parameters": ["location"],
                "preconditions": ["navigable(location)"],
                "effects": ["at(location)"]
            },
            "handover": {
                "parameters": ["object", "agent"],
                "preconditions": ["holding(object)", "near(agent)"],
                "effects": ["has(agent, object)", "hand_empty()", "not holding(object)"]
            },
            "clean": {
                "parameters": ["object"],
                "preconditions": ["holding(object)", "dirty(object)"],
                "effects": ["clean(object)"]
            }
        }

    def generate_collaboration_plan(self,
                                   task_goal: TaskGoal,
                                   env_state: EnvironmentState,
                                   agent_beliefs: Dict[str, AgentBelief],
                                   robot_id: str = "robot") -> List[Action]:
        """生成协作计划

        Args:
            task_goal: 任务目标
            env_state: 环境状态
            agent_beliefs: 各代理的信念模型
            robot_id: 机器人ID

        Returns:
            协作动作计划
        """
        # 1. 使用LLM生成高级规划
        high_level_plan = self._generate_high_level_plan(task_goal, env_state, agent_beliefs)

        # 2. 使用MCTS将高级规划转化为具体动作序列
        action_sequence = self._plan_with_mcts(high_level_plan, env_state, agent_beliefs, robot_id)

        # 3. 验证计划的可执行性
        validated_plan = self._validate_plan(action_sequence, env_state)

        # 4. 添加自我反思和错误恢复策略
        final_plan = self._add_recovery_strategies(validated_plan, task_goal)

        return final_plan

    def _generate_high_level_plan(self,
                                 task_goal: TaskGoal,
                                 env_state: EnvironmentState,
                                 agent_beliefs: Dict[str, AgentBelief]) -> List[str]:
        """使用LLM生成高级规划"""
        # 构建提示文本
        prompt = f"""
        基于任务目标和环境状态，生成实现目标的高级计划步骤。
        
        任务目标:
        {task_goal.to_text_representation()}
        
        环境状态:
        {env_state.to_text_representation()}
        
        协作代理信念模型:
        """

        # 添加代理信念信息
        for agent_id, belief in agent_beliefs.items():
            if agent_id != "robot":  # 排除机器人自身
                prompt += f"\n{belief.to_text_representation()}\n"

        prompt += """
        提供一个高级协作计划，包括:
        1. 明确的协作步骤
        2. 每个步骤的执行者(指定代理)
        3. 考虑到人类状态和信念的协作策略
        
        计划格式:
        1. [执行者] [动作描述]
        2. [执行者] [动作描述]
        ...
        """

        # 实际实现中，这里调用LLM API
        # 此处简化为示例输出
        if "清理" in task_goal.name:
            return [
                "robot 移动到餐桌附近",
                "robot 收集餐桌上的脏餐具",
                "human 打开水龙头准备清洗",
                "robot 将脏餐具递给human",
                "human 清洗餐具",
                "robot 准备擦布",
                "robot 清理餐桌表面",
                "human 将干净餐具放回架子"
            ]
        else:
            return [
                "robot 扫描房间确定要整理的物品",
                "robot 收集散落的衣物",
                "human 整理书桌",
                "robot 将衣物递给human",
                "human 将衣物放入衣柜",
                "robot 清理地面杂物",
                "human 检查房间确认完成情况"
            ]

    def _plan_with_mcts(self,
                       high_level_plan: List[str],
                       env_state: EnvironmentState,
                       agent_beliefs: Dict[str, AgentBelief],
                       robot_id: str) -> List[Action]:
        """使用蒙特卡洛树搜索将高级计划转化为具体动作"""
        action_sequence = []

        for step in high_level_plan:
            parts = step.split(' ', 1)
            agent_id = parts[0]
            action_desc = parts[1]

            if agent_id == robot_id:
                # 机器人动作：使用MCTS搜索具体实现方式
                robot_actions = self._mcts_action_search(action_desc, env_state)
                action_sequence.extend(robot_actions)
            else:
                # 人类动作：基于信念模型预测可能的行为
                human_action = self._predict_human_action(agent_id, action_desc, env_state, agent_beliefs.get(agent_id))
                if human_action:
                    action_sequence.append(human_action)

        return action_sequence

    def _mcts_action_search(self,
                           action_desc: str,
                           env_state: EnvironmentState) -> List[Action]:
        """使用MCTS搜索动作实现方案"""
        # 实际系统应实现完整的MCTS算法
        # 此处简化为基于规则的动作生成

        actions = []

        # 简单的动作匹配逻辑
        if "移动" in action_desc:
            target = self._extract_target_from_desc(action_desc, env_state)
            actions.append(Action(
                agent_id="robot",
                action_type="move",
                targets=[target],
                parameters={"speed": 0.5},
                preconditions=["navigable(location)"],
                effects=["at(location)"],
                duration=2.0
            ))

        elif "收集" in action_desc or "拿起" in action_desc:
            objects = self._extract_objects_from_desc(action_desc, env_state)
            for obj in objects:
                # 先移动到物体附近
                actions.append(Action(
                    agent_id="robot",
                    action_type="move",
                    targets=[obj],
                    parameters={"speed": 0.5},
                    preconditions=["navigable(location)"],
                    effects=["near(object)"],
                    duration=1.5
                ))

                # 拿起物体
                actions.append(Action(
                    agent_id="robot",
                    action_type="pick",
                    targets=[obj],
                    parameters={},
                    preconditions=["reachable(object)", "graspable(object)"],
                    effects=["holding(object)"],
                    duration=1.0
                ))

        elif "递给" in action_desc or "交给" in action_desc:
            obj = self._extract_objects_from_desc(action_desc, env_state)[0]
            recipient = "human" if "human" in action_desc else next((a for a in env_state.agents if a != "robot"), "human")

            actions.append(Action(
                agent_id="robot",
                action_type="move",
                targets=[recipient],
                parameters={"speed": 0.5},
                preconditions=["navigable(location)"],
                effects=["near(agent)"],
                duration=2.0
            ))

            actions.append(Action(
                agent_id="robot",
                action_type="handover",
                targets=[obj, recipient],
                parameters={},
                preconditions=["holding(object)", "near(agent)"],
                effects=["has(agent, object)", "hand_empty()"],
                duration=1.5
            ))

        elif "清理" in action_desc or "擦拭" in action_desc:
            target = self._extract_target_from_desc(action_desc, env_state)

            # 准备清洁工具
            if "准备" in action_desc:
                cleaning_tool = "cloth" if "布" in action_desc else "sponge"
                actions.append(Action(
                    agent_id="robot",
                    action_type="pick",
                    targets=[cleaning_tool],
                    parameters={},
                    preconditions=["reachable(object)", "graspable(object)"],
                    effects=["holding(object)"],
                    duration=1.0
                ))
            else:
                # 移动到目标位置
                actions.append(Action(
                    agent_id="robot",
                    action_type="move",
                    targets=[target],
                    parameters={"speed": 0.5},
                    preconditions=["navigable(location)"],
                    effects=["near(object)"],
                    duration=1.5
                ))

                # 清洁动作
                actions.append(Action(
                    agent_id="robot",
                    action_type="clean",
                    targets=[target],
                    parameters={},
                    preconditions=["near(object)"],
                    effects=["clean(object)"],
                    duration=3.0
                ))

        # 如果无法匹配任何动作，添加一个空移动动作
        if not actions:
            actions.append(Action(
                agent_id="robot",
                action_type="move",
                targets=["center"],
                parameters={"speed": 0.5},
                preconditions=[],
                effects=[],
                duration=1.0,
                confidence=0.5  # 低置信度表示这是一个回退策略
            ))

        return actions

    def _extract_target_from_desc(self, desc: str, env_state: EnvironmentState) -> str:
        """从描述中提取目标位置"""
        locations = ["桌子", "厨房", "客厅", "卧室", "浴室", "水槽"]

        for loc in locations:
            if loc in desc:
                return loc

        # 如果没有明确位置，返回一个默认值
        return "center"

    def _extract_objects_from_desc(self, desc: str, env_state: EnvironmentState) -> List[str]:
        """从描述中提取对象名称"""
        # 在实际系统中，这应该使用NLP方法提取
        common_objects = ["杯子", "盘子", "餐具", "衣物", "书", "玩具", "擦布", "杂物"]

        found_objects = []
        for obj in common_objects:
            if obj in desc:
                found_objects.append(obj)

        # 如果没找到具体对象，尝试从环境中推断
        if not found_objects:
            if "餐具" in desc or "清理" in desc and "桌子" in desc:
                for obj_id in env_state.objects:
                    if "plate" in obj_id or "cup" in obj_id or "utensil" in obj_id:
                        found_objects.append(obj_id)
            elif "衣物" in desc:
                for obj_id in env_state.objects:
                    if "cloth" in obj_id or "shirt" in obj_id or "pants" in obj_id:
                        found_objects.append(obj_id)

        # 如果仍然没有找到对象，返回通用对象
        if not found_objects:
            found_objects = ["general_object"]

        return found_objects

    def _predict_human_action(self,
                             agent_id: str,
                             action_desc: str,
                             env_state: EnvironmentState,
                             belief: Optional[AgentBelief] = None) -> Optional[Action]:
        """基于信念模型预测人类可能的行为"""
        if not belief:
            # 如果没有信念模型，创建一个基本行为
            return Action(
                agent_id=agent_id,
                action_type=self._infer_action_type(action_desc),
                targets=self._extract_objects_from_desc(action_desc, env_state),
                parameters={},
                preconditions=[],
                effects=[],
                duration=2.0,
                confidence=0.6
            )

        # 使用信念模型推断更精确的行为
        action_type = self._infer_action_type(action_desc)
        targets = self._extract_objects_from_desc(action_desc, env_state)

        # 根据注意力焦点调整目标优先级
        if targets and belief.attention:
            for target in list(targets):
                for attn_obj, weight in belief.attention.items():
                    if target in attn_obj or attn_obj in target:
                        # 将高注意力的对象移到列表前面
                        if weight > 0.7 and target != targets[0]:
                            targets.remove(target)
                            targets.insert(0, target)

        # 构建人类动作
        return Action(
            agent_id=agent_id,
            action_type=action_type,
            targets=targets,
            parameters={},
            preconditions=[],  # 人类动作不检查前提条件
            effects=self._infer_action_effects(action_type, targets),
            duration=3.0,  # 假设人类动作略慢于机器人
            confidence=belief.confidence  # 使用整体信念置信度
        )

    def _infer_action_type(self, desc: str) -> str:
        """从描述中推断动作类型"""
        action_keywords = {
            "拿": "pick",
            "取": "pick",
            "放": "place",
            "移动": "move",
            "走": "move",
            "给": "handover",
            "递": "handover",
            "清洗": "clean",
            "清理": "clean",
            "擦": "clean",
            "检查": "check",
            "确认": "check",
            "打开": "open"
        }

        for keyword, action in action_keywords.items():
            if keyword in desc:
                return action

        return "move"  # 默认动作

    def _infer_action_effects(self, action_type: str, targets: List[str]) -> List[str]:
        """推断动作的效果"""
        effects = []

        if action_type == "pick" and targets:
            effects.append(f"holding({targets[0]})")
        elif action_type == "place" and len(targets) >= 2:
            effects.append(f"on({targets[0]}, {targets[1]})")
        elif action_type == "clean" and targets:
            effects.append(f"clean({targets[0]})")

        return effects

    def _validate_plan(self,
                      action_sequence: List[Action],
                      env_state: EnvironmentState) -> List[Action]:
        """验证计划的可执行性"""
        validated_actions = []
        current_state = self._simulate_initial_state(env_state)

        for action in action_sequence:
            # 检查前提条件
            if self._check_preconditions(action, current_state):
                validated_actions.append(action)
                # 更新模拟状态
                current_state = self._apply_effects(action, current_state)
            else:
                # 条件不满足，尝试添加必要的先决动作
                prerequisite_actions = self._generate_prerequisite_actions(action, current_state)
                if prerequisite_actions:
                    validated_actions.extend(prerequisite_actions)
                    # 重新应用先决动作的效果
                    for pre_action in prerequisite_actions:
                        current_state = self._apply_effects(pre_action, current_state)

                    # 再次尝试原始动作
                    if self._check_preconditions(action, current_state):
                        validated_actions.append(action)
                        current_state = self._apply_effects(action, current_state)

        return validated_actions

    def _simulate_initial_state(self, env_state: EnvironmentState) -> Dict:
        """创建用于规划验证的简化状态表示"""
        # 将环境状态转换为简化的状态字典
        state = {
            "objects": {},
            "locations": {},
            "agents": {},
            "robot_state": {
                "holding": None,
                "hand_empty": True,
                "location": "unknown"
            }
        }

        # 提取对象信息
        for obj_id, attrs in env_state.objects.items():
            state["objects"][obj_id] = attrs.copy()

            # 推断位置信息
            if "location" in attrs:
                loc = attrs["location"]
                if loc not in state["locations"]:
                    state["locations"][loc] = []
                state["locations"][loc].append(obj_id)

        # 提取代理信息
        for agent_id, attrs in env_state.agents.items():
            state["agents"][agent_id] = attrs.copy()

            if agent_id == "robot":
                if "location" in attrs:
                    state["robot_state"]["location"] = attrs["location"]
                if "holding" in attrs and attrs["holding"]:
                    state["robot_state"]["holding"] = attrs["holding"]
                    state["robot_state"]["hand_empty"] = False

        return state

    def _check_preconditions(self, action: Action, state: Dict) -> bool:
        """检查动作的前提条件是否满足"""
        if action.agent_id != "robot":
            # 人类动作不检查前提条件
            return True

        for precond in action.preconditions:
            if "reachable" in precond:
                obj = precond.split('(')[1].split(')')[0]
                # 检查对象是否在机器人当前位置可及范围内
                if state["robot_state"]["location"] != self._get_object_location(obj, state):
                    return False

            elif "graspable" in precond:
                # 简化实现：所有对象默认可抓取
                pass

            elif "hand_empty" in precond:
                if not state["robot_state"]["hand_empty"]:
                    return False

            elif "holding" in precond:
                obj = precond.split('(')[1].split(')')[0]
                if state["robot_state"]["holding"] != obj:
                    return False

            elif "navigable" in precond:
                # 简化实现：所有位置默认可导航
                pass

            elif "near" in precond:
                target = precond.split('(')[1].split(')')[0]
                # 检查机器人是否在目标附近
                if "agent" in target:
                    agent_loc = state["agents"].get(target, {}).get("location", "unknown")
                    if state["robot_state"]["location"] != agent_loc:
                        return False
                else:
                    obj_loc = self._get_object_location(target, state)
                    if state["robot_state"]["location"] != obj_loc:
                        return False

        return True

    def _get_object_location(self, obj: str, state: Dict) -> str:
        """获取对象的位置"""
        # 直接查询对象属性
        if obj in state["objects"] and "location" in state["objects"][obj]:
            return state["objects"][obj]["location"]

        # 在位置列表中查找
        for loc, objects in state["locations"].items():
            if obj in objects:
                return loc

        return "unknown"

    def _apply_effects(self, action: Action, state: Dict) -> Dict:
        """应用动作效果到状态"""
        # 创建状态的深拷贝以避免修改原始状态
        new_state = {
            "objects": {k: v.copy() for k, v in state["objects"].items()},
            "locations": {k: v.copy() for k, v in state["locations"].items()},
            "agents": {k: v.copy() for k, v in state["agents"].items()},
            "robot_state": state["robot_state"].copy()
        }

        for effect in action.effects:
            if "holding" in effect:
                obj = effect.split('(')[1].split(')')[0]
                new_state["robot_state"]["holding"] = obj
                new_state["robot_state"]["hand_empty"] = False

                # 更新对象位置
                obj_loc = self._get_object_location(obj, new_state)
                if obj_loc != "unknown" and obj in new_state["locations"].get(obj_loc, []):
                    new_state["locations"][obj_loc].remove(obj)

            elif "hand_empty" in effect:
                new_state["robot_state"]["holding"] = None
                new_state["robot_state"]["hand_empty"] = True

            elif "at" in effect:
                loc = effect.split('(')[1].split(')')[0]
                new_state["robot_state"]["location"] = loc

            elif "on" in effect:
                parts = effect.split('(')[1].split(')')[0].split(', ')
                obj, target = parts[0], parts[1]

                # 更新对象位置
                if obj in new_state["objects"]:
                    new_state["objects"][obj]["location"] = target

                # 更新位置列表
                for loc, objects in new_state["locations"].items():
                    if obj in objects:
                        objects.remove(obj)

                if target not in new_state["locations"]:
                    new_state["locations"][target] = []
                new_state["locations"][target].append(obj)

            elif "clean" in effect:
                obj = effect.split('(')[1].split(')')[0]
                if obj in new_state["objects"]:
                    new_state["objects"][obj]["clean"] = True

            elif effect.startswith("not "):
                # 处理否定效果，如"not holding(object)"
                neg_effect = effect[4:]  # 去除"not "前缀

                if "holding" in neg_effect:
                    obj = neg_effect.split('(')[1].split(')')[0]
                    if new_state["robot_state"]["holding"] == obj:
                        new_state["robot_state"]["holding"] = None

            elif "has" in effect:
                parts = effect.split('(')[1].split(')')[0].split(', ')
                agent, obj = parts[0], parts[1]

                # 更新代理状态
                if agent in new_state["agents"]:
                    if "holding" not in new_state["agents"][agent]:
                        new_state["agents"][agent]["holding"] = []
                    new_state["agents"][agent]["holding"].append(obj)

                # 更新对象位置
                obj_loc = self._get_object_location(obj, new_state)
                if obj_loc != "unknown" and obj in new_state["locations"].get(obj_loc, []):
                    new_state["locations"][obj_loc].remove(obj)

        return new_state

    def _generate_prerequisite_actions(self, action: Action, state: Dict) -> List[Action]:
        """生成满足前提条件的先决动作"""
        prerequisites = []

        if action.agent_id != "robot":
            return prerequisites  # 不为人类动作生成先决条件

        for precond in action.preconditions:
            if "reachable" in precond:
                obj = precond.split('(')[1].split(')')[0]
                obj_loc = self._get_object_location(obj, state)

                if obj_loc != "unknown" and state["robot_state"]["location"] != obj_loc:
                    # 添加移动动作
                    move_action = Action(
                        agent_id="robot",
                        action_type="move",
                        targets=[obj_loc],
                        parameters={"speed": 0.5},
                        preconditions=[],  # 避免递归
                        effects=[f"at({obj_loc})"],
                        duration=2.0
                    )
                    prerequisites.append(move_action)

            elif "hand_empty" in precond and not state["robot_state"]["hand_empty"]:
                # 如果手不空，需要先放下物体
                holding_obj = state["robot_state"]["holding"]
                if holding_obj:
                    place_action = Action(
                        agent_id="robot",
                        action_type="place",
                        targets=[holding_obj, "table"],  # 默认放在桌子上
                        parameters={},
                        preconditions=[],  # 避免递归
                        effects=[f"on({holding_obj}, table)", "hand_empty()"],
                        duration=1.5
                    )
                    prerequisites.append(place_action)

            elif "near" in precond:
                target = precond.split('(')[1].split(')')[0]
                target_loc = ""

                if "agent" in target and target in state["agents"]:
                    target_loc = state["agents"][target].get("location", "unknown")
                else:
                    target_loc = self._get_object_location(target, state)

                if target_loc != "unknown" and state["robot_state"]["location"] != target_loc:
                    # 添加移动动作
                    move_action = Action(
                        agent_id="robot",
                        action_type="move",
                        targets=[target_loc],
                        parameters={"speed": 0.5},
                        preconditions=[],  # 避免递归
                        effects=[f"at({target_loc})"],
                        duration=2.0
                    )
                    prerequisites.append(move_action)

        return prerequisites

    def _add_recovery_strategies(self, plan: List[Action], task_goal: TaskGoal) -> List[Action]:
        """添加自我反思和错误恢复策略"""
        # 在实际系统中，这将添加条件检查和恢复动作
        # 此处简化实现，仅添加计划完成后的验证动作

        final_plan = plan.copy()

        # 添加验证动作
        if task_goal.name:
            check_action = Action(
                agent_id="robot",
                action_type="check",
                targets=["task_completion"],
                parameters={"task_name": task_goal.name},
                preconditions=[],
                effects=[],
                duration=1.0,
                confidence=0.9
            )
            final_plan.append(check_action)

        return final_plan


# =====================================================================
# 5. 环境接口
# =====================================================================

class VirtualEnvironmentInterface:
    """虚拟协作环境接口"""

    def __init__(self, env_config: Dict = None):
        """初始化环境接口

        Args:
            env_config: 环境配置
        """
        self.env_config = env_config or {}
        self.current_state = None
        self.action_history = []
        self.last_observation = None

        # 初始化环境连接
        self._initialize_environment()

    def _initialize_environment(self):
        """初始化与虚拟环境的连接"""
        print("初始化VirtualHome-Social环境...")
        # 实际实现会与Unity或其他环境引擎建立连接

        # 加载环境配置
        self._load_environment_config()

        # 创建模拟环境状态
        self._create_initial_state()

    def _load_environment_config(self):
        """加载环境配置"""
        # 实际系统中，这将从配置文件加载
        if not self.env_config:
            # 默认环境配置
            self.env_config = {
                "scene": "apartment_1",
                "agents": ["robot", "human"],
                "objects": [
                    {"id": "cup_1", "type": "cup", "location": "table", "clean": False},
                    {"id": "cup_2", "type": "cup", "location": "table", "clean": False},
                    {"id": "plate_1", "type": "plate", "location": "table", "clean": False},
                    {"id": "sponge_1", "type": "sponge", "location": "sink", "clean": True},
                    {"id": "cloth_1", "type": "cloth", "location": "counter", "clean": True}
                ],
                "locations": ["table", "sink", "counter", "cabinet", "living_room", "kitchen"]
            }

    def _create_initial_state(self):
        """创建初始环境状态"""
        objects = {}
        for obj in self.env_config.get("objects", []):
            objects[obj["id"]] = {k: v for k, v in obj.items() if k != "id"}

        agents = {}
        for agent_id in self.env_config.get("agents", []):
            if agent_id == "robot":
                agents[agent_id] = {
                    "location": "kitchen",
                    "holding": None
                }
            else:
                agents[agent_id] = {
                    "location": "living_room",
                    "holding": None
                }

        relations = []
        # 添加一些基本关系
        for obj in self.env_config.get("objects", []):
            if "location" in obj:
                relations.append({
                    "subject": obj["id"],
                    "relation": "on",
                    "object": obj["location"]
                })

        self.current_state = EnvironmentState(
            objects=objects,
            agents=agents,
            relations=relations,
            timestamp=0.0
        )

        # 初始化动作历史
        self.action_history = []

    def reset(self):
        """重置环境到初始状态"""
        self._create_initial_state()
        return self.get_observation()

    def step(self, action: Action) -> Dict:
        """执行动作并更新环境状态

        Args:
            action: 要执行的动作

        Returns:
            观察结果
        """
        # 添加到动作历史
        self.action_history.append(self._action_to_dict(action))

        # 更新环境状态
        self._update_environment_state(action)

        # 获取新的观察
        observation = self.get_observation()
        self.last_observation = observation

        return observation

    def _action_to_dict(self, action: Action) -> Dict:
        """将Action对象转换为字典表示"""
        return {
            "agent": action.agent_id,
            "action": action.action_type,
            "targets": action.targets,
            "parameters": action.parameters,
            "time": self.current_state.timestamp
        }

    def _update_environment_state(self, action: Action):
        """根据动作更新环境状态"""
        # 更新时间戳
        self.current_state.timestamp += action.duration

        # 更新代理状态
        agent = self.current_state.agents.get(action.agent_id, {})

        # 根据动作类型更新状态
        if action.action_type == "move" and action.targets:
            target_location = action.targets[0]
            agent["location"] = target_location

        elif action.action_type == "pick" and action.targets:
            target_object = action.targets[0]
            if target_object in self.current_state.objects:
                agent["holding"] = target_object

                # 更新关系
                self._remove_relation(target_object, "on", None)
                self._add_relation(target_object, "held_by", action.agent_id)

        elif action.action_type == "place" and len(action.targets) >= 2:
            obj, target = action.targets[0], action.targets[1]
            if agent.get("holding") == obj:
                agent["holding"] = None

                # 更新对象位置
                if obj in self.current_state.objects:
                    self.current_state.objects[obj]["location"] = target

                # 更新关系
                self._remove_relation(obj, "held_by", action.agent_id)
                self._add_relation(obj, "on", target)

        elif action.action_type == "handover" and len(action.targets) >= 2:
            obj, recipient = action.targets[0], action.targets[1]

            if agent.get("holding") == obj:
                agent["holding"] = None

                # 更新接收者状态
                if recipient in self.current_state.agents:
                    recipient_agent = self.current_state.agents[recipient]
                    if "holding" not in recipient_agent:
                        recipient_agent["holding"] = []
                    elif not isinstance(recipient_agent["holding"], list):
                        recipient_agent["holding"] = [recipient_agent["holding"]]
                    recipient_agent["holding"].append(obj)

                # 更新关系
                self._remove_relation(obj, "held_by", action.agent_id)
                self._add_relation(obj, "held_by", recipient)

        elif action.action_type == "clean" and action.targets:
            target_object = action.targets[0]
            if target_object in self.current_state.objects:
                self.current_state.objects[target_object]["clean"] = True

        # 保存更新后的代理状态
        self.current_state.agents[action.agent_id] = agent

    def _add_relation(self, subject: str, relation: str, object_: str):
        """添加关系"""
        self.current_state.relations.append({
            "subject": subject,
            "relation": relation,
            "object": object_
        })

    def _remove_relation(self, subject: str, relation: str, object_: str):
        """移除关系"""
        # 如果object_为None，移除所有匹配主体和关系的关系
        self.current_state.relations = [
            r for r in self.current_state.relations
            if not (r["subject"] == subject and r["relation"] == relation and
                   (object_ is None or r["object"] == object_))
        ]

    def get_observation(self) -> Dict:
        """获取当前环境的观察结果"""
        # 在实际系统中，这将从环境引擎获取
        # 此处返回简化的观察结果

        return {
            "state": self.current_state,
            "visible_objects": self._get_visible_objects(),
            "agent_locations": {
                agent_id: info.get("location", "unknown")
                for agent_id, info in self.current_state.agents.items()
            },
            "timestamp": self.current_state.timestamp
        }

    def _get_visible_objects(self) -> List[str]:
        """获取当前可见的对象"""
        visible_objects = []

        # 简化实现：所有对象都可见
        for obj_id in self.current_state.objects:
            visible_objects.append(obj_id)

        return visible_objects

    def render(self, mode: str = "text") -> Union[str, None]:
        """渲染环境状态

        Args:
            mode: 渲染模式，'text'或'visual'

        Returns:
            渲染结果
        """
        if mode == "text":
            return self._render_text()
        elif mode == "visual":
            return self._render_visual()
        else:
            return None

    def _render_text(self) -> str:
        """文本方式渲染环境"""
        if not self.current_state:
            return "环境未初始化"

        text = "=== 虚拟环境状态 ===\n"
        text += f"时间: {self.current_state.timestamp:.1f}s\n\n"

        # 渲染代理状态
        text += "代理:\n"
        for agent_id, info in self.current_state.agents.items():
            text += f"- {agent_id} 位于 {info.get('location', '未知位置')}"
            if info.get("holding"):
                if isinstance(info["holding"], list):
                    text += f", 手持: {', '.join(info['holding'])}"
                else:
                    text += f", 手持: {info['holding']}"
            text += "\n"

        # 渲染对象状态
        text += "\n对象:\n"
        for obj_id, attrs in self.current_state.objects.items():
            text += f"- {obj_id}: "

            attr_strs = []
            for attr, value in attrs.items():
                if attr == "location":
                    attr_strs.append(f"位于{value}")
                elif attr == "clean" and value:
                    attr_strs.append("干净")
                elif attr == "clean" and not value:
                    attr_strs.append("脏")
                else:
                    attr_strs.append(f"{attr}={value}")

            text += ", ".join(attr_strs)
            text += "\n"

        # 渲染最近的动作
        if self.action_history:
            text += "\n最近动作:\n"
            for i, action in enumerate(self.action_history[-3:]):
                text += f"{len(self.action_history)-3+i+1}. {action['agent']} "
                text += f"执行 {action['action']} "
                if action['targets']:
                    text += f"对象: {', '.join(action['targets'])}"
                text += "\n"

        return text

    def _render_visual(self) -> None:
        """视觉方式渲染环境"""
        print("视觉渲染未实现，请使用实际Unity接口")
        # 实际系统中，这将调用Unity或其他引擎的渲染功能

        return None


# =====================================================================
# 6. 集成系统
# =====================================================================

class ToMCollaborationSystem:
    """基于心理理论的LLM协作系统"""

    def __init__(self, config: Dict = None):
        """初始化协作系统

        Args:
            config: 系统配置
        """
        self.config = config or self._default_config()

        # 初始化组件
        self._initialize_components()

        # 状态跟踪
        self.current_task = None
        self.agent_beliefs = {}
        self.action_history = []
        self.task_status = "idle"  # idle, in_progress, completed, failed

    def _default_config(self) -> Dict:
        """默认系统配置"""
        return {
            # 使用可选的实际Hugging Face模型ID，或保留为None以使用模拟模型
            "perception_model": None,  # 例如 "openai/clip-vit-base-patch32"
            "tom_model": None,  # 例如 "microsoft/deberta-v3-base"
            "planning_model": None,  # 例如 "gpt2"
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "robot_id": "robot",
            "env_config": None,
            "logging_enabled": True,
            "use_mock_models": True  # 是否使用模拟模型
        }

    # 在tom_llm_system.py文件中修改以下部分
    # 在tom_llm_system.py文件中修改以下部分

    def _initialize_components(self):
        """初始化系统组件"""
        print("初始化ToM协作系统组件...")

        try:
            # 尝试从Hugging Face加载模型
            # 初始化感知模块
            self.perception_module = PerceptionModule(
                model_path=self.config["perception_model"],
                device=self.config["device"]
            )

            # 初始化ToM模块
            self.tom_module = TheoryOfMindModule(
                model_path=self.config["tom_model"],
                device=self.config["device"]
            )

            # 初始化规划模块
            self.planning_module = CollaborationPlanningModule(
                model_path=self.config["planning_model"],
                device=self.config["device"]
            )
        except (OSError, ValueError) as e:
            print(f"加载预训练模型失败: {str(e)}")
            print("使用模拟模型进行初始化...")

            # 导入模拟模型
            try:
                from mock_models import create_mock_model_and_tokenizer

                # 使用模拟模型初始化各模块
                self._init_with_mock_models()
            except ImportError:
                print("未找到mock_models.py，创建简化版模块...")
                # 创建简化版模块
                self._init_with_simplified_modules()

        # 初始化环境接口
        self.environment = VirtualEnvironmentInterface(
            env_config=self.config.get("env_config")
        )

        print("系统初始化完成")

    def _init_with_mock_models(self):
        """使用模拟模型初始化模块"""
        from mock_models import create_mock_model_and_tokenizer

        # 创建感知模块的模拟模型
        perception_model, perception_tokenizer = create_mock_model_and_tokenizer("perception")
        self.perception_module = PerceptionModule.__new__(PerceptionModule)
        self.perception_module.__init__ = lambda *args, **kwargs: None
        self.perception_module.model = perception_model
        self.perception_module.tokenizer = perception_tokenizer
        self.perception_module.action_recognizer = self.perception_module._initialize_action_recognizer = lambda *args, **kwargs: {
            "actions": ["pick", "move", "place"], "confidence": 0.9}
        self.perception_module.goal_extractor = self.perception_module._initialize_goal_extractor = lambda *args, **kwargs: {
            "goal": "清理桌子", "steps": ["收集物品", "擦拭表面", "整理物品"]}
        self.perception_module.process_demonstration = self._mock_process_demonstration

        # 创建ToM模块的模拟模型
        tom_model, tom_tokenizer = create_mock_model_and_tokenizer("tom")
        self.tom_module = TheoryOfMindModule.__new__(TheoryOfMindModule)
        self.tom_module.__init__ = lambda *args, **kwargs: None
        self.tom_module.model = tom_model
        self.tom_module.tokenizer = tom_tokenizer
        self.tom_module.belief_history = {}
        self.tom_module.learning_rate = 0.05
        self.tom_module.infer_agent_belief = self._mock_infer_agent_belief

        # 创建规划模块的模拟模型
        planning_model, planning_tokenizer = create_mock_model_and_tokenizer("planning")
        self.planning_module = CollaborationPlanningModule.__new__(CollaborationPlanningModule)
        self.planning_module.__init__ = lambda *args, **kwargs: None
        self.planning_module.model = planning_model
        self.planning_module.tokenizer = planning_tokenizer
        self.planning_module.action_library = self.planning_module._initialize_action_library = lambda: {
            "pick": {"parameters": ["object"], "preconditions": [], "effects": []},
            "place": {"parameters": ["object", "target"], "preconditions": [], "effects": []},
            "move": {"parameters": ["location"], "preconditions": [], "effects": []},
            "handover": {"parameters": ["object", "agent"], "preconditions": [], "effects": []},
            "clean": {"parameters": ["object"], "preconditions": [], "effects": []}
        }
        self.planning_module.max_search_depth = 5
        self.planning_module.max_search_iterations = 100
        self.planning_module.exploration_constant = 1.41
        self.planning_module.generate_collaboration_plan = self._mock_generate_collaboration_plan

    def _init_with_simplified_modules(self):
        """使用简化版模块初始化系统"""
        # 创建简化版感知模块
        self.perception_module = type('SimplifiedPerceptionModule', (), {
            'process_demonstration': self._mock_process_demonstration
        })()

        # 创建简化版ToM模块
        self.tom_module = type('SimplifiedToMModule', (), {
            'infer_agent_belief': self._mock_infer_agent_belief,
            'belief_history': {},
            'update_belief_confidence': lambda *args, **kwargs: None
        })()

        # 创建简化版规划模块
        self.planning_module = type('SimplifiedPlanningModule', (), {
            'generate_collaboration_plan': self._mock_generate_collaboration_plan
        })()

    def _mock_process_demonstration(self, demo_data, env_context):
        """模拟演示处理"""
        print("模拟演示处理...")

        # 创建一个基本的任务目标
        if isinstance(demo_data, list) and len(demo_data) > 0:
            # 根据演示判断任务类型
            has_cup = any("cup" in str(action.get("object", "")) for action in demo_data)
            has_sink = any("sink" in str(action.get("target", "")) for action in demo_data)

            if has_cup and has_sink:
                task_name = "清理餐具"
                task_steps = ["收集餐桌上的脏餐具", "将脏餐具放入水槽", "清洗餐具", "将干净餐具放回架子"]
                target_state = {"cup_1": {"clean": True}, "plate_1": {"clean": True}}
            else:
                task_name = "整理房间"
                task_steps = ["收集散落的物品", "将物品归类", "放回原位"]
                target_state = {"room": {"clean": True}}
        else:
            task_name = "一般清理任务"
            task_steps = ["收集物品", "整理物品", "清洁表面"]
            target_state = {"table": {"clean": True}}

        return TaskGoal(
            name=task_name,
            target_state=target_state,
            constraints=["小心处理易碎物品"],
            decomposition=task_steps,
            priority=1.0
        )

    def _mock_infer_agent_belief(self, agent_id, env_state, actions_history, task_context=None):
        """模拟代理信念推断"""
        print(f"模拟推断代理 {agent_id} 的信念...")

        # 创建一个基本的信念模型
        return AgentBelief(
            agent_id=agent_id,
            knowledge={"物品位置": 0.9, "任务目标": 0.8, "环境布局": 0.7},
            goals=[{"清理": 0.9}, {"帮助": 0.7}],
            attention={"餐具": 0.8, "水槽": 0.6, "桌子": 0.7},
            confidence=0.8
        )

    def _mock_generate_collaboration_plan(self, task_goal, env_state, agent_beliefs, robot_id="robot"):
        """模拟协作计划生成"""
        print("模拟生成协作计划...")

        # 创建一个基本的动作计划
        if "清理餐具" in task_goal.name:
            return [
                Action(
                    agent_id=robot_id,
                    action_type="move",
                    targets=["table"],
                    parameters={"speed": 0.5},
                    preconditions=[],
                    effects=["at(table)"],
                    duration=2.0
                ),
                Action(
                    agent_id=robot_id,
                    action_type="pick",
                    targets=["cup_1"],
                    parameters={},
                    preconditions=[],
                    effects=["holding(cup_1)"],
                    duration=1.0
                ),
                Action(
                    agent_id=robot_id,
                    action_type="move",
                    targets=["sink"],
                    parameters={"speed": 0.5},
                    preconditions=[],
                    effects=["at(sink)"],
                    duration=2.0
                ),
                Action(
                    agent_id=robot_id,
                    action_type="place",
                    targets=["cup_1", "sink"],
                    parameters={},
                    preconditions=[],
                    effects=["on(cup_1, sink)", "hand_empty()"],
                    duration=1.0
                ),
                Action(
                    agent_id="human",
                    action_type="clean",
                    targets=["cup_1"],
                    parameters={},
                    preconditions=[],
                    effects=["clean(cup_1)"],
                    duration=3.0
                )
            ]
        else:
            return [
                Action(
                    agent_id=robot_id,
                    action_type="move",
                    targets=["center"],
                    parameters={"speed": 0.5},
                    preconditions=[],
                    effects=["at(center)"],
                    duration=2.0
                ),
                Action(
                    agent_id=robot_id,
                    action_type="pick",
                    targets=["item_1"],
                    parameters={},
                    preconditions=[],
                    effects=["holding(item_1)"],
                    duration=1.0
                ),
                Action(
                    agent_id=robot_id,
                    action_type="place",
                    targets=["item_1", "shelf"],
                    parameters={},
                    preconditions=[],
                    effects=["on(item_1, shelf)", "hand_empty()"],
                    duration=1.0
                ),
                Action(
                    agent_id="human",
                    action_type="check",
                    targets=["room"],
                    parameters={},
                    preconditions=[],
                    effects=["verified(room)"],
                    duration=2.0
                )
            ]

    # mock_models.py
    """
    模拟模型实现，用于在没有实际预训练模型的情况下进行系统测试
    """

    import torch
    from torch import nn
    import os
    from typing import Dict, List, Optional, Union, Any

    class MockModel(nn.Module):
        """模拟的预训练模型"""

        def __init__(self, config=None):
            super().__init__()
            self.config = config or {"model_type": "mock", "hidden_size": 768, "vocab_size": 30000}
            self.embeddings = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
            self.encoder = nn.Linear(self.config["hidden_size"], self.config["hidden_size"])
            self.decoder = nn.Linear(self.config["hidden_size"], self.config["hidden_size"])

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if input_ids is None:
                input_ids = torch.zeros(1, 10).long()

            embeddings = self.embeddings(input_ids)
            hidden_states = self.encoder(embeddings)
            outputs = self.decoder(hidden_states)

            return {
                "last_hidden_state": outputs,
                "hidden_states": [outputs] * 4  # 模拟多层输出
            }

        def to(self, device):
            """模拟设备转移"""
            return self

    class MockTokenizer:
        """模拟的分词器"""

        def __init__(self, config=None):
            self.config = config or {"model_type": "mock", "vocab_size": 30000}
            self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
            for i in range(5, self.config["vocab_size"]):
                self.vocab[f"token_{i}"] = i

        def encode(self, text, add_special_tokens=True, **kwargs):
            """模拟编码过程"""
            # 简单地将文本长度作为标记数量
            tokens = [2] if add_special_tokens else []  # [CLS]
            tokens.extend([1] * min(len(text.split()), 50))  # 用[UNK]填充
            if add_special_tokens:
                tokens.append(3)  # [SEP]
            return tokens

        def decode(self, token_ids, skip_special_tokens=True, **kwargs):
            """模拟解码过程"""
            if skip_special_tokens:
                token_ids = [t for t in token_ids if t not in [0, 2, 3, 4]]
            return " ".join([f"<{t}>" for t in token_ids])

        def __call__(self, text, padding=True, truncation=True, return_tensors=None, **kwargs):
            """模拟分词器调用"""
            if isinstance(text, str):
                tokens = self.encode(text)
            else:
                tokens = [self.encode(t) for t in text]
                max_len = max(len(t) for t in tokens)
                if padding:
                    tokens = [t + [0] * (max_len - len(t)) for t in tokens]

            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(tokens),
                    "attention_mask": torch.ones_like(torch.tensor(tokens))
                }
            else:
                return {
                    "input_ids": tokens,
                    "attention_mask": [[1] * len(t) for t in tokens]
                }

    def create_mock_model_and_tokenizer(model_type="base"):
        """创建指定类型的模拟模型和分词器"""
        if model_type == "perception":
            config = {"model_type": "vision-language", "hidden_size": 1024, "vocab_size": 50000}
        elif model_type == "tom":
            config = {"model_type": "language", "hidden_size": 1536, "vocab_size": 30000}
        elif model_type == "planning":
            config = {"model_type": "language", "hidden_size": 1024, "vocab_size": 30000}
        else:
            config = {"model_type": "base", "hidden_size": 768, "vocab_size": 30000}

        model = MockModel(config)
        tokenizer = MockTokenizer(config)

        return model, tokenizer

    def observe_and_infer_task(self, demo_data: Union[str, List]) -> TaskGoal:
        """观察演示并推断任务

        Args:
            demo_data: 演示数据(视频路径或动作序列)

        Returns:
            推断的任务目标
        """
        print("观察演示并推断任务...")

        # 获取当前环境状态
        env_state = self.environment.current_state

        # 使用感知模块处理演示
        task_goal = self.perception_module.process_demonstration(
            demo_data=demo_data,
            env_context=env_state
        )

        # 保存当前任务
        self.current_task = task_goal
        self.task_status = "inferred"

        print(f"任务推断完成: {task_goal.name}")
        return task_goal

    def infer_agent_beliefs(self) -> Dict[str, AgentBelief]:
        """推断所有代理的信念状态"""
        print("推断代理信念状态...")

        # 获取当前环境状态
        env_state = self.environment.current_state

        # 为每个代理(除了机器人)推断信念
        for agent_id in env_state.agents:
            if agent_id != self.config["robot_id"]:
                belief = self.tom_module.infer_agent_belief(
                    agent_id=agent_id,
                    env_state=env_state,
                    actions_history=self.action_history,
                    task_context=self.current_task
                )

                self.agent_beliefs[agent_id] = belief
                print(f"已推断代理 {agent_id} 的信念状态")

        return self.agent_beliefs

    def plan_collaboration(self) -> List[Action]:
        """生成协作计划"""
        if not self.current_task:
            raise ValueError("未定义任务目标，请先调用observe_and_infer_task")

        print("生成协作计划...")

        # 获取当前环境状态
        env_state = self.environment.current_state

        # 如果还没有推断代理信念，先推断
        if not self.agent_beliefs:
            self.infer_agent_beliefs()

        # 使用规划模块生成计划
        action_plan = self.planning_module.generate_collaboration_plan(
            task_goal=self.current_task,
            env_state=env_state,
            agent_beliefs=self.agent_beliefs,
            robot_id=self.config["robot_id"]
        )

        self.task_status = "planned"
        print(f"生成了 {len(action_plan)} 个动作的协作计划")

        return action_plan

    def execute_plan(self, action_plan: List[Action], max_steps: int = 100) -> str:
        """执行协作计划

        Args:
            action_plan: 要执行的动作计划
            max_steps: 最大执行步数

        Returns:
            执行结果状态
        """
        if not action_plan:
            return "no_plan"

        print("开始执行协作计划...")
        self.task_status = "in_progress"

        step_count = 0
        action_index = 0

        while action_index < len(action_plan) and step_count < max_steps:
            current_action = action_plan[action_index]

            # 执行动作
            print(f"执行动作: {current_action.agent_id} {current_action.action_type} {current_action.targets}")

            # 如果是机器人动作，直接执行
            if current_action.agent_id == self.config["robot_id"]:
                observation = self.environment.step(current_action)
                self.action_history.append(self.environment._action_to_dict(current_action))

            # 如果是人类动作，需要等待或模拟
            else:
                # 在实际系统中，这将等待人类执行
                # 在模拟中，我们直接执行预测的人类动作
                print(f"等待/模拟 {current_action.agent_id} 的动作...")
                observation = self.environment.step(current_action)
                self.action_history.append(self.environment._action_to_dict(current_action))

            # 更新信念模型
            self._update_beliefs_after_action(current_action, observation)

            # 显示当前环境状态
            if self.config["logging_enabled"]:
                print(self.environment.render("text"))

            # 前进到下一个动作
            action_index += 1
            step_count += 1

            # 检查任务是否完成
            if self._check_task_completion():
                self.task_status = "completed"
                print("任务成功完成!")
                break

        # 如果所有动作执行完但任务未完成
        if action_index >= len(action_plan) and self.task_status != "completed":
            if self._check_task_completion():
                self.task_status = "completed"
                print("任务成功完成!")
            else:
                self.task_status = "incomplete"
                print("计划执行完毕，但任务未完成")

        # 如果超过最大步数
        if step_count >= max_steps and action_index < len(action_plan):
            self.task_status = "timeout"
            print("执行超时")

        return self.task_status

    def _update_beliefs_after_action(self, action: Action, observation: Dict):
        """动作执行后更新信念模型"""
        # 提取环境状态
        env_state = observation["state"]

        # 对于人类动作，更新信念模型的准确性
        if action.agent_id != self.config["robot_id"]:
            # 评估信念预测的准确性
            prediction_feedback = self._evaluate_belief_prediction(action, env_state)

            # 更新信念模型
            self.tom_module.update_belief_confidence(
                agent_id=action.agent_id,
                feedback=prediction_feedback
            )

        # 如果环境发生重大变化，重新推断所有代理的信念
        if self._check_significant_change(observation):
            self.infer_agent_beliefs()

    def _evaluate_belief_prediction(self, action: Action, env_state: EnvironmentState) -> Dict:
        """评估信念预测的准确性"""
        # 简化实现，返回固定反馈
        # 实际系统应该比较预测和实际行为

        return {
            "knowledge": {"物品位置": 0.8, "任务目标": 0.9},
            "goals": [{"清理房间": 0.85}],
            "overall": 0.85
        }

    def _check_significant_change(self, observation: Dict) -> bool:
        """检查环境是否发生重大变化"""
        # 简化实现
        # 实际系统应比较当前和之前的状态
        return False

    def _check_task_completion(self) -> bool:
        """检查当前任务是否完成"""
        if not self.current_task:
            return False

        # 获取当前环境状态
        env_state = self.environment.current_state

        # 检查目标状态是否达成
        for obj_id, target_attrs in self.current_task.target_state.items():
            if obj_id not in env_state.objects:
                return False

            current_attrs = env_state.objects[obj_id]
            for attr, target_value in target_attrs.items():
                if attr not in current_attrs or current_attrs[attr] != target_value:
                    return False

        return True

    def reset(self):
        """重置系统状态"""
        # 重置环境
        self.environment.reset()

        # 重置系统状态
        self.current_task = None
        self.agent_beliefs = {}
        self.action_history = []
        self.task_status = "idle"

        print("系统已重置")

        return self.environment.get_observation()

    def run_complete_demo(self, demo_data: Union[str, List]) -> Dict:
        """运行完整的演示-推理-执行流程

        Args:
            demo_data: 演示数据

        Returns:
            运行结果
        """
        results = {}

        try:
            # 重置系统
            self.reset()

            # 1. 观察演示并推断任务
            task_goal = self.observe_and_infer_task(demo_data)
            results["task_goal"] = task_goal

            # 2. 推断代理信念
            agent_beliefs = self.infer_agent_beliefs()
            results["agent_beliefs"] = agent_beliefs

            # 3. 生成协作计划
            action_plan = self.plan_collaboration()
            results["action_plan"] = action_plan

            # 4. 执行计划
            execution_status = self.execute_plan(action_plan)
            results["execution_status"] = execution_status

            # 5. 收集最终状态
            final_observation = self.environment.get_observation()
            results["final_state"] = final_observation

            return results

        except Exception as e:
            print(f"运行过程中出错: {str(e)}")
            results["error"] = str(e)
            return results


# =====================================================================
# 7. 实验与评估模块
# =====================================================================

class ExperimentModule:
    """实验与评估模块"""

    def __init__(self):
        """初始化实验模块"""
        self.results = {}
        self.metrics = {}
        self.baselines = {}

    def run_experiment(self,
                      system: ToMCollaborationSystem,
                      demos: List[Union[str, List]],
                      experiment_name: str = "default_experiment") -> Dict:
        """运行实验

        Args:
            system: 协作系统实例
            demos: 演示数据列表
            experiment_name: 实验名称

        Returns:
            实验结果
        """
        print(f"开始实验: {experiment_name}")

        experiment_results = {
            "name": experiment_name,
            "timestamp": time.time(),
            "demos": len(demos),
            "demo_results": [],
            "metrics": {}
        }

        # 运行每个演示
        for i, demo in enumerate(demos):
            print(f"运行演示 {i+1}/{len(demos)}")

            # 运行完整流程
            demo_result = system.run_complete_demo(demo)
            experiment_results["demo_results"].append(demo_result)

            # 计算指标
            demo_metrics = self._calculate_metrics(demo_result)
            for metric, value in demo_metrics.items():
                if metric not in experiment_results["metrics"]:
                    experiment_results["metrics"][metric] = []
                experiment_results["metrics"][metric].append(value)

        # 计算汇总指标
        for metric, values in experiment_results["metrics"].items():
            experiment_results["metrics"][metric] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "values": values
            }

        # 保存结果
        self.results[experiment_name] = experiment_results

        print(f"实验 {experiment_name} 完成")
        print(f"平均任务完成率: {experiment_results['metrics'].get('success_rate', {}).get('mean', 0):.2f}")
        print(f"平均任务完成时间: {experiment_results['metrics'].get('completion_time', {}).get('mean', 0):.2f}s")

        return experiment_results

    def _calculate_metrics(self, demo_result: Dict) -> Dict:
        """计算单个演示的评估指标"""
        metrics = {}

        # 任务状态
        execution_status = demo_result.get("execution_status", "unknown")
        metrics["success"] = 1.0 if execution_status == "completed" else 0.0

        # 任务完成时间
        if "final_state" in demo_result and "timestamp" in demo_result["final_state"]:
            metrics["completion_time"] = demo_result["final_state"]["timestamp"]
        else:
            metrics["completion_time"] = float('inf')

        # 动作效率
        if "action_plan" in demo_result:
            action_plan = demo_result["action_plan"]
            total_actions = len(action_plan)
            robot_actions = sum(1 for a in action_plan if a.agent_id == "robot")
            metrics["action_count"] = total_actions
            metrics["robot_action_ratio"] = robot_actions / total_actions if total_actions > 0 else 0

        # 信念模型评估
        if "agent_beliefs" in demo_result:
            avg_confidence = 0.0
            belief_count = 0

            for agent_id, belief in demo_result["agent_beliefs"].items():
                if hasattr(belief, "confidence"):
                    avg_confidence += belief.confidence
                    belief_count += 1
                elif isinstance(belief, dict) and "confidence" in belief:
                    avg_confidence += belief["confidence"]
                    belief_count += 1

            if belief_count > 0:
                metrics["belief_confidence"] = avg_confidence / belief_count

        return metrics

    def compare_results(self, result1: Dict, result2: Dict) -> Dict:
        """比较两个实验结果"""
        comparison = {
            "experiment1": result1["name"],
            "experiment2": result2["name"],
            "metrics_comparison": {},
            "improvement": {}
        }

        # 比较各项指标
        for metric in set(result1["metrics"].keys()) | set(result2["metrics"].keys()):
            if metric in result1["metrics"] and metric in result2["metrics"]:
                val1 = result1["metrics"][metric]["mean"]
                val2 = result2["metrics"][metric]["mean"]

                comparison["metrics_comparison"][metric] = {
                    "experiment1": val1,
                    "experiment2": val2,
                    "difference": val1 - val2,
                    "percent_change": (val1 - val2) / val2 * 100 if val2 != 0 else float('inf')
                }

                # 判断改进情况
                if metric in ["success", "belief_confidence"]:
                    # 这些指标越高越好
                    comparison["improvement"][metric] = val1 > val2
                elif metric in ["completion_time", "action_count"]:
                    # 这些指标越低越好
                    comparison["improvement"][metric] = val1 < val2

        # 总体改进得分
        improvement_score = sum(1 for improved in comparison["improvement"].values() if improved)
        comparison["overall_improvement_score"] = improvement_score
        comparison["overall_improvement_ratio"] = improvement_score / len(comparison["improvement"]) if comparison["improvement"] else 0

        return comparison

    def generate_report(self, comparison: Dict) -> str:
        """生成对比结果报告"""
        report = f"# 实验对比报告: {comparison['experiment1']} vs {comparison['experiment2']}\n\n"

        # 添加总体改进情况
        improvement_ratio = comparison["overall_improvement_ratio"] * 100
        report += f"## 总体改进情况\n\n"
        report += f"- 改进得分: {comparison['overall_improvement_score']}/{len(comparison['improvement'])}\n"
        report += f"- 改进率: {improvement_ratio:.2f}%\n\n"

        # 添加指标对比
        report += "## 指标对比\n\n"
        report += "| 指标 | " + comparison['experiment1'] + " | " + comparison['experiment2'] + " | 差异 | 改进 |\n"
        report += "|------|-------|-------|------|------|\n"

        for metric, data in comparison["metrics_comparison"].items():
            val1 = data["experiment1"]
            val2 = data["experiment2"]
            diff = data["difference"]
            improved = comparison["improvement"].get(metric, False)

            if metric in ["completion_time", "action_count"]:
                # 这些指标越低越好，负差异表示改进
                report += f"| {metric} | {val1:.2f} | {val2:.2f} | {diff:.2f} | {'✓' if improved else '✗'} |\n"
            else:
                # 这些指标越高越好，正差异表示改进
                report += f"| {metric} | {val1:.2f} | {val2:.2f} | {diff:.2f} | {'✓' if improved else '✗'} |\n"

        # 添加详细分析
        report += "\n## 详细分析\n\n"

        success_improved = comparison["improvement"].get("success", False)
        time_improved = comparison["improvement"].get("completion_time", False)

        if success_improved and time_improved:
            report += "系统在任务成功率和完成时间上都有提升，表明ToM推理显著改善了协作效率。\n\n"
        elif success_improved:
            report += "系统在任务成功率上有提升，但完成时间可能需要优化。\n\n"
        elif time_improved:
            report += "系统在完成时间上有提升，但成功率可能需要进一步改进。\n\n"
        else:
            report += "系统在主要指标上未见明显改进，可能需要重新评估ToM实现或调整协作策略。\n\n"

        # 添加样本级别分析
        report += "## 样本级别分析\n\n"
        report += "针对不同演示场景的表现对比：\n\n"

        # 这里只是示例，实际报告会更详细
        report += "1. 简单任务场景：ToM系统表现更佳\n"
        report += "2. 复杂任务场景：两个系统表现接近\n"
        report += "3. 误解场景：ToM系统显著优于基线\n\n"

        report += "## 结论与建议\n\n"
        if improvement_ratio > 60:
            report += "ToM-LLM系统显著优于基线系统，建议继续优化当前方法。\n"
        elif improvement_ratio > 30:
            report += "ToM-LLM系统优于基线系统，但仍有改进空间。建议关注信念模型的准确性提升。\n"
        else:
            report += "ToM-LLM系统与基线系统相比优势不明显，建议重新评估ToM实现方法或增强LLM模型能力。\n"

        return report


# =====================================================================
# 8. 自动化测试与消融实验工具
# =====================================================================

class AblationExperiment:
    """消融实验工具"""

    def __init__(self, base_config: Dict = None):
        """初始化消融实验

        Args:
            base_config: 基础配置
        """
        self.base_config = base_config or {}
        self.experiment_module = ExperimentModule()
        self.variants = {}

    def add_variant(self, name: str, config_changes: Dict):
        """添加系统变体

        Args:
            name: 变体名称
            config_changes: 相对于基础配置的变更
        """
        # 合并配置
        variant_config = self.base_config.copy()
        self._deep_update(variant_config, config_changes)

        self.variants[name] = {
            "name": name,
            "config": variant_config,
            "system": None,
            "results": None
        }

        print(f"添加系统变体: {name}")
        return self

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def run_ablation_study(self, demos: List, output_dir: str = 'ablation_results'):
        """运行消融实验

        Args:
            demos: 演示数据
            output_dir: 输出目录
        """
        print(f"开始消融实验，共 {len(self.variants)} 个变体")
        os.makedirs(output_dir, exist_ok=True)

        # 创建和运行每个变体
        for variant_name, variant in self.variants.items():
            print(f"\n运行变体: {variant_name}")

            # 创建系统实例
            variant["system"] = ToMCollaborationSystem(variant["config"])

            # 运行实验
            variant["results"] = self.experiment_module.run_experiment(
                system=variant["system"],
                demos=demos,
                experiment_name=variant_name
            )

            # 保存结果
            result_path = os.path.join(output_dir, f"{variant_name}_results.json")
            with open(result_path, 'w') as f:
                json.dump(variant["results"], f, indent=2)

            print(f"变体 {variant_name} 运行完成，结果保存至 {result_path}")

        # 比较所有变体
        self._compare_all_variants(output_dir)

    def _compare_all_variants(self, output_dir: str):
        """比较所有变体的结果"""
        if len(self.variants) < 2:
            print("至少需要2个变体才能进行比较")
            return

        print("\n比较所有变体结果")

        # 创建比较矩阵
        comparisons = {}
        variant_names = list(self.variants.keys())

        for i, name1 in enumerate(variant_names):
            for name2 in enumerate(variant_names[i+1:], i+1):
                variant1 = self.variants[name1]
                variant2 = self.variants[name2[1]]

                comparison = self.experiment_module.compare_results(
                    variant1["results"],
                    variant2["results"]
                )

                comparison_name = f"{name1}_vs_{name2[1]}"
                comparisons[comparison_name] = comparison

                # 保存比较结果
                comparison_path = os.path.join(output_dir, f"{comparison_name}.json")
                with open(comparison_path, 'w') as f:
                    json.dump(comparison, f, indent=2)

                # 生成报告
                report = self.experiment_module.generate_report(comparison)
                report_path = os.path.join(output_dir, f"{comparison_name}_report.md")
                with open(report_path, 'w') as f:
                    f.write(report)

        # 生成总体报告
        self._generate_overall_report(comparisons, output_dir)

    def _generate_overall_report(self, comparisons: Dict, output_dir: str):
        """生成总体报告"""
        report = "# 消融实验总体报告\n\n"

        # 添加变体描述
        report += "## 系统变体\n\n"
        for name, variant in self.variants.items():
            report += f"### {name}\n\n"
            # 提取关键配置差异
            config_diff = {}
            for key, value in variant["config"].items():
                if key not in self.base_config or self.base_config[key] != value:
                    config_diff[key] = value

            report += f"配置差异: {json.dumps(config_diff, indent=2)}\n\n"

        # 添加比较结果摘要
        report += "## 比较结果摘要\n\n"
        report += "| 比较 | 总体改进得分 | 改进率 | 主要优势 |\n"
        report += "|------|------------|-------|--------|\n"

        for name, comparison in comparisons.items():
            score = comparison["overall_improvement_score"]
            ratio = comparison["overall_improvement_ratio"] * 100

            # 找出最显著的改进指标
            best_metric = None
            best_improvement = 0
            for metric, data in comparison["metrics_comparison"].items():
                if metric in ["success", "belief_confidence"]:
                    improvement = data["difference"]
                elif metric in ["completion_time", "action_count"]:
                    improvement = -data["difference"]  # 注意反转，因为这些指标是越小越好

                if best_metric is None or improvement > best_improvement:
                    best_metric = metric
                    best_improvement = improvement

            report += f"| {name} | {score}/{len(comparison['improvement'])} | {ratio:.2f}% | {best_metric} |\n"

        # 添加结论
        report += "\n## 消融实验结论\n\n"
        report += "基于上述比较结果，我们可以得出以下结论：\n\n"

        # 在实际应用中，这里会是根据实验结果生成的具体结论
        report += "1. ToM模块对系统性能有显著影响\n"
        report += "2. LLM的质量对协作效果至关重要\n"
        report += "3. 混合规划方法优于纯规则或纯学习方法\n"

        # 保存总体报告
        report_path = os.path.join(output_dir, "overall_report.md")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"总体报告已保存至 {report_path}")


# =====================================================================
# 运行示例
# =====================================================================

def run_simple_demo():
    """运行简单演示"""
    # 创建系统
    system = ToMCollaborationSystem()

    # 创建简单演示数据
    demo_data = [
        {"agent": "human", "action": "pick", "object": "cup_1", "time": 0.5},
        {"agent": "human", "action": "move", "object": "cup_1", "target": "sink", "time": 1.2},
        {"agent": "human", "action": "place", "object": "cup_1", "target": "sink", "time": 2.0}
    ]

    # 运行完整流程
    results = system.run_complete_demo(demo_data)

    # 打印结果
    print("\n运行结果:")
    print(f"任务: {results['task_goal'].name}")
    print(f"执行状态: {results['execution_status']}")

    # 打印最终环境状态
    print("\n最终环境状态:")
    print(system.environment.render("text"))

    return results


if __name__ == "__main__":
    # 运行简单演示
    run_simple_demo()