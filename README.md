# ToM-LLM 协作系统

这是一个基于心理理论（Theory of Mind, ToM）和大型语言模型（LLM）的机器人协作系统，受到Watch-And-Help项目的启发。系统能够通过观察人类行为演示，理解任务目标，并与人类协作完成任务。

## 安装指南

### 1. 环境要求
- Python 3.8+
- PyTorch 1.9+
- 其他依赖库

### 2. 安装步骤

1. 克隆仓库（或创建项目目录）
```bash
mkdir ToM_LLM
cd ToM_LLM
```

2. 创建并激活虚拟环境
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

3. 安装依赖
```bash
pip install torch numpy transformers
```

4. 创建必要的文件

将以下文件添加到项目中：
- `tom_llm_system.py` - 核心系统实现
- `mock_models.py` - 模拟模型实现（用于无网络环境下测试）
- `main.py` - 主程序入口

## 使用指南

### 1. 运行简单演示

```bash
python tom_llm_system.py
```

这将运行一个简单的演示，展示系统如何通过模拟模型执行协作任务。

### 2. 运行对比实验

```bash
python main.py --mode full --output results
```

这将运行一个完整的对比实验，比较带ToM和不带ToM的系统性能。

### 3. 使用真实预训练模型

如果要使用真实的预训练模型，请修改配置：

```python
config = {
    "perception_model": "openai/clip-vit-base-patch32",
    "tom_model": "microsoft/deberta-v3-base",
    "planning_model": "gpt2",
    "use_mock_models": False
}

system = ToMCollaborationSystem(config)
```

## 系统架构

系统包含以下主要模块：

1. **感知与目标推理模块**：从视频或动作序列中提取任务目标
2. **心理理论(ToM)模块**：推断人类意图、信念和知识状态
3. **协作规划模块**：基于任务目标和信念模型生成协作计划
4. **虚拟环境接口**：模拟环境状态并执行动作
5. **实验与评估模块**：进行对比实验和系统评估

## 故障排除

### 模型加载错误

如果遇到类似这样的错误：

```
OSError: model/perception_model is not a local folder and is not a valid model identifier
```

请确保：
1. 配置中使用了有效的HuggingFace模型ID，或者
2. 启用了模拟模型模式 (`"use_mock_models": True`)

### 执行错误

如果系统在执行过程中出现错误，请检查日志输出，并确保虚拟环境中安装了所有必要的依赖。