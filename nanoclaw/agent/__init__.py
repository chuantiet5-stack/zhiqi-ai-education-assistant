"""Agent 包

聚合 Agent 核心组件并声明公共导出接口：
- ContextBuilder: 上下文构建器，负责组装 System Prompt 与 messages 列表
- AgentLoop: 核心推理循环，驱动 LLM 与工具的多轮交互
"""

from .context import ContextBuilder
from .loop import AgentLoop

# 显式声明公开 API，约束 from agent import * 的导出范围，
# 防止子模块等内部符号意外泄漏
__all__ = ["ContextBuilder", "AgentLoop"]