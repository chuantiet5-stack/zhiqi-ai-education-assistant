"""
上下文构建器模块

ContextBuilder 负责组装 Agent 运行所需的完整上下文，包括：
- 人设定义（从 identity.md 加载）
- 当前时间（让 Agent 感知时间）
- 工作区路径信息
- 长期记忆（从 MEMORY.md 加载，第六章预留接口）
- 历史对话和当前用户消息

最终输出符合 OpenAI messages 格式的列表，可直接传给 LLM。
"""

import os
from datetime import datetime
from typing import Any


class ContextBuilder:
    """
    上下文构建器

    统一管理 Agent 的上下文信息，生成完整的 messages 列表。

    人设文件（identity.md）定义 Agent 的角色、行为准则和能力边界。
    长期记忆文件（MEMORY.md）存储跨会话的知识和经验。

    使用示例：
        builder = ContextBuilder(workspace="/path/to/project")
        messages = builder.build_messages(
            history=[{"role": "user", "content": "之前的问题"}],
            current_message="帮我写个测试"
        )
    """

    # 默认人设：当 identity.md 不存在时使用
    DEFAULT_IDENTITY = """你是 NanoClaw，一个智能代码助手。

你的职责是帮助用户完成编程任务，包括：
- 编写、修改、调试代码
- 解释代码逻辑和技术概念
- 提供最佳实践建议

工作准则：
- 仔细理解用户需求后再行动
- 代码修改要谨慎，确保不破坏现有功能
- 主动说明你的操作和理由
- 遇到不确定的情况，先询问用户"""

    def __init__(
        self,
        workspace: str,
        identity_file: str = "identity.md",
        skills_summary: str = "",
    ) -> None:
        """
        Args:
            workspace: 工作区根目录的绝对路径
            identity_file: 人设文件名，相对于 workspace，默认 identity.md
            skills_summary: 技能摘要字符串，由 SkillsLoader 生成
        """
        self.workspace = os.path.abspath(workspace)
        self.identity_file = identity_file
        self.skills_summary = skills_summary

    def _load_identity(self) -> str:
        """
        加载人设文件内容

        从 workspace/{identity_file} 读取人设定义。
        文件不存在时返回默认人设。

        Returns:
            str: 人设内容
        """
        identity_path = os.path.join(self.workspace, self.identity_file)

        if not os.path.isfile(identity_path):
            return self.DEFAULT_IDENTITY

        try:
            with open(identity_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            # 读取失败（编码/权限等）同样回退默认人设：人设缺失会让
            # System Prompt 失去行为约束，宁可降级也不允许为空
            return self.DEFAULT_IDENTITY

    def _load_memory(self) -> str:
        """
        加载长期记忆内容

        从 workspace/memory/MEMORY.md 读取长期记忆。
        文件不存在时返回空字符串。

        这是第六章预留的接口，用于存储跨会话的知识和经验。

        Returns:
            str: 长期记忆内容，不存在时为空字符串
        """
        memory_path = os.path.join(self.workspace, "workspace", "memory", "MEMORY.md")

        if not os.path.isfile(memory_path):
            return ""

        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            # 记忆属于增强信息而非必需品，读取失败时静默降级为空，
            # 不能因记忆文件损坏而阻断正常对话
            return ""

    def build_system_prompt(self) -> str:
        """
        构建完整的 System Prompt

        拼接人设、时间、工作区路径和长期记忆，生成完整的系统提示。

        Returns:
            str: 完整的 System Prompt
        """
        # 加载人设
        identity = self._load_identity()

        # 获取当前时间
        # 注入实时时间让 Agent 感知"现在"，否则模型只能依赖训练数据
        # 的知识截止时间，无法正确理解"今天""明天"等相对时间表达
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

        # 加载长期记忆
        memory = self._load_memory()

        # 构建基础 System Prompt
        system_prompt = f"""{identity}

## 环境信息

- 当前时间: {current_time}
- 工作区路径: {self.workspace}"""

        # 如果有长期记忆，追加
        # 记忆放入 system 消息而非对话消息：system 层级对模型的约束力
        # 最强，且不占用对话轮次，适合承载跨会话的稳定信息
        if memory:
            system_prompt += f"\n\n## 长期记忆\n\n{memory}"
            
        system_prompt += """
        \n\n## 记忆管理指引
当你在对话中发现以下类型的重要信息时，使用 write_file 工具更新 工作目录下的workspace/memory/MEMORY.md：
- 用户的姓名、职业、技术偏好
- 用户的项目信息和工作习惯
- 用户明确要求你记住的事情
- 用户纠正过你的错误（避免下次再犯）

更新时读取现有内容，在末尾追加新条目，保持 Markdown 列表格式。
不要记录琐碎的对话细节，只记录长期有价值的信息。"""

        # 如果有技能摘要，追加
        if self.skills_summary:
            system_prompt += f"\n\n## 可用技能\n\n{self.skills_summary}"

        return system_prompt

    def build_messages(
        self,
        history: list[dict[str, Any]] | None = None,
        current_message: str = "",
    ) -> list[dict[str, Any]]:
        """
        构建完整的 messages 列表

        按顺序拼接：System Prompt + 历史对话 + 当前用户消息
        输出符合 OpenAI messages 格式，可直接传给 LLM。

        Args:
            history: 历史对话列表，每项为 {"role": "...", "content": "..."}
            current_message: 当前用户消息

        Returns:
            list[dict]: 完整的 messages 列表
        """
        messages: list[dict[str, Any]] = []

        # 添加 System Prompt
        messages.append({
            "role": "system",
            "content": self.build_system_prompt(),
        })

        # 添加历史对话
        if history:
            messages.extend(history)

        # 添加当前用户消息
        # 为空时不追加，避免生成 content 为空的用户消息干扰模型
        if current_message:
            messages.append({
                "role": "user",
                "content": current_message,
            })

        return messages