"""
子代理工具模块

SpawnSubagentTool 允许 Agent 创建临时的子 Agent 执行子任务。
子 Agent 拥有独立的工具集和对话循环，完成后返回结果。

特性：
- 支持多级嵌套（通过 max_depth 控制）
- 子 Agent 不持久化（DummySessionManager）
- 可指定不同的模型处理子任务

使用示例：
    provider_factory = lambda model: OpenAICompatProvider(...)
    tools_registry = ToolRegistry()
    tools_registry.register(SpawnSubagentTool(
        provider_factory=provider_factory,
        tools_registry=tools_registry,
        workspace="/path/to/project",
        current_depth=0,
        max_depth=2
    ))
"""

from typing import Any, Callable

from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.loop import AgentLoop
from agent.context import ContextBuilder
from session.manager import SessionManager
from providers.base import LLMProvider


class DummySessionManager(SessionManager):
    """
    虚拟会话管理器

    用于子 Agent，不实际持久化对话历史。
    所有方法都是空操作或返回空数据。
    """

    def __init__(self) -> None:
        """初始化虚拟会话管理器"""
        # 不需要 sessions_dir，跳过父类构造
        self.sessions_dir = ""

    def save_message(self, session_key: str, message: dict[str, Any]) -> None:
        """空操作，不保存消息"""
        pass

    def get_history(self, session_key: str) -> list[dict[str, Any]]:
        """返回空列表，无历史记录"""
        return []

    def clear(self, session_key: str) -> None:
        """空操作，不清除任何内容"""
        pass

    def list_sessions(self) -> list[str]:
        """返回空列表，无会话记录"""
        return []


class SpawnSubagentTool(Tool):
    """
    子代理工具

    创建临时 AgentLoop 执行子任务，返回结果。
    支持多级嵌套（通过 current_depth 和 max_depth 控制）。
    """

    def __init__(
        self,
        provider_factory: Callable[[str | None], LLMProvider],
        tools_registry: ToolRegistry,
        workspace: str,
        current_depth: int = 0,
        max_depth: int = 2,
    ) -> None:
        """
        Args:
            provider_factory: Provider 创建函数，输入模型名，返回 LLMProvider 实例。
                LLM 可通过可选的 model 参数为子任务指定更便宜的模型（模型降级）：
                简单子任务交给低价模型处理，主 Agent 保留强模型做规划，
                从而显著降低多 Agent 场景的整体调用成本
            tools_registry: 工具注册表，用于复制工具给子 Agent
            workspace: 工作区路径
            current_depth: 当前嵌套深度，默认 0
            max_depth: 最大嵌套深度，默认 2
        """
        self.provider_factory = provider_factory
        self.tools_registry = tools_registry
        self.workspace = workspace
        self.current_depth = current_depth
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        """工具名称"""
        return "spawn_subagent"

    @property
    def description(self) -> str:
        """工具描述"""
        return "创建一个临时的子 Agent 执行子任务，适用于复杂的多步骤任务或需要独立处理流程的场景。子 Agent 完成任务后返回结果。"

    @property
    def parameters(self) -> dict:
        """参数定义"""
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "子任务描述，清晰说明需要子 Agent 完成的任务内容",
                },
                "model": {
                    "type": "string",
                    "description": "子 Agent 使用的模型名称（可选），不指定则使用默认模型",
                },
            },
            "required": ["task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """
        执行子任务

        1. 创建 Provider（用 provider_factory(model)）
        2. 复制工具（从 tools_registry 复制，跳过 spawn_subagent 自己）
        3. 如果 current_depth + 1 < max_depth，给子 Agent 也注册一个 SpawnSubagentTool（depth+1）
        4. 创建 AgentLoop（简化的 system prompt，不需要持久化）
        5. await agent.run(task)，返回结果

        Args:
            task: 子任务描述
            model: 模型名称（可选）

        Returns:
            str: 子 Agent 的执行结果
        """
        # 提取参数
        task = kwargs.get("task", "")
        model = kwargs.get("model")

        if not task:
            return "错误: 缺少必填参数 'task'"

        try:
            # 1. 创建 Provider
            provider = self.provider_factory(model)

            # 2. 复制工具注册表（跳过 spawn_subagent 自己）
            # 直接遍历父注册表内部字典以共享工具实例（工具无状态，可安全复用），
            # 无需为拷贝语义额外增加接口；排除 spawn_subagent 是因为当前实例的
            # current_depth 属于父层级，直接复用会导致深度计数永远不增长
            child_tools = ToolRegistry()
            for tool in self.tools_registry._tools.values():
                if tool.name != "spawn_subagent":
                    child_tools.register(tool)

            # 3. 如果未达到最大深度，给子 Agent 也注册一个 SpawnSubagentTool
            # 深度判断保证嵌套层数有硬上限，防止 Agent 之间无限递归派生
            if self.current_depth + 1 < self.max_depth:
                child_spawn_tool = SpawnSubagentTool(
                    provider_factory=self.provider_factory,
                    tools_registry=self.tools_registry,
                    workspace=self.workspace,
                    current_depth=self.current_depth + 1,
                    max_depth=self.max_depth,
                )
                child_tools.register(child_spawn_tool)

            # 4. 创建简化的上下文构建器
            # 子 Agent 是"任务专员"：不加载 identity.md 与技能摘要，
            # 用一句话人设替代完整 system prompt，减少每轮请求的 token 开销
            simple_context = ContextBuilder(
                workspace=self.workspace,
                identity_file="",  # 不加载 identity.md
                skills_summary="",  # 不加载技能摘要
            )

            # 覆盖默认人设（使用 DEFAULT_IDENTITY 的简短版本）
            simple_context.DEFAULT_IDENTITY = "你是任务专员，完成任务直接输出结果，无需过多解释。"

            # 5. 创建子 AgentLoop（不持久化）
            # 独立会话但用 DummySessionManager：子任务用完即弃，
            # 历史落盘只会污染主 Agent 的会话列表，因此全部空操作
            dummy_session = DummySessionManager()
            child_agent = AgentLoop(
                provider=provider,
                tools=child_tools,
                context=simple_context,
                session_manager=dummy_session,
                model=model,
                max_iterations=16,  # 限制迭代次数：防止子任务陷入循环无限消耗 token
                session_key="child:temp",  # 占位 key：Dummy 不落盘，不存在冲突风险
            )

            # 不挂载压缩器：子 Agent 会话短且迭代次数受限，上下文不易膨胀，
            # 压缩反而多花一次 LLM 调用
            child_agent.consolidator = None

            # 6. 执行子任务
            print(f"  🚀 启动子 Agent (深度 {self.current_depth + 1}/{self.max_depth})...")
            result = await child_agent.run(task)

            print(f"  ✅ 子 Agent 完成")

            return result

        except Exception as e:
            return f"错误: 子 Agent 执行失败 - {e}"