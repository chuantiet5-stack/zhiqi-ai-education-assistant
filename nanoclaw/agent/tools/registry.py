"""
工具注册表模块

提供 ToolRegistry 类，统一管理工具的注册、查询和执行。
Agent 通过注册表获取工具定义列表传给 LLM，并在收到调用请求时
通过注册表路由到对应工具执行。
"""

from agent.tools.base import Tool


class ToolRegistry:
    """
    工具注册表

    集中管理所有已注册的 Tool 实例，提供以下能力：
    - register: 注册工具，以工具 name 为 key 存储
    - get_definitions: 批量获取所有工具的 OpenAI function calling JSON 定义
    - execute: 根据工具名查找并异步执行，自动处理工具不存在和执行异常
    - list_tools: 列出所有已注册的工具名称

    使用示例：
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(CalculatorTool())

        # 获取定义，传给 OpenAI API
        definitions = registry.get_definitions()

        # 执行工具
        result = await registry.execute("search", {"query": "天气"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        注册一个工具

        以工具的 name 属性作为 key 存储。如果同名工具已存在则覆盖。

        Args:
            tool: 实现了 Tool 接口的工具实例
        """
        # 同名工具直接覆盖而非报错：便于在测试或运行期用 mock/升级版工具替换旧实例
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict]:
        """
        获取所有已注册工具的 OpenAI function calling 定义

        遍历注册表中的每个工具，调用其 to_function_definition() 方法，
        返回可直接传给 OpenAI chat completion API 的 tools 参数列表。

        Returns:
            list[dict]: 工具定义列表，每项格式为
                {"type": "function", "function": {...}}
        """
        return [tool.to_function_definition() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        """
        根据工具名异步执行工具

        在注册表中查找指定名称的工具，并使用 arguments 字典中的参数调用其
        execute 方法。如果工具不存在或执行过程中抛出异常，返回错误信息字符串
        而非抛出异常，确保调用方可以安全地将结果回传给 LLM。

        Args:
            name: 工具名称
            arguments: 工具参数字典，将作为 **kwargs 传入 execute

        Returns:
            str: 工具执行结果；出错时返回 "错误: ..." 格式的错误信息
        """
        tool = self._tools.get(name)
        if tool is None:
            # 工具不存在也不抛异常：让 LLM 从错误文本中得知工具名不对，可自行纠正重试
            return f"错误: 工具 '{name}' 不存在"

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            # 异常兜底：把工具执行异常转为错误字符串而非向上抛出。
            # Agent 主循环拿到这里的返回值后会作为 tool 消息回传给 LLM，
            # 若在此处抛异常会中断整个 Agent 对话循环；返回错误文本则让 LLM
            # 有机会换参数重试、换工具或放弃任务，循环得以继续运转
            return f"错误: 执行工具 '{name}' 失败 - {e}"

    def list_tools(self) -> list[str]:
        """
        返回所有已注册工具的名称列表

        Returns:
            list[str]: 工具名称列表
        """
        return list(self._tools.keys())
