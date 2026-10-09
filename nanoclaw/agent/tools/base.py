"""
Tool 抽象基类模块

定义所有工具的统一接口规范，子类必须实现 name、description、parameters
三个属性和 execute 异步方法。to_function_definition 方法自动生成 OpenAI
function calling 所需的 JSON 结构，无需手动拼装。
"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """
    工具抽象基类

    所有可供 Agent 调用的工具都必须继承此类，并实现以下内容：
    - name: 工具的唯一标识名称，用于路由调用
    - description: 工具功能的自然语言描述，供 LLM 理解工具用途
    - parameters: 工具参数的 JSON Schema 描述，遵循 OpenAI function calling 规范
    - execute: 异步执行方法，接收关键字参数，返回字符串结果

    使用示例：
        class SearchTool(Tool):
            @property
            def name(self) -> str:
                return "search"

            @property
            def description(self) -> str:
                return "搜索互联网获取信息"

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }

            async def execute(self, **kwargs) -> str:
                query = kwargs["query"]
                return f"搜索结果: {query}"
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，唯一标识，用于匹配和路由调用"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能的自然语言描述，供 LLM 判断何时调用此工具"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        工具参数的 JSON Schema 定义

        格式遵循 OpenAI function calling 规范，示例：
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        异步执行工具逻辑

        Args:
            **kwargs: 工具参数，与 parameters 中定义的 schema 对应

        Returns:
            str: 工具执行结果的字符串表示。
                约定返回纯字符串：执行结果会作为 tool 消息直接回传给 LLM，
                复杂对象需自行序列化为 LLM 可读的文本形式
        """
        ...

    def to_function_definition(self) -> dict:
        """
        将工具定义转换为 OpenAI function calling 格式

        自动将 name、description、parameters 组装为 OpenAI API 所需的
        tools JSON 结构，可直接传入 chat completion 请求的 tools 参数。

        Returns:
            dict: OpenAI 格式的工具定义，结构如下：
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        # 按 OpenAI 规范组装两层结构：外层 type 固定为 "function"，
        # 工具的名称/描述/参数收拢在内层 function 字段中
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
