"""
LLM Provider 抽象基类模块

定义 LLM 调用的统一接口，支持多模型提供商切换。
ToolCallRequest 和 LLMResponse 封装了模型响应的标准化结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    """
    工具调用请求

    当 LLM 决定调用工具时返回的结构，包含调用所需的全部信息。
    reasoning_content 字段保存支持"思考"功能的模型（如 Kimi-K2.5、DeepSeek-R1）
    在工具调用前的推理过程。

    Attributes:
        id: 工具调用的唯一标识，用于后续回传 tool message 时关联
        name: 要调用的工具名称
        arguments: 工具参数，已解析为字典
        reasoning_content: 模型的推理过程（可选），部分模型支持
    """

    id: str
    name: str
    arguments: dict[str, Any]
    reasoning_content: str | None = None


@dataclass
class LLMResponse:
    """
    LLM 响应结构

    标准化 LLM 返回结果，包含文本内容、工具调用、结束原因和用量统计。

    Attributes:
        content: 文本内容，无工具调用时有值
        tool_calls: 工具调用列表，模型决定调用工具时有值
        finish_reason: 结束原因，常见值：stop、tool_calls、length
        usage: 用量统计，包含 prompt_tokens、completion_tokens 等
    """

    content: str | None = None
    # 可变类型不能直接作为 dataclass 默认值（所有实例会共享同一个列表/字典），
    # 必须用 default_factory 每次生成新对象
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """判断响应是否包含工具调用"""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """
    LLM 提供商抽象基类

    定义与 LLM 交互的统一接口，所有提供商适配器必须实现 chat 方法。
    支持传入工具定义（OpenAI function calling 格式）和模型选择。

    使用示例：
        class OpenAIProvider(LLMProvider):
            def __init__(self, api_key: str):
                self.client = OpenAI(api_key=api_key)

            async def chat(
                self,
                messages: list[dict],
                tools: list[dict] | None = None,
                model: str | None = None
            ) -> LLMResponse:
                response = await self.client.chat.completions.create(
                    model=model or "gpt-4",
                    messages=messages,
                    tools=tools
                )
                # ... 解析响应
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """
        发送对话请求并获取响应

        Args:
            messages: 对话消息列表，格式为 OpenAI messages 格式
                [{"role": "user", "content": "..."}]
            tools: 工具定义列表，OpenAI function calling 格式
                [{"type": "function", "function": {...}}]
            model: 指定使用的模型，None 时使用默认模型

        Returns:
            LLMResponse: 标准化的 LLM 响应
        """
        ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        """流式对话：逐段 yield 文本增量（可选实现）。

        仅用于无工具调用的直答场景（如学伴对话的打字机效果）；
        需要工具调用的路径请继续使用 chat()。

        默认实现抛 NotImplementedError，由支持的 Provider
        （如 OpenAICompatProvider）覆写。
        """
        raise NotImplementedError("当前 Provider 不支持流式输出")