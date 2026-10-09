"""
OpenAI 兼容 Provider 实现

兼容 OpenAI API 格式的各种服务商，包括：
- OpenAI 官方 API
- 硅基流动 (SiliconFlow)
- Kimi (Moonshot)
- DeepSeek
- 其他 OpenAI 兼容的第三方服务

通过统一接口封装，自动处理工具调用参数、推理内容提取和异常容错。
"""

import json
from typing import Any

from openai import AsyncOpenAI

from providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OpenAICompatProvider(LLMProvider):
    """
    OpenAI 兼容 API Provider

    支持所有 OpenAI API 格式的服务，通过 base_url 切换不同提供商。
    内置异常容错，API 调用失败时返回错误响应而非抛出异常。

    关键特性：
    - 仅在 tools 参数存在时传递 tool_choice，避免部分 API 报错
    - 自动提取 reasoning_content（支持 Kimi-K2.5、DeepSeek-R1 等思考模型）
    - 自动解析 tool_calls 的 arguments JSON

    使用示例：
        # OpenAI 官方
        provider = OpenAICompatProvider(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-4"
        )

        # 硅基流动
        provider = OpenAICompatProvider(
            api_key="sk-xxx",
            base_url="https://api.siliconflow.cn/v1",
            model="Qwen/Qwen2.5-72B-Instruct"
        )

        # DeepSeek
        provider = OpenAICompatProvider(
            api_key="sk-xxx",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat"
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        """
        Args:
            api_key: API 密钥
            base_url: API 基础 URL，不同服务商地址不同
            model: 默认使用的模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # 显式超时（默认 600s 过长）+ 最多重试 1 次（默认 2 次）：
        # 图片轮的请求体达几 MB，重试会成倍放大等待时间
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=1,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        发送对话请求

        动态构建请求参数：仅在 tools 存在时传递 tool_choice，
        避免硅基流动等 API 报 "tools must be specified when tool_choice is utilized" 错误。

        Args:
            messages: OpenAI 格式的消息列表（content 可为数组实现多模态输入）
            tools: 工具定义列表（可选）
            model: 临时覆盖默认模型（可选）
            extra_params: 额外请求参数（可选），如 temperature、max_tokens，
                直接合并进请求体，用于调用级微调生成行为
        """
        try:
            # 动态构建请求参数
            request_params: dict[str, Any] = {
                "model": model or self.model,
                "messages": messages,
            }

            # 仅在 tools 存在时添加工具相关参数
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            # 合并调用级额外参数（temperature / max_tokens 等）
            if extra_params:
                request_params.update(extra_params)

            # 调用 API
            response = await self.client.chat.completions.create(**request_params)

            # 解析响应
            message = response.choices[0].message
            # 部分兼容服务可能返回 None，统一兜底为 "stop"，避免下游判断逻辑遇到 None
            finish_reason = response.choices[0].finish_reason or "stop"

            # 提取 usage
            usage: dict[str, Any] = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # 获取 message 顶层的 reasoning_content（部分模型放在这里）；
            # 用 getattr 容错取值：不同 SDK 版本/服务商的响应对象不一定有该属性，
            # 直接访问属性会抛 AttributeError
            message_reasoning = getattr(message, "reasoning_content", None)

            # 解析 tool_calls
            tool_calls: list[ToolCallRequest] = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    # 解析 arguments JSON；模型偶尔会输出非法 JSON，
                    # 此时保留原始字符串供排查，而不是丢弃整次工具调用
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}

                    # 获取 tool_call 级别的 reasoning_content
                    tc_reasoning = getattr(tc, "reasoning_content", None)

                    # 合并 reasoning：优先 tool_call 级别，其次 message 级别
                    reasoning_content = tc_reasoning or message_reasoning

                    tool_calls.append(
                        ToolCallRequest(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                            reasoning_content=reasoning_content,
                        )
                    )

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

        except Exception as e:
            # 异常容错：返回错误响应而非抛出异常
            return LLMResponse(
                content=f"错误: API 调用失败 - {e}",
                finish_reason="error",
            )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        """流式对话：逐段 yield 文本增量（打字机效果）。

        仅用于无工具调用的直答场景；
        需要工具调用的流式路径请使用 chat_stream_ex()。

        Args:
            messages: OpenAI 格式的消息列表（content 可为数组实现多模态输入）
            model: 临时覆盖默认模型（可选）
            extra_params: 额外请求参数（可选），如 temperature、max_tokens

        Yields:
            str: 模型输出的文本增量片段

        容错约定：开始生成后发生的异常（网络中断、服务端错误等）
        转为 yield 一段提示文本，保证下游 SSE 通道能自然收尾；
        建连前的失败同样以 yield 错误文本代替抛异常。
        """
        async for event_type, payload in self.chat_stream_ex(
            messages, model=model, extra_params=extra_params
        ):
            if event_type == "delta":
                yield payload

    async def chat_stream_ex(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        """流式对话（扩展版）：支持工具调用的事件流。

        在 chat_stream 的纯文本流基础上扩展：
        - 支持传入 tools 定义，模型可在流式过程中请求工具调用；
        - 以 (event_type, payload) 元组形式产出事件，调用方按类型分派：
            ("delta", str)                 文本增量片段
            ("tool_calls", [ToolCallRequest])  流结束后一次性产出完整工具调用
            ("done", str)                  结束原因（stop / length / tool_calls / error）

        工具调用增量（tool_call delta）按 index 归并：id / 函数名直接覆盖，
        arguments 为 JSON 字符串分片需拼接，流结束后统一解析。

        容错约定：与 chat_stream 相同——异常转为 delta 文本 + done(error)，
        不向上抛异常，保证 Agent 循环与 SSE 通道都能自然收尾。
        """
        request_params: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
        }
        # 仅在 tools 存在时传递工具参数（与 chat() 的容错逻辑一致）
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        if extra_params:
            request_params.update(extra_params)

        # 工具调用增量归并缓冲：index -> {"id", "name", "arguments"}
        tool_acc: dict[int, dict[str, str]] = {}
        finish_reason = "stop"

        try:
            stream = await self.client.chat.completions.create(**request_params)
            async for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # 结束原因：流式响应在最后一个 chunk 携带 finish_reason
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

                # 工具调用增量归并（此时不应有文本增量输出给用户）
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        slot = tool_acc.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

                # 文本增量
                piece = getattr(delta, "content", None)
                if piece:
                    yield ("delta", piece)

            # 流结束：归并结果里有工具调用则一次性产出
            if tool_acc:
                tool_calls: list[ToolCallRequest] = []
                for idx in sorted(tool_acc.keys()):
                    slot = tool_acc[idx]
                    try:
                        args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                    tool_calls.append(ToolCallRequest(
                        id=slot["id"] or f"call_{idx}",
                        name=slot["name"],
                        arguments=args,
                    ))
                yield ("tool_calls", tool_calls)
                finish_reason = "tool_calls"

            yield ("done", finish_reason)

        except Exception as e:
            yield ("delta", f"\n\n> ⚠️ 生成中断：{e}")
            yield ("done", "error")