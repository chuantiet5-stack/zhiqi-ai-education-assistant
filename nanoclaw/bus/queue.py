"""
消息总线模块

MessageBus 提供异步消息队列，支持多渠道接入：
- inbound_queue：入站消息队列（用户发给 Agent）
- outbound_queue：出站消息队列（Agent 回复给用户）

支持多种渠道接入，如飞书、QQ、Web、CLI 等。

使用示例：
    bus = MessageBus()

    # 发布入站消息
    await bus.publish_inbound(InboundMessage(
        channel="feishu",
        sender_id="user123",
        chat_id="group456",
        content="你好"
    ))

    # 消费入站消息
    msg = await bus.consume_inbound()

    # 发布出站消息
    await bus.publish_outbound(OutboundMessage(
        channel="feishu",
        chat_id="group456",
        content="你好！有什么可以帮你的？"
    ))
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InboundMessage:
    """
    入站消息结构

    用户发给 Agent 的消息，包含来源渠道、发送者、会话和内容。

    Attributes:
        channel: 来源渠道名称，如 "feishu"、"qq"、"web"、"cli"
        sender_id: 发送者标识（用户 ID）
        chat_id: 会话标识（群聊或私聊的 ID）
        content: 消息正文
        raw: 原始消息数据，调试用（可选）
    """

    channel: str
    sender_id: str
    chat_id: str
    content: str
    raw: dict[str, Any] | None = None


@dataclass
class OutboundMessage:
    """
    出站消息结构

    Agent 回复给用户的消息，包含目标渠道、会话和内容。

    Attributes:
        channel: 目标渠道名称
        chat_id: 目标会话 ID
        content: 回复正文
        reply_to: 引用的消息 ID（可选）
        meta: 结构化附加数据（可选）。普通聊天为 None；教育助手的
            流式增量（edu_delta）与最终结果（edu_result）靠它区分，
            渠道侧据此把消息投递给等待中的 SSE 响应而非 WebSocket 连接
    """

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    meta: dict[str, Any] | None = None


class MessageBus:
    """
    消息总线

    使用 asyncio.Queue 实现异步消息队列，支持多渠道并发接入。
    在架构中位于渠道层与网关层之间：渠道和网关都只与总线交互，
    彼此不直接引用，从而实现解耦——新增渠道无需修改网关代码。

    inbound_queue：入站消息队列，存放用户发给 Agent 的消息
    outbound_queue：出站消息队列，存放 Agent 回复给用户的消息

    使用示例：
        bus = MessageBus()

        # 启动消费者（后台任务）
        asyncio.create_task(process_inbound(bus))
        asyncio.create_task(process_outbound(bus))

        # 发布消息
        await bus.publish_inbound(msg)
    """

    def __init__(self) -> None:
        """初始化消息总线，创建两个异步队列"""
        # 使用无界队列：本应用消息量小且消费者常驻，无需背压控制；
        # 有界队列反而可能在积压时让渠道侧 publish 被阻塞
        self.inbound_queue: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound_queue: asyncio.Queue[OutboundMessage] = asyncio.Queue()
    #发布入站消息
    async def publish_inbound(self, msg: InboundMessage) -> None:
        """
        发布入站消息

        将入站消息放入 inbound_queue，等待 Agent 消费处理。

        Args:
            msg: 入站消息实例
        """
        await self.inbound_queue.put(msg)
    #消费入站消息
    async def consume_inbound(self) -> InboundMessage:
        """
        消费入站消息

        从 inbound_queue 取出一条消息。如果队列空，自动等待直到有消息。

        Returns:
            InboundMessage: 入站消息实例
        """
        return await self.inbound_queue.get()
   #发布出站消息
    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """
        发布出站消息

        将出站消息放入 outbound_queue，等待渠道适配器发送。

        Args:
            msg: 出站消息实例
        """
        await self.outbound_queue.put(msg)
    #消费出站消息
    async def consume_outbound(self) -> OutboundMessage:
        """
        消费出站消息

        从 outbound_queue 取出一条消息。如果队列空，自动等待直到有消息。

        Returns:
            OutboundMessage: 出站消息实例
        """
        return await self.outbound_queue.get()