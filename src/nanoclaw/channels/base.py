"""
渠道适配器抽象基类模块。

Channel 定义所有渠道适配器（CLI / 飞书 / Web）必须遵循的统一接口契约，
支持多种消息平台以一致的方式接入。MessageBus 是渠道与 Agent 之间唯一的
通信桥梁，消息沿两条单向队列流动：

- 入站方向：渠道收到用户消息后构造 InboundMessage，发布到 bus.inbound_queue，
  由 Agent 消费处理；
- 出站方向：Agent 产生回复后发布到 bus.outbound_queue，
  渠道从中消费 Agent 回复，并通过 send() 投递给用户。

线程模型说明：
    各渠道的并发模型差异很大（飞书 SDK 回调运行在子线程、botpy 原生异步、
    CLI 的阻塞式 input() 放入线程池、Web 的多路 WebSocket 连接），
    本基类刻意不约束线程模型，只要求 start()/send() 为可 await 的协程，
    由各子类自行完成线程/协程之间的转换。

使用示例：
    class FeishuChannel(Channel):
        def __init__(self, bus: MessageBus):
            super().__init__(name="feishu", bus=bus)

        async def start(self):
            # 监听飞书事件，收到消息后发布到 bus.inbound_queue
            ...

        async def send(self, message: OutboundMessage):
            # 调用飞书 API 发送消息
            ...
"""

from abc import ABC, abstractmethod

from bus.queue import MessageBus, OutboundMessage


class Channel(ABC):
    """
    渠道适配器抽象基类

    定义渠道的生命周期和消息发送接口。
    所有渠道适配器必须实现 start() 和 send() 方法。

    设计意图：
        通过 ABC + @abstractmethod 强制每个渠道显式实现完整生命周期，
        避免子类遗漏 send() 导致 Agent 的回复"发出去就石沉大海"；
        收发逻辑不做默认实现，因为各渠道的收发机制差异过大，
        强行统一抽象反而会引入不必要的约束。

    Attributes:
        name: 渠道名称，如 "feishu"、"qq"、"web"、"cli"
        bus: 消息总线实例，用于发布/消费消息
    """

    def __init__(self, name: str, bus: MessageBus) -> None:
        """
        初始化渠道适配器的基础字段。

        Args:
            name: 渠道名称，会写入消息的 channel 字段，
                供总线侧区分消息来源
            bus: 消息总线实例
        """
        self.name = name
        self.bus = bus

    @abstractmethod
    async def start(self) -> None:
        """
        启动渠道适配器

        渠道的生命周期入口，通常是一个无限循环：
        - 监听用户消息 → 发布到 bus.inbound_queue
        - 或从 bus.outbound_queue 消费回复 → 发送给用户

        该方法在后台任务中运行，不应阻塞主线程。
        """
        ...

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """
        发送出站消息给用户

        从 bus.outbound_queue 消费到消息后，调用此方法发送给用户。
        具体实现依赖各渠道的 API（如飞书 Webhook、QQ HTTP API）。

        Args:
            message: 出站消息实例
        """
        ...

    async def stop(self) -> None:
        """
        停止渠道适配器（可选）

        清理资源、关闭连接等。默认实现为空，可由子类覆盖。
        """
        # 默认不做任何清理：并非所有渠道都持有需要显式释放的资源，
        # 子类按需覆盖（如飞书/QQ 需断开长连接，Web 需关闭 uvicorn）
        pass