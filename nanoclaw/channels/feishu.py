"""飞书渠道实现模块。

通过 lark-oapi SDK 的 WebSocket 长连接接收飞书消息事件，通过 REST API 发送回复。

线程模型特点（本渠道的关键设计点）：
    lark SDK 的 ws.Client.start() 内部自起网络线程，并在【子线程】中
    回调 _on_message，回调内无法直接 await。因此：

    - start() 时先用 asyncio.get_running_loop() 保存主事件循环引用；
    - _on_message 在子线程中解析消息、构造好 InboundMessage 后，
      用 asyncio.run_coroutine_threadsafe() 把 publish_inbound 协程
      投递回主事件循环执行，保证对 MessageBus 的访问全部发生在
      主循环协程中，避免跨线程竞争；
    - 发送方向相反：send() 在主循环中执行，但 lark REST 客户端是
      同步阻塞的，需用 asyncio.to_thread 包装放入线程池，
      避免发送期间卡住主循环。
"""

import asyncio
import json
import re
from typing import Any

import lark_oapi as lark
import lark_oapi.api.im.v1 as im_v1

from bus import InboundMessage, MessageBus, OutboundMessage
from .base import Channel

class FeishuChannel(Channel):
    """
    飞书渠道，通过 WebSocket 接收消息并通过 API 发送回复。

    设计意图：
        收发使用两个独立的 lark 客户端——ws.Client 专职接收事件
        （内部自起子线程），lark.Client 专职调用发送 API（同步阻塞）；
        二者职责互不干扰，stop 时也可分别释放。
    """

    def __init__(self, bus: MessageBus, app_id: str, app_secret: str):
        """初始化飞书渠道。

        Args:
            bus: 消息总线实例
            app_id: 飞书 App ID
            app_secret: 飞书 App Secret

        Attributes:
            _client: lark WebSocket 客户端，负责接收消息事件
            _lark_api_client: lark REST API 客户端，负责发送回复
            _running: 运行状态标志
            _loop: 主事件循环引用，供子线程回调投递协程使用
        """
        super().__init__(name="feishu", bus=bus)
        self.app_id = app_id
        self.app_secret = app_secret
        self._client: Any = None
        self._lark_api_client: Any = None  # 用于发送消息的 API 客户端
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None  # 保存主事件循环引用

    async def start(self) -> None:
        """启动飞书 WebSocket 客户端，监听消息事件。"""
        # ★ 保存当前事件循环的引用，供 WebSocket 子线程回调使用
        self._loop = asyncio.get_running_loop()

        # 创建用于发送消息的 API 客户端
        self._lark_api_client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()

        # 创建事件处理器（使用 P2 版本协议）
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._on_message) \
            .build()

        # 创建 WebSocket 客户端
        self._client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        self._running = True
        print("[FeishuChannel] WebSocket 连接中...")

        # 启动 WebSocket 连接：lark 的 client.start() 是阻塞调用且
        # 内部自起网络线程，直接调用会卡死主循环，故放进线程池执行
        await asyncio.to_thread(self._client.start)

    def _on_message(self, data: Any) -> None:
        """处理飞书消息接收事件（P2 协议）。

        注意：此方法在飞书 SDK 的 WebSocket 子线程中被调用，
        不能直接 await，需要通过 run_coroutine_threadsafe 投递到主事件循环。

        Args:
            data: 飞书 P2 事件回调数据
        """
        try:
            event = data.event
            # 提取消息信息
            sender_id = event.sender.sender_id.open_id
            chat_id = event.message.chat_id
            message_type = event.message.message_type
            sender_type = event.sender.sender_type

            print(f"[FeishuChannel] 收到消息 sender={sender_id} type={message_type} sender_type={sender_type}")

            # 过滤：忽略 Bot 自己发送的消息
            if sender_type == "app":
                return

            # 过滤：只处理文本消息
            if message_type != "text":
                return

            # 解析消息内容（飞书消息是 JSON 格式）
            content = json.loads(event.message.content)
            text = content.get("text", "")

            # 去掉 @Bot 的前缀（飞书群消息中会有 @_user_xxx 前缀）
            text = re.sub(r'@_all|@\S+', "", text).strip()

            # 跳过空消息
            if not text:
                return

            print(f"[FeishuChannel] 解析文本: {text}")

            # 构造入站消息
            message = InboundMessage(
                channel="feishu",
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                raw={
                    "sender_id": sender_id,
                    "chat_id": chat_id,
                    "message_type": message_type,
                    "content": content,
                },
            )

            # ★ 使用启动时保存的主事件循环引用，线程安全地投递
            if self._loop is not None and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.bus.publish_inbound(message),
                    self._loop,
                )

                # 观察投递结果：失败时打印日志，避免消息静默丢失
                def _on_delivered(fut, msg_text=text):
                    exc = fut.exception()
                    if exc is not None:
                        print(f"[FeishuChannel] 消息投递主循环失败: {exc} (内容: {msg_text[:50]})")

                future.add_done_callback(_on_delivered)
            else:
                print(f"[FeishuChannel] 警告: 主事件循环不可用，消息丢弃: {text[:50]}")

        except Exception as e:
            # 回调运行在 SDK 子线程，异常若不在此捕获只会消失在子线程
            # 栈里、不影响主循环，排查时无从知晓，故原地打印完整堆栈
            import traceback
            print(f"[FeishuChannel] 消息处理异常: {e}")
            traceback.print_exc()

    async def send(self, message: OutboundMessage) -> None:
        """发送消息到飞书。

        发送策略：失败自动重试一次（共两次尝试），两次均失败后
        放弃并打印日志，避免无限重试在总线上堆积消息。

        Args:
            message: 出站消息实例
        """
        if self._lark_api_client is None:
            print("[FeishuChannel] API 客户端未初始化，无法发送消息")
            return

        # 空内容保护：飞书 API 拒绝空文本消息，发送必然失败导致用户收不到任何回复
        if not message.content or not message.content.strip():
            print("[FeishuChannel] 跳过空回复")
            return

        # 构造飞书消息请求
        request = im_v1.CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(
                im_v1.CreateMessageRequestBody.builder()
                .receive_id(message.chat_id)
                .msg_type("text")
                .content(json.dumps({"text": message.content}, ensure_ascii=False))
                .build()
            ) \
            .build()

        # 发送消息（飞书 API 是同步的，用 asyncio.to_thread 包装）
        # 失败自动重试一次：飞书偶发网络抖动/频控时避免用户收不到回复
        for attempt in range(2):
            try:
                response = await asyncio.to_thread(
                    self._lark_api_client.im.v1.message.create,
                    request,
                )

                if response.success():
                    return

                print(f"[FeishuChannel] 发送失败 (第{attempt + 1}次): code={response.code} msg={response.msg}")

            except Exception as e:
                print(f"[FeishuChannel] 发送消息异常 (第{attempt + 1}次): {e}")

            # 未成功，间隔后重试
            if attempt == 0:
                await asyncio.sleep(1)

        print(f"[FeishuChannel] 回复最终发送失败，chat_id={message.chat_id}")

    async def stop(self) -> None:
        """停止飞书客户端连接。

        仅释放各客户端引用让连接随对象回收；_loop 置空后，子线程
        回调会因主循环引用缺失而走"消息丢弃"分支，不会误投递。
        """
        self._running = False
        self._loop = None
        if self._client is not None:
            self._client = None
        self._lark_api_client = None