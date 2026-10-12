"""
Gateway 网关模块

Gateway 是多渠道接入的核心网关，负责：
- 管理多个渠道适配器（Channel）
- 处理入站消息：从 MessageBus 消费 → 分发给对应的 Agent
- 分发出站消息：从 MessageBus 消费 → 发送给对应的渠道
- 缓存 Agent 实例（按 session_key）

使用示例：
    bus = MessageBus()
    cli_channel = CLIChannel(bus)

    def create_agent(session_key: str) -> AgentLoop:
        # 根据 session_key 创建 Agent
        ...

    gateway = Gateway(
        bus=bus,
        channels=[cli_channel],
        agent_factory=create_agent
    )

    # 启动网关
    asyncio.run(gateway.run())
"""

import asyncio
from typing import Callable

from bus.queue import MessageBus, InboundMessage, OutboundMessage
from channels.base import Channel
from agent.loop import AgentLoop


class Gateway:
    """
    网关核心类

    管理多渠道接入和消息路由。
    """

    def __init__(
        self,
        bus: MessageBus,
        channels: list[Channel],
        agent_factory: Callable[[str], AgentLoop],
        inbound_hook: "Callable[[InboundMessage], Any] | None" = None,
    ) -> None:
        """
        Args:
            bus: 消息总线实例
            channels: 渠道适配器列表
            agent_factory: Agent 创建函数，输入 session_key，返回 AgentLoop 实例
            inbound_hook: 入站拦截钩子（可选）。在消息进入 Agent 之前调用，
                签名 async hook(msg) -> bool：返回 True 表示消息已被钩子
                处理完毕（如家长报告指令），不再转给 Agent；False/异常则
                照常走 Agent 流程
        """
        self.bus = bus
        self.channels = channels
        self.agent_factory = agent_factory
        self.inbound_hook = inbound_hook

        # 内部状态
        self._agents: dict[str, AgentLoop] = {}  # 按 session_key 缓存 Agent
        self._channel_map: dict[str, Channel] = {}  # 按渠道名索引渠道

        # 每个会话一个消息队列 + 一个 worker 任务：
        # 同一会话的消息严格串行（保证对话历史顺序），不同会话之间并发处理，
        # 避免一条消息的多轮工具调用（可能耗时数十秒）阻塞其他渠道的消息
        self._session_queues: dict[str, asyncio.Queue] = {}
        self._session_workers: dict[str, asyncio.Task] = {}

        # 构建渠道映射表
        for channel in channels:
            self._channel_map[channel.name] = channel

    async def run(self) -> None:
        """
        启动网关

        用 asyncio.TaskGroup 并发启动：
        - 所有渠道的 start()
        - _process_inbound() 入站消费循环
        - _dispatch_outbound() 出站分发循环

        设计原则：CLI 渠道的正常结束（用户 /exit 或 EOF）不应中断其他渠道（QQ、飞书、Web）。
        因此 CLI start() 完成（返回或异常）不触发全局 shutdown，只打印日志；只有当
        核心循环（入站/出站）失败时，才整体 shutdown。
        """
        # 用 task 包装：每个任务作为可单独取消的 task，便于区分关键任务与非关键任务
        running_tasks: dict[str, asyncio.Task] = {}

        # 关键循环（入站/出站）：任何一个失败都要关闭整个网关
        core_tasks = {
            "_inbound": asyncio.create_task(self._process_inbound()),
            "_outbound": asyncio.create_task(self._dispatch_outbound()),
        }

        # 渠道任务：CLI 被视为非关键（用户退出 CLI 不影响其他渠道）
        for channel in self.channels:
            key = f"channel:{channel.name}"
            running_tasks[key] = asyncio.create_task(channel.start())

        running_tasks.update(core_tasks)

        # 监听所有任务，直到：
        #   - core 循环任一 done（关键失败/退出 → 整体关闭）
        #   - 所有任务 done（所有渠道都已自然退出）
        #
        # 之所以用 asyncio.wait(FIRST_COMPLETED) 而不是 TaskGroup / gather：
        # - TaskGroup 中任一任务异常会立刻取消其余全部任务，CLI 渠道正常退出
        #   会误伤 QQ/飞书/Web 等长驻渠道
        # - gather 要等所有任务结束才返回，无法在核心循环失败时"及时感知"
        # - wait(FIRST_COMPLETED) 让主循环在任一任务结束时醒来，逐个检查其结果，
        #   由这里自行决定是整体关闭还是继续等待剩余任务
        try:
            while True:
                done, _pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # 先清理已完成的项并打印日志
                finished_keys = [k for k, t in running_tasks.items() if t in done]
                for k in finished_keys:
                    task = running_tasks.pop(k)
                    exc = None
                    # task.exception() 在任务被取消或尚未结束时也会抛异常，
                    # 此时视作"无异常"处理，仅记录日志，不中断监听循环
                    try:
                        exc = task.exception()
                    except asyncio.CancelledError:
                        pass
                    except asyncio.InvalidStateError:
                        pass
                    if exc is not None:
                        print(f"[Gateway] 任务 {k} 出错: {exc}")
                    else:
                        print(f"[Gateway] 任务 {k} 已自然结束")

                # 判断核心循环是否已经停掉；如果是就退出并关闭整个网关
                core_still_alive = any(
                    t in running_tasks.values() for t in core_tasks.values()
                )
                if not core_still_alive:
                    print("[Gateway] 核心入站/出站循环已停止，准备关闭")
                    break

                # 如果没有任何任务剩下，也退出
                if not running_tasks:
                    print("[Gateway] 所有任务结束")
                    break

        finally:
            # 取消所有仍在跑的任务
            for t in running_tasks.values():
                if not t.done():
                    t.cancel()
            # 等待取消落地（忽略异常）
            if running_tasks:
                await asyncio.gather(*running_tasks.values(), return_exceptions=True)
            await self.shutdown()

    async def _process_inbound(self) -> None:
        """
        处理入站消息循环

        while True 循环：
        - 从 bus.consume_inbound() 取消息
        - 构造 session_key = f"{msg.channel}:{msg.sender_id}"
        - 按 session_key 分发到对应会话队列（同会话串行，跨会话并发）
        - 消息处理在 _handle_message 中完成
        """
        while True:
            try:
                # 从消息总线消费入站消息
                inbound_msg = await self.bus.consume_inbound()

                # 构造 session_key
                session_key = f"{inbound_msg.channel}:{inbound_msg.sender_id}"

                # 获取（或创建）该会话的消息队列和 worker：
                # 必须先创建 worker 再投递消息，保证队列始终有消费者在等待
                queue = self._session_queues.get(session_key)
                if queue is None:
                    queue = asyncio.Queue()
                    self._session_queues[session_key] = queue
                    self._session_workers[session_key] = asyncio.create_task(
                        self._session_worker(session_key, queue)
                    )

                # 投递给该会话的 worker（同会话内排队串行处理）；
                # 入队后立即返回、不等待处理结果，因此入站循环
                # 不会被某条消息的长耗时处理（多轮工具调用）阻塞
                await queue.put(inbound_msg)

            except asyncio.CancelledError:
                # 任务被取消，退出循环
                break

            except Exception as e:
                # 消费循环异常，打印日志并继续
                print(f"警告: 入站处理循环异常 - {e}")
                continue

    async def _session_worker(self, session_key: str, queue: asyncio.Queue) -> None:
        """
        单个会话的消息处理 worker

        依次处理该会话队列中的消息，保证同一会话内消息按顺序执行。
        这是"同会话串行、跨会话并发"调度模型中"串行"的一侧：
        每个 worker 只消费自己的队列，不同 worker 之间互不干扰，
        天然实现跨会话并发。

        worker 常驻运行：queue.get() 在无消息时挂起并让出事件循环，不空转占 CPU。
        单条消息处理失败仅记录日志并继续，worker 不退出——
        否则该会话后续的所有消息都会永久滞留在队列中。

        Args:
            session_key: 会话标识
            queue: 该会话的入站消息队列
        """
        while True:
            try:
                inbound_msg = await queue.get()
                await self._handle_message(session_key, inbound_msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"警告: 会话 {session_key} 处理循环异常 - {e}")
                continue

    async def _handle_message(self, session_key: str, inbound_msg) -> None:
        """
        处理单条入站消息：调用 Agent 并把回复发布到出站队列

        教育链路扩展（raw.edu_kind 存在时）：
        - meta：把 InboundMessage.raw 整体作为请求元数据传给 Agent
          （EduAgentLoop 据此组装教育上下文：年级/科目/历史/图片等）
        - on_delta：流式增量回调，把每段文本增量包装成 OutboundMessage
          （meta.edu_delta 标记）发布回消息总线，由 Web 渠道路由给
          等待中的 SSE 响应——流式增量也走总线，与最终回复同一条路
        - 结果为 dict（EduAgentLoop 返回）时，正文放 content，
          完整结果（含知识点/视频/笔记面板数据）放 meta.edu_result

        Args:
            session_key: 会话标识
            inbound_msg: 入站消息
        """
        # 从缓存获取 Agent，不存在则创建：
        # 同一会话必须复用同一 Agent 实例，内存中的对话历史才能跨消息延续；
        # 缓存命中与否由 session_key 决定，与消息来源渠道一一对应
        if session_key not in self._agents:
            agent = self.agent_factory(session_key)
            self._agents[session_key] = agent
        else:
            agent = self._agents[session_key]

        raw = inbound_msg.raw if isinstance(inbound_msg.raw, dict) else {}
        is_edu = bool(raw.get("edu_kind"))

        # 入站拦截钩子：返回 True 表示消息已被处理（如家长报告指令），
        # 不再转给 Agent；钩子自身异常时按未处理处理，照常走 Agent。
        # 教育请求（edu_kind 存在）必须跳过钩子：它们是前端结构化 API
        # 调用而非聊天文本，学生消息里若恰好含家长指令关键词（如"报告"）
        # 会被钩子吞掉，HTTP 端将永远等不到回复
        if self.inbound_hook is not None and not is_edu:
            try:
                if await self.inbound_hook(inbound_msg):
                    return
            except Exception as e:
                print(f"警告: 入站拦截钩子异常 - {e}")

        # 教育流式请求：构建增量回调，把文本增量发布回总线
        on_delta = None
        if is_edu and raw.get("stream"):
            async def on_delta(piece: str) -> None:
                await self.bus.publish_outbound(OutboundMessage(
                    channel=inbound_msg.channel,
                    chat_id=inbound_msg.chat_id,
                    content=piece,
                    meta={"edu_delta": True},
                ))

        # 调用 Agent 处理消息
        try:
            response = await agent.run(inbound_msg.content, meta=raw, on_delta=on_delta)

            # 教育链路：dict 结果 → 正文进 content，面板数据进 meta
            if isinstance(response, dict):
                outbound_msg = OutboundMessage(
                    channel=inbound_msg.channel,
                    chat_id=inbound_msg.chat_id,
                    content=str(response.get("markdown", "") or ""),
                    meta={"edu_result": response},
                )
            else:
                # 构造出站消息
                outbound_msg = OutboundMessage(
                    channel=inbound_msg.channel,
                    chat_id=inbound_msg.chat_id,
                    content=response,
                )

            # 发布到出站队列
            await self.bus.publish_outbound(outbound_msg)

        except Exception as e:
            # Agent 出错，发送友好错误消息（附带原因，便于定位）
            error_msg = OutboundMessage(
                channel=inbound_msg.channel,
                chat_id=inbound_msg.chat_id,
                content=f"抱歉，处理消息时发生了错误。请稍后重试。（{e}）",
                meta={"edu_error": str(e)} if is_edu else None,
            )
            await self.bus.publish_outbound(error_msg)

            # 打印错误日志（调试用）
            print(f"警告: Agent 处理出错 - {e}")

    async def _dispatch_outbound(self) -> None:
        """
        分发出站消息循环

        while True 循环：
        - 从 bus.consume_outbound() 取回复
        - 根据 msg.channel 找到对应的 Channel
        - 调用 channel.send(msg)
        """
        while True:
            try:
                # 从消息总线消费出站消息
                outbound_msg = await self.bus.consume_outbound()

                # 查找对应的渠道
                channel = self._channel_map.get(outbound_msg.channel)

                if channel is None:
                    print(f"警告: 渠道 '{outbound_msg.channel}' 不存在，无法发送消息")
                    continue

                # 调用渠道发送消息
                try:
                    await channel.send(outbound_msg)

                except Exception as e:
                    # 渠道发送失败，打印日志
                    print(f"警告: 渠道 '{outbound_msg.channel}' 发送失败 - {e}")

            except asyncio.CancelledError:
                # 任务被取消，退出循环
                break

            except Exception as e:
                # 分发循环异常，打印日志并继续
                print(f"警告: 出站分发循环异常 - {e}")
                continue

    async def shutdown(self) -> None:
        """
        关闭网关

        - 遍历 channels 调用 stop()
        - 取消所有会话 worker 任务
        - 清空 _agents 缓存
        """
        # 先取消会话 worker，再停渠道：
        # 尽快终止排队中/处理中的消息，之后渠道关闭就不会再触发新的入站处理，
        # 避免出现"渠道已停但 worker 还在写出站队列"的竞态
        for task in self._session_workers.values():
            if not task.done():
                task.cancel()
        self._session_workers.clear()
        self._session_queues.clear()

        # 停止所有渠道
        for channel in self.channels:
            try:
                await channel.stop()
            except Exception as e:
                print(f"警告: 渠道 '{channel.name}' 关闭失败 - {e}")

        # 清空 Agent 缓存
        self._agents.clear()