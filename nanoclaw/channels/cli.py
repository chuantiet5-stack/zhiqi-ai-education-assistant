"""
CLI 渠道适配器模块

CLIChannel 提供命令行交互界面，支持用户通过终端与 Agent 对话。
特性：
- 支持命令：/exit（退出）、/clear（清空历史）、/tools（查看工具列表）
- 使用 asyncio.Event 同步输入和回复，形成严格的一问一答节奏
- input() 用 asyncio.to_thread 包装，避免阻塞事件循环
- 当 stdin 不是交互式终端（后台启动 / IDE 输出面板 / 重定向）时，
  不主动读取 input()，也不打印交互 banner，只作为静默渠道保留。

线程模型特点：
    阻塞式 input() 无法被 asyncio 直接调度，故放入线程池
    （asyncio.to_thread）执行——事件循环在等待用户输入期间
    仍可处理其他渠道的任务；输入与回复之间用 asyncio.Event 在
    协程层面同步：发布用户消息后挂起 start() 循环，直到 send()
    打印完回复再 set，保证终端上"一问一答"的串行体验。

使用示例：
    bus = MessageBus()
    cli_channel = CLIChannel(bus)

    # 注入工具列表和清空回调（由 main.py 完成）
    cli_channel.tool_names = ["read_file", "write_file"]
    cli_channel._clear_callback = lambda: print("已清空")

    # 启动渠道
    asyncio.create_task(cli_channel.start())
"""

import asyncio
import sys
from bus.queue import MessageBus, InboundMessage, OutboundMessage
from channels.base import Channel


class CLIChannel(Channel):
    """
    CLI 渠道适配器

    提供命令行交互界面，监听用户输入并发布到 MessageBus。

    设计意图：
        与网络渠道不同，CLI 天然只有一个"用户"和一个会话，因此
        sender_id/chat_id 使用固定占位值（local/direct）；通过
        Event 阻塞 start() 循环来保证上一条回复打印完毕前不会
        读取下一条输入，避免用户输入与 Agent 回复在终端上交错。
    """

    def __init__(self, bus: MessageBus) -> None:
        """
        初始化 CLI 渠道。

        Args:
            bus: 消息总线实例

        Attributes（实例级补充）:
            _response_event: 协程间同步事件，start() 在发布用户消息后
                等待它，send() 打印回复后触发它
            tool_names: /tools 命令展示的工具列表，由 main.py 注入
            _clear_callback: /clear 命令触发的清空历史回调，由 main.py 注入
            _interactive: stdin/stdout 是否均为交互式终端（TTY）
        """
        super().__init__(name="cli", bus=bus)

        # 用于同步"等待回复完成"
        self._response_event = asyncio.Event()

        # 预留属性，由 main.py 在创建后注入
        self.tool_names: list[str] = []
        self._clear_callback: callable | None = None

        # 是否为交互式终端：stdin 和 stdout 必须同时是 TTY。
        # 之所以两者都检查，是因为重定向（如 python main.py > log）
        # 或后台守护进程中 input() 会立即返回 EOF 或读不到数据，
        # 与其让交互循环空转报错，不如提前降级为静默模式
        self._interactive = sys.stdin.isatty() and sys.stdout.isatty()

    async def start(self) -> None:
        """
        启动 CLI 交互循环

        交互终端（TTY）：循环读取用户输入，支持命令和普通对话；
        非交互终端（重定向/后台启动/IDE 内置非交互 stdout）：
            不读取 stdin，只是作为一个"永不结束"的渠道挂起，
            避免因为 input() EOF 让整个 CLI 任务立即完成。
        """
        if not self._interactive:
            # 静默挂起：只响应 send()（如果有其他渠道发布到 cli，
            # 也能照常打印），不主动退出
            print("[CLIChannel] 标准输入非交互式，已以静默模式启动")
            print("[CLIChannel] （如需终端对话，请在实际终端运行 python main.py）")
            await self._wait_forever()
            return

        # 稍候片刻再打印就绪横幅：飞书/Web 等渠道在 Gateway 里是并发
        # 启动的，不等待的话它们的连接日志会插到"你>"提示符后面，
        # 把首轮对话界面搅乱
        await asyncio.sleep(0.3)
        print("\n" + "-" * 60)
        print("  ✓ 就绪，开始对话（/exit 退出 · /clear 清空 · /tools 查看工具）")
        print("-" * 60)

        while True:
            try:
                # 用 asyncio.to_thread 包装 input()，避免阻塞事件循环
                user_input = await asyncio.to_thread(input, "\n你> ")
                user_input = user_input.strip()

                # 空输入跳过
                if not user_input:
                    continue

                # 处理命令
                if user_input == "/exit":
                    print("再见！")
                    break

                if user_input == "/clear":
                    # 调用清空回调（如果有）
                    if self._clear_callback:
                        self._clear_callback()
                    print("对话历史已清空")
                    continue

                if user_input == "/tools":
                    # 打印工具列表
                    print(f"可用工具: {', '.join(self.tool_names)}")
                    continue

                # 正常输入 → 构造入站消息
                inbound_msg = InboundMessage(
                    channel="cli",
                    sender_id="local",
                    chat_id="direct",
                    content=user_input,
                )

                # 发布到消息总线
                await self.bus.publish_inbound(inbound_msg)

                # 重置 Event，等待回复完成：
                # 挂起本循环直到 send() 打印回复后 set，既保证一问一答
                # 节奏，也避免等待期间循环回头读取下一条输入造成输出交错
                self._response_event.clear()
                await self._response_event.wait()

            except KeyboardInterrupt:
                # Ctrl+C 优雅退出
                print("\n\n再见！")
                break

            except EOFError:
                # 输入结束（如管道输入完毕）
                print("\n再见！")
                break

    async def _wait_forever(self) -> None:
        """
        静默模式：永不完成的挂起协程，等待外部取消。

        之所以要永久挂起而不是直接 return：start() 返回意味着渠道
        生命周期结束，若在后台模式下立即返回，整个程序可能随之退出；
        挂起可让进程保持存活，send() 仍能把其他渠道产生的回复
        打印到 stdout。
        """
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass

    async def send(self, message: OutboundMessage) -> None:
        """
        发送回复给用户

        TTY 模式：打印回复，通知 start() 可以继续读取下一个输入。
        静默模式：只打印回复（如果 stdout 仍连接），不用 Event 通知。
        """
        # 打印回复
        print(f"\n🤖 {message.content}")

        # 只在交互模式下通知 start() 可以继续了：
        # 静默模式中 start() 已挂起在 _wait_forever，没有协程在等待
        # 这个事件，set 它没有意义，故跳过
        if self._interactive:
            self._response_event.set()