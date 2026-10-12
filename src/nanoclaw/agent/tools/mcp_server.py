"""
MCP (Model Context Protocol) 工具模块

提供 MCPClientManager 和 MCPTool，把外部 MCP Server 的工具集成进 NanoClaw：
启动时建立连接并拉取工具清单，包装成标准 Tool 注册给 Agent，
运行期透传 LLM 的工具调用，退出时统一清理子进程与连接。

连接流程（stdio 传输）：
1. 依据配置以子进程方式拉起 MCP Server（command + args），
   通过其 stdin/stdout 交换 JSON-RPC 消息，无网络端口暴露；
2. 客户端发送 initialize 请求与 Server 完成握手（协商协议版本与能力），
   握手成功前协议不允许调用 list_tools / call_tool；
3. list_tools 拉取工具清单，逐个包装为 MCPTool 注册进工具注册表。

关键约束：
- 所有 print 不使用 emoji（Windows GBK 会崩），用 [MCP]、[!] 代替
- ClientSession 必须手动 __aenter__()：不进入上下文则内部消息循环 Task 不会
  启动，initialize() 将永远等不到响应而卡死
"""

import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.tools.base import Tool


class MCPTool(Tool):
    """
    MCP 工具包装器

    将 MCP Server 提供的工具包装为 NanoClaw 的 Tool 接口。
    工具名格式: {server_name}__{tool_name}，避免跨 server 冲突。
    """

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        client_manager: "MCPClientManager",
    ):
        """
        Args:
            server_name: MCP Server 名称（配置中的 key）
            tool_name: MCP Server 提供的工具名
            tool_description: 工具描述（从 MCP list_tools 获取）
            input_schema: 工具参数 schema（从 MCP list_tools 获取）
            client_manager: MCP 客户端管理器，用于执行工具调用
        """
        self.server_name = server_name
        self.tool_name = tool_name
        self._description = tool_description
        self._input_schema = input_schema
        self.client_manager = client_manager

    @property
    def name(self) -> str:
        """工具名称，格式: {server}__{tool}"""
        return f"{self.server_name}__{self.tool_name}"

    @property
    def description(self) -> str:
        """工具描述"""
        return self._description

    @property
    def parameters(self) -> dict:
        """工具参数 schema，直接使用 inputSchema"""
        return self._input_schema

    async def execute(self, **kwargs: Any) -> str:
        """
        执行 MCP 工具

        通过 client_manager 调用远程 MCP Server 的工具。

        Args:
            **kwargs: 工具参数

        Returns:
            str: 工具执行结果
        """
        return await self.client_manager.call_tool(
            self.server_name, self.tool_name, kwargs
        )


class MCPClientManager:
    """
    MCP 客户端管理器

    管理 MCP Server 的连接生命周期和工具调用。
    支持 stdio 传输协议，通过 StdioServerParameters 启动子进程。

    使用流程：
    1. 传入 mcp_config（格式同 Claude Desktop）
    2. 调用 connect_all() 启动所有 server
    3. 调用 get_tools() 获取所有 MCPTool
    4. 调用 call_tool() 执行工具
    5. 调用 shutdown() 清理连接
    """

    def __init__(self, mcp_config: dict):
        """
        Args:
            mcp_config: MCP Server 配置字典，格式：
                {
                    "server_name": {
                        "command": "python",
                        "args": ["path/to/server.py"]
                    }
                }
        """
        self.mcp_config = mcp_config
        self._context_managers = {}  # stdio_client 上下文管理器
        self._session_managers = {}  # ClientSession 上下文管理器
        self._sessions = {}  # 已连接的 session
        self._tools = []  # MCPTool 列表

    async def _connect_one(self, server_name: str, server_config: dict) -> None:
        """
        连接单个 MCP Server

        流程：
        1. 创建 StdioServerParameters
        2. 调用 stdio_client() 获取上下文管理器，手动 __aenter__ 获取读写流
        3. 创建 ClientSession，手动 __aenter__ 启动内部消息循环
        4. 调用 session.initialize() 初始化连接
        5. 调用 session.list_tools() 获取工具列表
        6. 为每个工具创建 MCPTool

        Args:
            server_name: Server 名称
            server_config: Server 配置（command、args）

        Raises:
            Exception: 连接失败、初始化失败、工具加载失败
        """
        print(f"[MCP] 正在连接 {server_name}...")

        # 1. 创建 StdioServerParameters
        #   env 处理：基于当前进程 os.environ 拷贝，再叠加 server_config 里的覆盖值；
        #   若用户直接传 None，则不传 env，子进程将继承当前环境（更安全、符合 stdio 默认）。
        #   为什么不整体替换环境：常见 server 由 uvx/npx/python 拉起，
        #   依赖 PATH、HOME 等变量定位可执行文件，全量替换会导致子进程启动即失败
        import os as _os

        raw_env = server_config.get("env")
        if raw_env is None:
            resolved_env = None
        else:
            resolved_env = dict(_os.environ)
            # 展开 config 里可能写的 %VAR% / $VAR
            for k, v in raw_env.items():
                resolved_env[k] = _os.path.expandvars(str(v))

        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=resolved_env,
        )

        # 2. stdio_client 上下文管理器，手动 __aenter__
        # 不能用 async with：连接生命周期要跨越 connect_all 建立连接、
        # Agent 多轮对话期间反复调用、shutdown 统一关闭三个阶段，
        # 无法收拢在单个 async with 块内，因此保存上下文对象延迟退出
        ctx = stdio_client(server_params)
        read_stream, write_stream = await ctx.__aenter__()
        self._context_managers[server_name] = ctx

        # 3. ClientSession 上下文管理器，手动 __aenter__
        # __aenter__ 会启动内部消息接收循环 Task；不进入上下文则
        # initialize() 的响应永远无人读取，调用将永久卡死
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        self._session_managers[server_name] = session

        # 4. 初始化连接（必须！）
        # MCP 协议握手：客户端先发 initialize 请求协商协议版本与能力，
        # Server 应答后才允许调用 list_tools / call_tool 等后续方法
        await session.initialize()

        # 5. 获取工具列表
        tools_response = await session.list_tools()

        # 6. 创建 MCPTool
        for tool_info in tools_response.tools:
            # 兼容 mcp 1.x / 2.x：1.x 是 inputSchema（驼峰），2.x 改为 input_schema（下划线）
            schema = getattr(tool_info, "inputSchema", None) or getattr(
                tool_info, "input_schema", {}
            )
            mcp_tool = MCPTool(
                server_name=server_name,
                tool_name=tool_info.name,
                tool_description=tool_info.description or "",
                input_schema=schema or {},
                client_manager=self,
            )
            self._tools.append(mcp_tool)
            # 描述只取第一行并截断：完整描述含多行 Args/Returns，
            # 直接打印会把启动日志撑成十几行乱码
            desc = (tool_info.description or "").strip()
            desc_first_line = desc.splitlines()[0][:40] if desc else ""
            print(f"[MCP]   · {mcp_tool.name}  {desc_first_line}")

        # 保存 session
        self._sessions[server_name] = session
        print(f"[MCP] {server_name} 连接成功，共 {len(tools_response.tools)} 个工具")

    async def connect_all(self, timeout: float = 30.0) -> None:
        """
        连接所有 MCP Server

        遍历配置，每个 server 用 asyncio.timeout(timeout) 包裹（避免
        wait_for 新建 Task 导致 anyio cancel scope 跨 Task 退出报错）。
        超时或异常则跳过，打印警告。

        Args:
            timeout: 单个 server 连接超时秒数
        """
        if not self.mcp_config:
            print("[MCP] 未配置 MCP Server，跳过")
            return

        print(f"[MCP] 开始连接 {len(self.mcp_config)} 个 MCP Server...")

        for server_name, server_config in self.mcp_config.items():
            try:
                # 用 asyncio.timeout 而非 asyncio.wait_for：wait_for 会把协程
                # 包装成新 Task 执行，而 stdio_client 内部 anyio 的 cancel scope
                # 绑定创建它的 Task，跨 Task 取消会抛 RuntimeError；
                # asyncio.timeout 直接在当前 Task 内生效，取消沿同一 Task 传播
                async with asyncio.timeout(timeout):
                    await self._connect_one(server_name, server_config)
            # 超时/失败都只跳过当前 server：个别 server 不可用不应阻塞
            # 整个 Agent 的启动，其余 server 的工具仍可正常使用
            except TimeoutError:
                print(f"[!] MCP Server {server_name} 连接超时（{timeout}s），跳过")
            except Exception as e:
                print(f"[!] MCP Server {server_name} 连接失败: {e}，跳过")

    def get_tools(self) -> list[Tool]:
        """
        获取所有 MCPTool

        Returns:
            list[Tool]: MCPTool 列表
        """
        return self._tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> str:
        """
        调用 MCP Server 的工具

        Args:
            server_name: Server 名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            str: 工具执行结果（文本格式）
        """
        session = self._sessions.get(server_name)
        if session is None:
            return f"错误: MCP Server '{server_name}' 未连接"

        try:
            # 调用 session.call_tool
            result = await session.call_tool(tool_name, arguments)

            # 提取文本内容
            if result.content and len(result.content) > 0:
                # content 是 list[Content]，提取 text 类型的内容
                text_parts = []
                for content_item in result.content:
                    if hasattr(content_item, "text"):
                        text_parts.append(content_item.text)
                    elif hasattr(content_item, "data"):
                        # 处理其他类型（如 image）
                        text_parts.append(str(content_item.data))
                return "\n".join(text_parts) if text_parts else "（无文本内容）"
            else:
                return "（无返回内容）"

        except Exception as e:
            # 异常转为错误字符串而非抛出（与 ToolRegistry.execute 兜底策略一致）：
            # 保证 Agent 主循环不因远程 server 的异常而中断
            return f"错误: MCP 工具调用失败 - {e}"

    async def shutdown(self) -> None:
        """
        清理所有连接

        按连接时的反向顺序释放：先退出 ClientSession（内层），再退出
        stdio_client 上下文（外层）。注意 anyio 的 cancel scope 必须在
        创建它的 task 中退出，因此任何跨 task 调用的 cancel scope 报错
        都直接吞掉：对生命周期末尾而言不影响工具调用的正确性。
        """
        print("[MCP] 正在关闭 MCP Server 连接...")

        # 按注册反向顺序，同一 server 先退出内层 session 再退出外层 ctx
        # （因为任何 server 的 ctx 与 session 一一对应，只要顺序对即可）
        for server_name in reversed(list(self._session_managers.keys())):
            session = self._session_managers[server_name]
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass  # anyio cancel scope 跨 task 等关闭期噪音直接忽略
            ctx = self._context_managers.get(server_name)
            if ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass

        print("[MCP] MCP Server 连接已全部关闭")