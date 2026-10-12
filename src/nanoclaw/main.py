"""
NanoClaw 入口文件

提供命令行交互界面，组装 Agent 各组件并启动对话循环。

使用方法：
    python main.py

命令：
    /exit   - 退出程序
    /clear  - 清空对话历史
    /tools  - 查看可用工具列表
"""

# 必须放在所有 import 之前：faiss-cpu 与 PyTorch 各自带了一份 OpenMP
# 运行时（libomp140.x86_64.dll / libiomp5md.dll），同进程共存会触发
# "OMP Error #15" 直接崩溃（复现路径：focus_monitor 先加载 torch，
# RAG 检索首次调用 FAISS 计算时崩溃）。此变量允许二者共存，
# 是 Windows + conda 环境下的标准 workaround。
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 必须在所有项目/SDK import 之前固定工作目录：botpy SDK 在 import 阶段就用
# os.getcwd() 把日志文件路径定格为绝对路径（botpy/logging.py 的
# DEFAULT_FILE_HANDLER），import 之后再 chdir 无法改变它的落点，导致
# botpy.log 散落在启动目录（如项目外层）。提前 chdir 让 botpy.log、
# workspace、config.json 等所有相对路径一律落在 main.py 所在目录，
# 无论从哪里启动（python src/nanoclaw/main.py 也一样）。
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

import asyncio
import os
import sys

from config import load_config
from db.database import Database
from edu_service import EducationService
from parent_report import ParentReportService
from providers.openai_compat import OpenAICompatProvider
from agent.tools.registry import ToolRegistry
from agent.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool
from agent.tools.shell import ExecTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from agent.tools.bili_video import BiliVideoSearchTool
from agent.tools.spawn import SpawnSubagentTool
from agent.tools.mcp_server import MCPClientManager
from agent.skills import SkillsLoader
from agent.context import ContextBuilder
from agent.loop import AgentLoop
from agent.edu_loop import EduAgentLoop
from agent.tools.edu_tools import EduVideoSearchTool, KnowledgeLinkTool, SaveNoteTool
from session.manager import SessionManager
from agent.memory import MemoryConsolidator
from bus.queue import MessageBus
from channels.cli import CLIChannel
from channels.feishu import FeishuChannel
from channels.web import WebChannel
from gateway import Gateway

# RAG 知识库检索器（全局共享单例：Web 管理接口、学伴对话增强、
# Agent 的 rag_search 工具都使用同一实例，保证增删查互相可见）。
# 构建失败（缺 faiss-cpu 依赖 / Embedding 初始化异常）时置 None，
# 全链路自动降级，不影响其他功能。
_rag_retriever = None


def init_rag_retriever(config):
    """按配置构建全局 RAG 检索器，失败时降级为 None 并打印原因。"""
    global _rag_retriever
    if not config.rag_enabled:
        print("[RAG] 配置未启用（rag.enabled=false），跳过")
        return
    try:
        from agent.rag.retriever import FAISSRetriever
        from agent.rag.embedding_client import SiliconFlowEmbedding

        embed_client = SiliconFlowEmbedding(api_key=config.api_key, base_url=config.base_url)
        index_dir = config.rag_index_path
        _rag_retriever = FAISSRetriever(
            embed_client,
            index_path=os.path.join(index_dir, "index.faiss"),
            metadata_path=os.path.join(index_dir, "metadata.pkl"),
            default_threshold=config.rag_threshold,
        )
        _rag_retriever.default_top_k = config.rag_top_k
        print(f"[RAG] 知识库就绪，当前共 {_rag_retriever.count()} 个块")

        # 自动导入 knowledge_base 目录下的教材（目录存在且索引为空时）
        if os.path.isdir(config.rag_knowledge_base) and _rag_retriever.count() == 0:
            added = _rag_retriever.load_from_directory(config.rag_knowledge_base)
            if added:
                print(f"[RAG] 从 {config.rag_knowledge_base} 自动导入 {added} 个块")
    except ImportError as e:
        _rag_retriever = None
        print(f"[RAG] 依赖未安装，RAG 功能降级禁用（可 pip install faiss-cpu）: {e}")
    except Exception as e:
        _rag_retriever = None
        print(f"[RAG] 初始化失败，RAG 功能降级禁用: {e}")


def build_agent(config, session_key: str, mcp_manager: MCPClientManager = None) -> AgentLoop:
    """
    组装 Agent 实例
    Args:
        config: 配置对象（由 load_config() 返回）
        session_key: 会话标识，格式为 "{channel}:{sender_id}"
        mcp_manager: MCP 客户端管理器（可选）
    加载配置、创建各组件并组装 AgentLoop：
    1. 加载配置文件
    2. 验证 API 密钥
    3. 创建 LLM Provider
    4. 注册文件系统工具
    5. 注册 MCP 工具（如果有）
    6. 创建上下文构建器
    7. 组装 AgentLoop

    Returns:
        AgentLoop: 组装完成的 Agent 实例
    """
    # 加载配置：覆盖传入的 config 参数，统一以磁盘上的 config.json 为准，
    # 保证所有会话创建的 Agent 使用同一份配置
    config = load_config()

    # 验证 API 密钥
    if not config.api_key:
        print("错误: 未配置 API 密钥")
        print("请在 config.json 中设置 api_key，或设置环境变量 NANOCLAW_API_KEY")
        sys.exit(1)

    # 创建 LLM Provider
    provider = OpenAICompatProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )

    # 创建工具注册表并注册文件系统工具
    tools = ToolRegistry()
    tools.register(ReadFileTool(config.workspace))
    tools.register(WriteFileTool(config.workspace))
    tools.register(ListDirTool(config.workspace))
    tools.register(ExecTool(config.workspace))
    tools.register(WebSearchTool())
    tools.register(WebFetchTool())
    tools.register(BiliVideoSearchTool())

    # RAG 知识库检索工具（检索器就绪时才注册，供 Agent 查教材）
    if _rag_retriever is not None:
        from agent.tools.rag_tool import RAGSearchTool
        tools.register(RAGSearchTool(_rag_retriever))

        # 注册子 Agent 工具
    def create_provider(model=None):
        """Provider 工厂函数，子 Agent 用。"""
        # 子 Agent 默认用 subagent 模型（更便宜）
        if model is None and config.models:
            model = config.models.get("subagent", config.model)
        return OpenAICompatProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=model or config.model
        )

    tools.register(SpawnSubagentTool(
        provider_factory=create_provider,
        tools_registry=tools,
        workspace=config.workspace,
    ))

    # 注册 MCP 工具（如果有）；数量并入下方工具总表，不单独刷行
    if mcp_manager:
        mcp_tools = mcp_manager.get_tools()
        for mcp_tool in mcp_tools:
            tools.register(mcp_tool)

    # 加载技能摘要
    skills_loader = SkillsLoader(
        skills_dir=os.path.join(config.workspace, "skills")
    )
    skills_summary = skills_loader.build_skills_summary()

    # 如果发现技能，打印数量
    if skills_summary:
        skills_list = skills_loader.list_skills()
        print(f"{session_key} 已发现技能: {len(skills_list)} 个")

    # 创建会话管理器
    session_manager = SessionManager(
        sessions_dir=os.path.join(config.workspace, "workspace", "sessions")
    )

    # session_key = "cli:direct"
    # 如果有历史，显示恢复提示
    existing_history = session_manager.get_history(session_key)
    if existing_history:
        print(f"[{session_key}] 已恢复 {len(existing_history)} 条历史消息")

    # 打印已注册的工具：一行 4 个、列对齐排版，避免十几个名字挤成
    # 一整行 Python 列表难以阅读
    tool_names = tools.list_tools()
    print(f"[{session_key}] 已注册 {len(tool_names)} 个工具:")
    for i in range(0, len(tool_names), 4):
        print("     " + "  ".join(f"{name:<20}" for name in tool_names[i:i + 4]))

    # 创建上下文构建器
    context = ContextBuilder(
        workspace=config.workspace,
        identity_file=config.identity_file,
        skills_summary=skills_summary,
    )

    # 创建 Token 压缩器
    consolidator = MemoryConsolidator(
        provider=provider,
        workspace=config.workspace,
        token_budget=16000
    )

    # 组装 AgentLoop
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        context=context,
        session_manager=session_manager,
        model=config.model,
        max_iterations=config.max_iterations,
        # session_key="cli:direct",
        session_key=session_key,
    )

    agent.consolidator = consolidator  # 挂载压缩器

    return agent


def build_edu_agent(session_key: str) -> EduAgentLoop:
    """
    组装教育 Agent（EduAgentLoop）

    教育助手的三大核心链路（学伴对话 / 题目批改 / 学习计划）全部经由
    核心骨架执行：Web 渠道发布入站消息 → Gateway 按 session_key 调本
    工厂 → EduAgentLoop 复用 AgentLoop 的推理-工具循环体（function
    calling、防爆熔断、流式、取消响应）完成生成。

    与 build_agent（CLI/飞书/WebSocket 通用 Agent）的区别：
    - 工具集：只注册教育工具（视频搜索 / 百科链接 / 笔记保存 / RAG 查教材），
      不注册文件系统 / Shell / 联网搜索等通用工具——学生对话不需要也不应该
      拿到这些能力
    - 无本地会话：EduAgentLoop 不读写 JSONL 会话文件，对话历史每次随
      请求元数据传入（SQLite 由 Web 渠道负责持久化），因此同一个
      session_key 下多请求共享本实例是安全的
    - 人设：教育 prompt 常量由 edu_service 承载，不走 identity.md

    Args:
        session_key: 会话标识，形如 "web:edu:<username>"

    Returns:
        EduAgentLoop: 组装完成的教育 Agent 实例
    """
    config = load_config()

    if not config.api_key:
        print("错误: 未配置 API 密钥")
        print("请在 config.json 中设置 api_key，或设置环境变量 NANOCLAW_API_KEY")
        sys.exit(1)

    provider = OpenAICompatProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )

    # 教育工具注册表：function calling 取代旧版"输出标记 + 后处理"
    tools = ToolRegistry()
    tools.register(EduVideoSearchTool())   # bili_video_search → 视频面板
    tools.register(KnowledgeLinkTool())    # knowledge_link    → 知识点面板
    tools.register(SaveNoteTool())         # save_note         → 本地笔记文件

    # RAG 查教材工具（检索器就绪时注册；何时检索由 Agent 在循环中自主决策）
    if _rag_retriever is not None:
        from agent.tools.rag_tool import RAGSearchTool
        tools.register(RAGSearchTool(_rag_retriever))

    agent = EduAgentLoop(
        provider=provider,
        tools=tools,
        model=config.model,
        vision_model=(config.models or {}).get("vision"),
        # 工具循环上限 6 轮：教材检索 + 视频 + 百科 + 笔记通常一轮并发调用，
        # 防呆余量即可
        max_iterations=6,
    )
    print(f"[{session_key}] 教育 Agent 就绪（工具: {', '.join(tools.list_tools())}）")
    return agent


# 新版：通过 Gateway 启动
async def async_main() -> None:
    """
    异步主入口

    启动 MCP Server、组装 Agent、启动 Gateway。
    """
    # 加载配置文件（先于 banner：模型/服务信息直接从配置读取，
    # 避免换模型后 banner 还显示写死的旧名字）
    config = load_config()

    # 启动 banner
    print("=" * 60)
    print()
    print("        NanoClaw · 中小学生智能学伴")
    print(f"        模型: {config.model}")
    print(f"        服务: {config.base_url}")
    print()
    print("=" * 60)

    # 构建 RAG 检索器（Web 管理接口 / 学伴对话增强 / Agent 工具共用）
    init_rag_retriever(config)

    # 消息总线是渠道与网关之间的唯一通信枢纽，渠道与网关彼此不直接引用
    bus = MessageBus()

    # 启动 MCP Server（如果有配置）
    mcp_manager = None
    if config.mcp_servers:
        mcp_manager = MCPClientManager(config.mcp_servers)
        await mcp_manager.connect_all(timeout=30)
        # await mcp_manager.connect_all(timeout=30.0)

    # 定义 Agent 工厂函数：教育会话返回 EduAgentLoop，其余返回通用 Agent。
    # 教育会话的 session_key 以 ":edu:" 为标记（web.py 的 _edu_sender_id
    # 生成 "web:edu:<username>"，详见 WebChannel._edu_sender_id）
    def create_agent(session_key: str) -> AgentLoop:
        if ":edu:" in session_key:
            return build_edu_agent(session_key)
        return build_agent(config, session_key, mcp_manager)

    # 注册渠道
    cli_channel = CLIChannel(bus)
    channels = [cli_channel]

    # 飞书渠道（如果配置了 feishu app_id / app_secret 就自动启用）
    if config.feishu_app_id and config.feishu_app_secret:
        feishu_channel = FeishuChannel(bus, config.feishu_app_id, config.feishu_app_secret)
        channels.append(feishu_channel)
        print("[启动] 已启用飞书渠道")
    else:
        print("[启动] 未配置飞书，跳过飞书渠道")

    # Web 渠道（默认启用，不需要外部凭证）
    if config.web_enabled:
        # 教育助手辅助服务：承载错题集分析与 STT/TTS/RAG 问答的 Provider
        # 复用；核心对话/批改/计划已迁移到 EduAgentLoop（经网关与总线），
        # 视觉模型名供流式接口建流前校验图片请求
        edu_provider = OpenAICompatProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
        )
        education_service = EducationService(
            edu_provider,
            vision_model=(config.models or {}).get("vision"),
        )
        web_channel = WebChannel(
            bus,
            host=config.web_host,
            port=config.web_port,
            education_service=education_service,
            rag_retriever=_rag_retriever,
            stt_model=config.stt_model,
            tts_model=config.tts_model,
            tts_voice=config.tts_voice,
        )
        channels.append(web_channel)
        # 0.0.0.0 是监听地址，浏览器里打不开，显示 127.0.0.1
        display_host = "127.0.0.1" if config.web_host == "0.0.0.0" else config.web_host
        print(f"[启动] 已启用 Web 渠道: http://{display_host}:{config.web_port}")
        print(f"[启动] 教育助手已启用: http://{display_host}:{config.web_port}/edu")
    else:
        print("[启动] Web 渠道已禁用")

    # 启动网关
    # 家长报告服务：独立的 SQLite 连接 + 独立 Provider（优先用 cheap 模型
    # 生成总结），拦截飞书/QQ 的绑定与总结指令，并启动每日定时推送任务
    parent_db = Database("nanoclaw.db")
    await parent_db.connect()
    await parent_db.init_db()
    cheap_model = (config.models or {}).get("cheap")
    report_provider = OpenAICompatProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=cheap_model or config.model,
    )
    parent_service = ParentReportService(
        provider=report_provider,
        db=parent_db,
        bus=bus,
        model=cheap_model,
        fallback_model=config.model,
        push_hour=config.parent_push_hour,
        push_minute=config.parent_push_minute,
    )
    asyncio.create_task(parent_service.daily_push_task())

    gateway = Gateway(bus, channels, create_agent, inbound_hook=parent_service.handle_message)

    # 预创建 CLI Agent，让初始化信息在启动时就显示
    cli_session_key = "cli:local"
    agent = create_agent(cli_session_key)
    gateway._agents[cli_session_key] = agent

    # 注入工具列表和清空回调给 CLI 渠道
    cli_channel.tool_names = agent.tools.list_tools()
    cli_channel._clear_callback = lambda: agent.clear_history()

    try:
        await gateway.run()
    except KeyboardInterrupt:
        print("\n[NanoClaw] 正在退出...")
    finally:
        # 清理 MCP 连接
        if mcp_manager:
            await mcp_manager.shutdown()


def main() -> None:
    """
    主入口（同步包装）

    先固定工作目录、重配置终端编码（Windows GBK 兼容），
    再启动异步主入口 async_main()，由其完成 banner 打印、Agent 组装
    与 Gateway 消息循环。
    """
    # 工作目录已在模块顶部（所有 import 之前）固定为 main.py 所在目录，
    # 保证 config.json 的 workspace="." 等相对路径解析正确，botpy.log
    # 等 SDK 日志也不会散落在外。

    # 中文 Windows 终端默认 GBK 编码，无法编码 Agent 循环打印的 emoji（💭🛠️✅），
    # 导致 print 抛 UnicodeEncodeError，网关误报"处理消息时发生错误"。
    # 统一重配置为 UTF-8，无法编码的字符用替换符兜底
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # 无论哪条路径退出（Ctrl+C / asyncio.run 正常返回 / 未预期异常），
    # 都在 finally 里强制结束进程：lark、botpy、企微等 SDK 的 WebSocket
    # 网络线程是非守护线程，主循环结束后它们仍在运行，会把进程"吊住"
    # 形成僵尸残留（现象：Ctrl+C 后进程还在，继续抢飞书消息但不回复）。
    exit_code = 0
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[NanoClaw] 正在退出...")
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        os._exit(exit_code)  # 强制退出，杀掉全部残留的 SDK 子线程


if __name__ == "__main__":
    main()