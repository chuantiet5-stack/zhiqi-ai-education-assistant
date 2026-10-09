"""Web 渠道实现模块，提供 HTTP 页面、教育助手 API 与 WebSocket 接口。

基于 FastAPI + uvicorn 提供：
- GET /         登录页（channels/web_ui/index.html，注册/登录后进入 /edu）
- GET /edu      知契页面（channels/web_ui/edu.html，三栏学习工作台）
- /api/auth/*   注册 / 登录 / 资料修改（SQLite users 表，bcrypt 加密密码）
- /api/edu/*    批改 / 计划 / 学伴对话（带 SQLite 持久化与历史恢复）
- /static/*     页面静态资源（本地 marked.min.js 等，避免 CDN 不可达）
- WebSocket /ws Agent 聊天通道

线程模型特点：
    FastAPI/uvicorn 与本渠道运行在同一个 asyncio 事件循环中，
    WebSocket 收发均为原生协程，全程零线程切换。

会话模型特点：
    每个浏览器标签页建立 WebSocket 连接时都会生成一个独立的
    uuid 作为 client_id，并作为 sender_id / chat_id 写入
    InboundMessage；回复时通过 chat_id 从连接表 _connections
    找回对应连接下发。因此多标签页/多用户互不串扰，
    天然支持并发会话。

学伴对话持久化：
    对话历史的"唯一真相源"是 SQLite（nanoclaw.db）：
    - 浏览器 cookie（edu_session）定位会话，页面关闭重开自动续接
    - 登录用户按请求里的 username 绑定会话，同一浏览器多账号互不串扰
    - 未登录请求回退到免登录访客账号（web_guest）
    - 前端只负责"最新一条用户消息"的透传，历史统一从数据库加载
"""

import asyncio
import base64
import json
import re
import secrets
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bus import InboundMessage, MessageBus, OutboundMessage
from db.database import Database
from edu_service import (
    EducationService,
    EducationServiceError,
    ChatRequest,
    GradeRequest,
    PlanRequest,
    WRONGBOOK_SUBJECTS,
    SUBJECTS_BY_GRADE,
    MAX_IMAGE_DATA_URL_LENGTH,
    validate_grade_subject,
)
from focus_monitor import FocusMonitor

# RAG 检索器为可选依赖（缺 faiss-cpu 等包时降级，不影响其他功能）
try:
    from agent.rag.retriever import FAISSRetriever
    from agent.rag.doc_parser import DocParseError, parse_document
except Exception:  # ImportError 及 faiss 底层加载异常一并兜底
    FAISSRetriever = None

from .base import Channel

# 学伴对话的浏览器侧会话 cookie 名（httponly，前端无需读取）
EDU_SESSION_COOKIE = "edu_session"
# cookie 有效期（180 天），期内同一浏览器自动续接同一 SQLite 会话
EDU_SESSION_MAX_AGE = 180 * 24 * 3600
# 教育页免登录访客账号：未带登录态的学伴请求共用一个 user，会话隔离靠 cookie
EDU_GUEST_USERNAME = "web_guest"


# ============ 用户认证请求模型（登录页 / 资料弹窗的接口契约） ============

class AuthRegister(BaseModel):
    """注册请求体（注册仅需账号密码，年级进入后可在资料里设置）。"""

    username: str = Field(..., min_length=2, max_length=20, description="用户名（2-20 字符）")
    password: str = Field(..., min_length=4, max_length=64, description="密码（至少 4 位）")
    grade_level: int = Field(3, ge=1, le=9, description="年级编号（注册不再收集，默认三年级）")


class AuthLogin(BaseModel):
    """登录请求体。"""

    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class AuthProfile(BaseModel):
    """资料更新请求体（学伴页头像处的"编辑资料"弹窗）。

    账号（username）是登录凭证、users 表的唯一键，不可修改；
    可修改的是昵称（nickname，首次默认"小同学"）、头像（emoji 或
    本地导入的压缩图片 data URL）和年级。
    """

    username: str = Field(..., description="账号（登录凭证，用于定位用户，不可改）")
    grade_level: int = Field(3, ge=1, le=9, description="年级编号：1-6 小学，7-9 初中")
    nickname: str | None = Field(None, min_length=1, max_length=20, description="昵称（不改时省略）")
    avatar: str | None = Field(
        None, max_length=200_000, description="头像：emoji 短字符串或本地导入的压缩图片 data URL"
    )


class WrongBookItem(BaseModel):
    """错题集收录项（前端 extractWrongItems 提取的单道错题）。"""

    question: str = Field(..., min_length=1, max_length=300, description="题干摘要")
    verdict: str = Field(..., min_length=1, max_length=20, description='判定文案（如 "❌ 错误"）')
    markdown: str = Field(..., min_length=1, max_length=20_000, description="完整批改讲解 Markdown 块")
    subject: str | None = Field(None, description="学科（缺省由后端自动分类）")


class WrongBookAdd(BaseModel):
    """错题集批量收录请求体（支持一次收录多道，也兼容单道）。"""

    username: str = Field(..., description="登录用户名（定位归属用户）")
    items: list[WrongBookItem] = Field(..., min_length=1, max_length=50, description="错题列表")


class TTSRequest(BaseModel):
    """朗读请求体（学伴回复气泡旁 🔊 按钮的载荷）。"""

    text: str = Field(..., min_length=1, max_length=20_000, description="待朗读的 Markdown 文本")


# TTS 单次合成文本上限（CosyVoice2 对长文本按 token 有限制，
# 且朗读超过一两屏对孩子也不友好），超长时截断
TTS_MAX_CHARS = 500


def _md_to_plain(md: str) -> str:
    """轻量 Markdown → 朗读文本（去代码块/图片/链接/标记符号）。

    朗读场景下代码、图片、表格分隔线等内容念出来没有意义，
    做一层清洗让语音自然；列表符号转成口语化的"·"。
    """
    text = re.sub(r"```[\s\S]*?```", "这里有一段代码示例。", md)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"^\s*[-:| ]+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|", "，", text)
    text = re.sub(r"^\s*[-*+]\s+", "· ", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _user_payload(row) -> dict:
    """users 表行 → 前端登录态对象（字段与 2.html 登录页 / 1.html 学伴页约定一致）。"""
    return {
        "ok": True,
        "username": row["username"],
        "nickname": row["nickname"] or row["username"],
        "grade_level": row["grade_level"] if row["grade_level"] is not None else 3,
        "avatar": row["avatar"] or "🧑‍🎓",
    }


class WebChannel(Channel):
    """
    Web 渠道，通过 WebSocket 接收和发送消息。

    设计意图：
        用 client_id（uuid）同时充当 sender_id 和 chat_id，把
        "浏览器标签页"抽象成与其他渠道一致的会话概念；连接表
        _connections 是发送方向的关键——回复到达时凭 chat_id 找回
        当初的连接，无需在 OutboundMessage 中额外携带套接字对象。
    """

    def __init__(
        self,
        bus: MessageBus,
        host: str = "0.0.0.0",
        port: int = 8080,
        education_service: "EducationService | None" = None,
        rag_retriever: "FAISSRetriever | None" = None,
        stt_model: str = "FunAudioLLM/SenseVoiceSmall",
        tts_model: str = "FunAudioLLM/CosyVoice2-0.5B",
        tts_voice: str = "anna",
    ):
        """初始化 Web 渠道。

        Args:
            bus: 消息总线实例
            host: 监听地址
            port: 监听端口
            education_service: 教育助手服务（可选）。注入后额外挂载 /edu
                页面与批改/计划/学伴对话 API；为 None 时完全不注册教育路由，
                聊天功能不受影响
            rag_retriever: RAG 知识库检索器（可选）。注入后挂载 /api/rag/*
                管理与检索接口（rag_admin.html 的数据来源）
            stt_model: 语音转文字模型（硅基流动 SenseVoice）
            tts_model: 文字转语音模型（硅基流动 CosyVoice2）
            tts_voice: TTS 预置音色名

        Attributes:
            _connections: 活跃连接表，key 为 client_id，value 为对应的
                WebSocket；连接建立时写入、断开/发送失败时移除
        """
        super().__init__(name="web", bus=bus)
        self.host = host
        self.port = port
        self.education_service = education_service
        self.rag_retriever = rag_retriever
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self._connections: dict[str, WebSocket] = {}  # client_id -> WebSocket
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        # 学伴对话持久化（首次调用教育/认证接口时惰性初始化）
        self._db: Database | None = None
        self._edu_user_id: int | None = None
        # 专注模式监督器（首次调用 focus 接口时惰性创建）
        self._focus: FocusMonitor | None = None
        # 等待中的教育请求表：key = chat_id（请求级 uuid），value = 出站队列。
        # 教育端点把请求发布到消息总线后在此挂号，send() 凭 meta 标记
        # （edu_delta / edu_result / edu_error）把网关的回复投回对应请求，
        # 实现「HTTP 端点 → 总线 → 网关 → Agent → 总线 → HTTP 响应」闭环
        self._edu_waiters: dict[str, asyncio.Queue] = {}

    async def _ensure_edu_storage(self) -> None:
        """惰性初始化学伴对话的 SQLite 持久层与访客账号

        首次收到教育/认证接口请求时才建连（复用 nanoclaw.db，与 debug_demo
        等路径共享同一数据库）；之后直接复用，不重复初始化。
        """
        if self._db is not None:
            return

        db = Database("nanoclaw.db")
        await db.connect()
        await db.init_db()

        # 免登录访客用户：未带登录态的请求兜底共用，会话隔离由 cookie 保证
        user = await db.get_user_by_username(EDU_GUEST_USERNAME)
        if user is None:
            user_id = await db.create_user(EDU_GUEST_USERNAME, "web-guest-no-login")
            print(f"[WebChannel] 已创建教育访客账号: {EDU_GUEST_USERNAME} (user_id={user_id})")
        else:
            user_id = user["id"]

        self._db = db
        self._edu_user_id = user_id
        print(f"[WebChannel] 学伴对话持久化就绪（nanoclaw.db, guest user_id={user_id}）")

    @staticmethod
    def _edu_sender_id(username: str | None) -> str:
        """教育请求的总线发送者标识：按登录用户隔离网关会话队列。

        sender_id = "edu:<username>"，网关据此得到 session_key
        "web:edu:<username>"：同一用户的请求在网关内排队串行（保证
        SQLite 写入顺序），不同用户并发；main.py 的 Agent 工厂凭
        ":edu:" 段识别教育会话，创建 EduAgentLoop 而非通用 Agent。

        Args:
            username: 登录用户名（未登录时回退访客标识）

        Returns:
            str: 形如 "edu:alice" 的发送者标识
        """
        name = (username or "").strip() or EDU_GUEST_USERNAME
        return f"edu:{name}"

    async def _drive_edu_request(self, sender_id: str, edu_meta: dict):
        """教育请求经核心骨架的完整驱动器（学伴对话/批改/计划共用）。

        流程：注册本请求的出站等待队列 → 发布入站消息到总线 →
        Gateway 分派给 EduAgentLoop → Agent 的流式增量与最终结果
        经总线回到 WebChannel.send() → send() 投入等待队列 →
        这里逐条产出给调用方。

        Args:
            sender_id: _edu_sender_id 的返回值（网关会话队列的 key）
            edu_meta: 请求元数据，必须含 edu_kind；对话链路另含
                grade_level / subject / history / stream / cancel_event，
                批改链路含 question_text / image_data_url，
                计划链路含 grade_level / subject / weakness / weekly_hours

        Yields:
            (kind, payload) 二元组：
            - ("delta", str)    回复文本增量（仅 stream=True 的对话请求）
            - ("result", dict)  终态：EduAgentLoop 的完整结果
                （正常含 markdown/knowledge/videos/saved_file，
                失败含 error/http_status，被打断含 cancelled=True）
            - ("error", dict)   网关层异常兜底（Agent 未能给出结果），
                payload 为 {"error": 友好文案, "http_status": 502}
        """
        chat_id = uuid.uuid4().hex  # 请求级 uuid：与 WebSocket 的 client_id 互不相干
        queue: asyncio.Queue = asyncio.Queue()
        self._edu_waiters[chat_id] = queue
        try:
            await self.bus.publish_inbound(InboundMessage(
                channel=self.name,
                sender_id=sender_id,
                chat_id=chat_id,
                content=str(edu_meta.get("edu_content") or ""),
                raw=edu_meta,  # 教育元数据整体随消息过总线，Gateway 透传给 Agent
            ))
            while True:
                msg = await queue.get()
                meta = msg.meta or {}
                if meta.get("edu_delta"):
                    yield "delta", msg.content
                elif "edu_result" in meta:
                    yield "result", meta["edu_result"]
                    return
                elif "edu_error" in meta:
                    yield "error", {"error": msg.content, "http_status": 502}
                    return
                # 其他出站消息（家长推送等）与本请求无关，忽略
        finally:
            # 请求结束/断开时注销等待队列：此后到达的迟到回复
            # （如学生打断后 Agent 才收尾）由 send() 直接丢弃
            self._edu_waiters.pop(chat_id, None)

    async def start(self) -> None:
        """启动 FastAPI + WebSocket 服务。"""
        # 创建 FastAPI 应用
        self._app = FastAPI(title="NanoClaw Web")

        # 静态资源：web_ui 目录下的本地库文件（如 marked.min.js），
        # 页面脚本走 /static/ 引用，避免外部 CDN 不可达导致页面功能失效
        self._app.mount(
            "/static",
            StaticFiles(directory=Path(__file__).parent / "web_ui"),
            name="static",
        )

        # 学伴对话的作业图片落盘目录：/uploads/* 直接以文件形式对外提供，
        # 历史消息里的 <img> 凭 URL 即可回显（刷新/换设备不丢图）
        self._uploads_dir = Path("workspace/uploads")
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._app.mount(
            "/uploads",
            StaticFiles(directory=self._uploads_dir),
            name="uploads",
        )

        # 注册路由
        self._register_routes()

        # 使用 uvicorn.Server.serve() 启动（异步方式，不阻塞事件循环）：
        # uvicorn.run() 会自建事件循环并阻塞整个线程，与本渠道共存于
        # 主循环的运行模式冲突；serve() 则直接复用当前事件循环。
        # 日志精简：屏蔽启动横幅（Started server process / Application
        # startup complete / Uvicorn running 等与 [启动] 提示重复的行），
        # 但保留每个请求的 access 日志，接口调试不受影响
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.error": {"level": "WARNING"},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        }
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_config=log_config,
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    def _register_routes(self) -> None:
        """注册 FastAPI 路由。"""
        if self._app is None:
            return

        # GET / -> 返回 index.html（登录页）
        # 禁用缓存，避免页面改版后浏览器仍加载旧版导致功能异常
        @self._app.get("/")
        async def index():
            """返回登录页。"""
            index_path = Path(__file__).parent / "web_ui" / "index.html"
            no_cache = {"Cache-Control": "no-store, must-revalidate"}
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text(encoding="utf-8"), headers=no_cache)
            return HTMLResponse(content="<h1>index.html not found</h1>", headers=no_cache)

        # GET /rag_admin -> 返回 RAG 知识库管理页（edu.html「本地资料」的跳转目标）
        @self._app.get("/rag_admin")
        async def rag_admin_page():
            """返回 RAG 知识库管理页面。"""
            page_path = Path(__file__).parent / "web_ui" / "rag_admin.html"
            no_cache = {"Cache-Control": "no-store, must-revalidate"}
            if page_path.exists():
                return HTMLResponse(content=page_path.read_text(encoding="utf-8"), headers=no_cache)
            return HTMLResponse(content="<h1>rag_admin.html not found</h1>", headers=no_cache)

        # ============ 用户注册 / 登录 / 资料（SQLite users 表，bcrypt 加密） ============

        @self._app.post("/api/auth/register")
        async def auth_register(request: AuthRegister):
            """注册新用户：账号查重 → bcrypt 哈希密码 → 持久化到 users 表。

            注册只收集账号密码；昵称默认"小同学"、年级默认三年级、
            头像默认学生 emoji，进入学伴后都可在"编辑资料"弹窗修改。
            """
            username = request.username.strip()
            try:
                await self._ensure_edu_storage()
                if await self._db.get_user_by_username(username) is not None:
                    return JSONResponse(status_code=400, content={"error": "账号已存在，请换一个试试"})
                await self._db.create_user(
                    username,
                    request.password,
                    nickname="小同学",
                    grade_level=request.grade_level,
                    avatar="🧑‍🎓",
                )
            except Exception as e:
                print(f"[WebChannel] 注册接口异常: {e}")
                return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})
            print(f"[WebChannel] 新用户注册: {username}")
            row = await self._db.get_user_by_username(username)
            return _user_payload(row)

        @self._app.post("/api/auth/login")
        async def auth_login(request: AuthLogin):
            """登录校验：账号不存在或 bcrypt 校验失败一律返回 401（不泄露账号是否存在）。"""
            try:
                await self._ensure_edu_storage()
                row = await self._db.verify_password(request.username.strip(), request.password)
            except Exception as e:
                print(f"[WebChannel] 登录接口异常: {e}")
                return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})
            if row is None:
                return JSONResponse(status_code=401, content={"error": "账号或密码错误"})
            print(f"[WebChannel] 用户登录: {request.username.strip()}")
            return _user_payload(row)

        @self._app.post("/api/auth/profile")
        async def auth_profile(request: AuthProfile):
            """更新用户资料：昵称 / 头像（emoji 或导入图片）/ 年级。

            账号不可改，仅作定位用户用；头像为压缩后的图片 data URL
            或 emoji，直接存 users 表（导入时前端已压缩到 64px）。
            """
            username = request.username.strip()
            try:
                await self._ensure_edu_storage()
                row = await self._db.get_user_by_username(username)
                if row is None:
                    return JSONResponse(status_code=404, content={"error": "用户不存在"})

                # 昵称/头像未提供时保留原值（与旧版 users.json 行为一致）
                nickname = (request.nickname.strip() if request.nickname and request.nickname.strip() else None) \
                    or row["nickname"] or "小同学"
                avatar = request.avatar or row["avatar"] or "🧑‍🎓"
                await self._db.update_user_profile(username, nickname, request.grade_level, avatar)
            except Exception as e:
                print(f"[WebChannel] 资料接口异常: {e}")
                return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})
            print(f"[WebChannel] 用户更新资料: {username} (grade={request.grade_level})")
            row = await self._db.get_user_by_username(username)
            return _user_payload(row)

        # ============ 教育助手路由（注入 education_service 后才启用） ============
        if self.education_service is not None:
            # 学科分析缓存：{(user_id, subject): (数据签名, 分析文本)}。
            # 签名 = (错题数, 最新收录时间)，错题涌入导致签名变化时自动重算，
            # 实现"分析随错题更新"且同数据不重复调用 LLM
            self._wrong_analysis_cache: dict[tuple[int, str], tuple[tuple, str]] = {}

            # GET /edu -> 知契页面
            # 注意：必须禁用缓存（no-store），否则页面更新后浏览器仍加载旧版，
            # 导致新功能（如错题集）不生效且难以排查
            no_cache_headers = {"Cache-Control": "no-store, must-revalidate"}

            @self._app.get("/edu")
            async def edu_page(request: Request):
                """返回知契页面，并绑定浏览器会话 cookie。

                cookie（edu_session）是"页面关闭后仍能续接对话"的关键：
                SQLite 会话 id 存在 cookie 里，页面重开后凭它从库中
                加载历史；登录用户的会话归属由请求体里的 username 决定。
                """
                page_path = Path(__file__).parent / "web_ui" / "edu.html"
                if not page_path.exists():
                    return HTMLResponse(content="<h1>edu.html not found</h1>")

                response = HTMLResponse(
                    content=page_path.read_text(encoding="utf-8"),
                    headers=no_cache_headers,
                )
                # 仅在浏览器没有会话 cookie 时下发新的（保留老用户的历史续接）
                if not request.cookies.get(EDU_SESSION_COOKIE):
                    response.set_cookie(
                        EDU_SESSION_COOKIE,
                        str(uuid.uuid4()),
                        max_age=EDU_SESSION_MAX_AGE,
                        httponly=True,
                        samesite="lax",
                    )
                return response

            @self._app.post("/api/edu/grade")
            async def edu_grade(request: GradeRequest):
                """题目识别批改（经消息总线 → 网关 → EduAgentLoop）。

                批改是一次性任务（无流式），发布入站后等待终态出站消息：
                结果含 error 时转对应 HTTP 状态码，否则返回 markdown。
                """
                try:
                    result: dict = {"error": "生成出现问题，请稍后重试", "http_status": 502}
                    async for kind, payload in self._drive_edu_request(
                        sender_id=self._edu_sender_id(None),
                        edu_meta={
                            "edu_kind": "edu_grade",
                            "edu_content": request.question_text or "（学生提交了题目图片）",
                            "question_text": request.question_text,
                            "image_data_url": request.image_data_url,
                        },
                    ):
                        if kind in ("result", "error"):
                            result = payload
                            break
                    if result.get("error"):
                        return JSONResponse(
                            status_code=result.get("http_status", 502),
                            content={"error": result["error"]},
                        )
                    return {"markdown": result.get("markdown", "")}
                except Exception as e:
                    print(f"[WebChannel] 教育批改接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            @self._app.post("/api/edu/plan")
            async def edu_plan(raw_request: Request, request: PlanRequest):
                """生成周学习计划（经总线 → 网关 → EduAgentLoop）并落库。

                落库动机：计划的展示载体是聊天区，历史恢复以 SQLite 为
                真相源——不落库的话页面刷新/切换后计划气泡会凭空消失，
                而普通聊天消息都在。落库后行为与普通对话完全一致。
                """
                try:
                    sid, new_cookie = await _resolve_edu_session(
                        raw_request, request.username
                    )

                    result: dict = {"error": "生成出现问题，请稍后重试", "http_status": 502}
                    async for kind, payload in self._drive_edu_request(
                        sender_id=self._edu_sender_id(request.username),
                        edu_meta={
                            "edu_kind": "edu_plan",
                            "edu_content": f"生成{request.subject}周学习计划",
                            "grade_level": request.grade_level,
                            "subject": request.subject,
                            "weakness": request.weakness,
                            "weekly_hours": request.weekly_hours,
                        },
                    ):
                        if kind in ("result", "error"):
                            result = payload
                            break

                    if result.get("error"):
                        return JSONResponse(
                            status_code=result.get("http_status", 502),
                            content={"error": result["error"]},
                        )
                    markdown = result.get("markdown", "")

                    # 与前端气泡显示一致的文本一并入库（请求 + 计划正文）
                    # 年级文案与前端 GRADE_LABELS 对齐（中文数字），保证
                    # 刷新后历史恢复的显示与生成时完全一致
                    _cn = "一二三四五六七八九"
                    n = request.grade_level
                    grade_label = (
                        f"小学{_cn[n - 1]}年级" if n <= 6 else f"初中{_cn[n - 7]}年级"
                    )
                    user_text = f"📅 帮我生成{grade_label}{request.subject}的周学习计划"
                    reply = f"📅 已为{grade_label}同学生成{request.subject}周学习计划：\n\n{markdown}"
                    await self._db.add_message(sid, "user", user_text)
                    await self._db.add_message(sid, "assistant", reply)

                    response = JSONResponse(content={"markdown": markdown})
                    if new_cookie:
                        response = _edu_session_cookie(response, sid)
                    return response
                except EducationServiceError as e:
                    return JSONResponse(status_code=e.http_status, content={"error": str(e)})
                except Exception as e:
                    print(f"[WebChannel] 教育计划接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            async def _resolve_edu_session(raw_request: Request, username: str | None):
                """定位登录用户与 SQLite 会话（chat / plan 等学伴端点共用）。

                - username 有效时绑定该登录用户，否则回退免登录访客
                - cookie 缺失/过期/指向他人会话时新建会话并标记重发 cookie

                Returns:
                    (sid, new_cookie)：会话 id、是否需要重发 edu_session cookie
                """
                await self._ensure_edu_storage()

                # 定位登录用户：前端登录态带来的 username；
                # 未带或账号不存在时回退到免登录访客
                user_id = self._edu_user_id
                if username and username.strip():
                    row = await self._db.get_user_by_username(username.strip())
                    if row is not None:
                        user_id = row["id"]

                # 定位/创建 SQLite 会话（会话必须归属当前用户）
                sid = raw_request.cookies.get(EDU_SESSION_COOKIE)
                new_cookie = False
                session = await self._db.get_session(sid) if sid else None
                if session is None or session["user_id"] != user_id:
                    # cookie 缺失/过期/指向他人会话 → 建新会话并重发 cookie
                    # 注意：create_session 内部生成主键并返回，必须接住
                    sid = await self._db.create_session(user_id, title="学伴对话")
                    new_cookie = True
                return sid, new_cookie

            def _edu_session_cookie(response: JSONResponse, sid: str) -> JSONResponse:
                """给响应续接 edu_session cookie（新会话时调用）。"""
                response.set_cookie(
                    EDU_SESSION_COOKIE,
                    sid,
                    max_age=EDU_SESSION_MAX_AGE,
                    httponly=True,
                    samesite="lax",
                )
                return response

            async def _save_edu_image(data_url: str) -> str | None:
                """把 data URL 图片落盘为文件，返回可公开访问的 /uploads 路径。

                仅接受 data:image/{png,jpeg,jpg,webp,gif};base64, 前缀，
                解码失败/格式不符返回 None（调用方回退为纯文字占位）。
                """
                match = re.match(
                    r"^data:image/(png|jpeg|jpg|webp|gif);base64,(.+)$",
                    data_url.strip(),
                    re.DOTALL,
                )
                if not match:
                    return None
                ext, payload = match.group(1), match.group(2)
                try:
                    raw = base64.b64decode(payload, validate=True)
                except Exception:
                    return None
                if not raw:
                    return None
                name = f"edu_{uuid.uuid4().hex[:12]}_{secrets.token_hex(4)}.{ext}"
                (self._uploads_dir / name).write_bytes(raw)
                return f"/uploads/{name}"

            async def _prepare_edu_chat(raw_request: Request, request: ChatRequest):
                """学伴对话的共用前置：定位用户/会话、用户消息入库、组装历史。

                一次性（/api/edu/chat）与流式（/api/edu/chat/stream）两条
                路径共用，保证两条路径的会话与落库行为完全一致。

                Returns:
                    (sid, new_cookie, history)：会话 id、是否需要重发 cookie、
                    以数据库为真相源的对话历史（图片 data URL 已挂回末条）
                """
                sid, new_cookie = await _resolve_edu_session(
                    raw_request, request.username
                )
                frontend_msgs = [m.model_dump() for m in request.messages]
                latest = frontend_msgs[-1]

                # 2. 最新一条用户消息入库。带图片时图片本体落盘 uploads/
                #    目录，库里记 "[发送了图片: /uploads/xx.png]" 标记——
                #    历史加载时前端凭该标记把图片渲染回气泡（刷新不丢图）；
                #    data URL 同时挂回末条供视觉模型识别（不入库）
                user_text = latest.get("text") or ""
                if latest.get("image_data_url"):
                    img_url = await _save_edu_image(latest["image_data_url"])
                    placeholder = f"[发送了图片: {img_url}]" if img_url else "[发送了图片]"
                    user_text = (user_text + " " if user_text else "") + placeholder
                # 防重复：网络重试/重复提交时，库末尾已是同一条用户消息则跳过
                last_rows = await self._db.get_messages(sid, limit=1)
                is_dup = bool(last_rows) and last_rows[-1]["role"] == "user" \
                    and last_rows[-1]["content"] == user_text
                if not is_dup:
                    await self._db.add_message(sid, "user", user_text)

                # 3. 以数据库为真相源组装对话历史
                rows = await self._db.get_messages(sid)
                history = [{"role": r["role"], "text": r["content"]} for r in rows]
                # 当前消息若带图片，把图片 data URL 临时挂回最后一条，
                # 供视觉模型识别（不入库）
                if latest.get("image_data_url"):
                    history[-1]["image_data_url"] = latest["image_data_url"]

                return sid, new_cookie, history

            @self._app.post("/api/edu/chat")
            async def edu_chat(raw_request: Request, request: ChatRequest):
                """知契多轮对话（一次性返回，经总线 → 网关 → EduAgentLoop）。

                历史的"唯一真相源"是 SQLite：
                - 浏览器 cookie 里的 edu_session 定位会话（不存在则新建）
                - 请求带登录用户名时，会话归属该用户（同一浏览器多账号隔离）
                - 每次只把前端最新一条用户消息入库（前端重发的历史不重复入库）
                - 发给模型的历史从数据库加载（最近 16 条由 EduAgentLoop 截断）
                - 助教回复同步入库；页面关闭后重开即自动续接

                响应体透传 Agent 结果 {markdown, knowledge, videos, saved_file}：
                markdown 进对话气泡，knowledge/videos/saved_file 供右侧
                "知识点补充 / 推荐教学视频"面板与保存提示展示。
                """
                try:
                    sid, new_cookie, history = await _prepare_edu_chat(raw_request, request)
                except EducationServiceError as e:
                    return JSONResponse(status_code=e.http_status, content={"error": str(e)})
                except Exception as e:
                    print(f"[WebChannel] 学伴对话接口前置异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

                try:
                    result: dict = {"error": "生成出现问题，请稍后重试", "http_status": 502}
                    async for kind, payload in self._drive_edu_request(
                        sender_id=self._edu_sender_id(request.username),
                        edu_meta={
                            "edu_kind": "edu_chat",
                            "stream": False,
                            "grade_level": request.grade_level,
                            "subject": request.subject,
                            "history": history,
                        },
                    ):
                        if kind in ("result", "error"):
                            result = payload
                            break

                    if result.get("error"):
                        return JSONResponse(
                            status_code=result.get("http_status", 502),
                            content={"error": result["error"]},
                        )

                    # 助教回复入库（只存正文；knowledge/videos 为派生面板数据）
                    markdown = result.get("markdown", "")
                    await self._db.add_message(sid, "assistant", markdown)

                    response = JSONResponse(content={
                        "markdown": markdown,
                        "knowledge": result.get("knowledge", []),
                        "videos": result.get("videos", []),
                        "saved_file": result.get("saved_file"),
                    })
                    if new_cookie:
                        response.set_cookie(
                            EDU_SESSION_COOKIE,
                            sid,
                            max_age=EDU_SESSION_MAX_AGE,
                            httponly=True,
                            samesite="lax",
                        )
                    return response
                except Exception as e:
                    print(f"[WebChannel] 学伴对话接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            @self._app.post("/api/edu/chat/stream")
            async def edu_chat_stream(raw_request: Request, request: ChatRequest):
                """知契多轮对话流式版（SSE 打字机输出，经总线 → 网关 → EduAgentLoop）。

                会话/落库行为与 /api/edu/chat 完全一致（共用前置），
                区别仅在传输方式：发布 stream=True 的入站请求后，Agent
                的文本增量经网关 → 总线 → send() → 等待队列逐段回到
                这里，以 text/event-stream 下发，首字延迟从"整段生成完"
                降到模型出首个 token。

                事件协议（每行 data: JSON）：
                - {"type": "delta", "text": "..."}     回复文本增量
                - {"type": "done", "markdown", "knowledge", "videos",
                   "saved_file"}
                                                        生成结束（正文为
                                                        清洗后的最终 Markdown，
                                                        已同步入库；
                                                        saved_file 为笔记
                                                        保存结果，未请求
                                                        保存时为 null）
                - {"type": "error", "error": "..."}    生成中途异常

                学生打断（前端 abort 断开连接）时 FastAPI 会取消生成器，
                CancelledError 分支置位 cancel_event——该事件已随请求
                元数据传入 Agent 循环，循环在每个推理轮/每个流式 chunk
                处检查并退出；回复未完成不落库。
                """
                try:
                    sid, new_cookie, history = await _prepare_edu_chat(raw_request, request)
                except EducationServiceError as e:
                    return JSONResponse(status_code=e.http_status, content={"error": str(e)})
                except Exception as e:
                    print(f"[WebChannel] 学伴流式接口前置异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

                # 参数 / 输入校验在建流前完成：失败能以正常 HTTP 状态码
                # 返回（StreamingResponse 建立后就只能走 SSE 事件了）。
                # 校验规则与 EduAgentLoop._run_chat 内部保持一致
                subject = request.subject or "通用"
                try:
                    if subject == "通用":
                        if request.grade_level not in SUBJECTS_BY_GRADE:
                            raise EducationServiceError(
                                f"年级编号 {request.grade_level} 不支持，范围为 1-9（小学 1-6、初中 7-9）"
                            )
                    else:
                        validate_grade_subject(request.grade_level, subject)
                    if not history or history[-1].get("role") != "user":
                        raise EducationServiceError("对话历史最后一条必须是用户消息")
                    image_url = history[-1].get("image_data_url")
                    if image_url:
                        if not image_url.startswith("data:image/"):
                            raise EducationServiceError("图片格式不正确，请上传 JPG/PNG 格式的图片")
                        if len(image_url) > MAX_IMAGE_DATA_URL_LENGTH:
                            raise EducationServiceError("图片过大，请压缩后重试", http_status=413)
                        if not (self.education_service and self.education_service.vision_model):
                            raise EducationServiceError(
                                "图片对话需要视觉模型，请在 config.json 的 models.vision 中填写",
                                http_status=502,
                            )
                except EducationServiceError as e:
                    return JSONResponse(status_code=e.http_status, content={"error": str(e)})

                # 打断事件：SSE 生成器被取消时置位，Agent 循环在检查点退出
                cancel_event = asyncio.Event()

                async def event_gen():
                    try:
                        async for kind, payload in self._drive_edu_request(
                            sender_id=self._edu_sender_id(request.username),
                            edu_meta={
                                "edu_kind": "edu_chat",
                                "stream": True,
                                "cancel_event": cancel_event,
                                "grade_level": request.grade_level,
                                "subject": subject,
                                "history": history,
                            },
                        ):
                            if kind == "delta":
                                yield "data: " + json.dumps(
                                    {"type": "delta", "text": payload}, ensure_ascii=False
                                ) + "\n\n"
                                continue

                            # result / error 都是终态
                            if payload.get("error"):
                                yield "data: " + json.dumps(
                                    {"type": "error", "error": payload["error"]},
                                    ensure_ascii=False,
                                ) + "\n\n"
                                return
                            if payload.get("cancelled"):
                                # 学生打断：回复未完成，不落库
                                return

                            markdown = payload.get("markdown", "")
                            await self._db.add_message(sid, "assistant", markdown)
                            yield "data: " + json.dumps(
                                {
                                    "type": "done",
                                    "markdown": markdown,
                                    "knowledge": payload.get("knowledge", []),
                                    "videos": payload.get("videos", []),
                                    "saved_file": payload.get("saved_file"),
                                },
                                ensure_ascii=False,
                            ) + "\n\n"
                    except asyncio.CancelledError:
                        # 学生打断 / 连接断开：通知 Agent 停止生成；
                        # 迟到的结果因等待队列已注销而被 send() 丢弃
                        cancel_event.set()
                        raise
                    except Exception as e:
                        print(f"[WebChannel] 流式对话生成异常: {e}")
                        yield "data: " + json.dumps(
                            {"type": "error", "error": "生成出现问题，请稍后重试"},
                            ensure_ascii=False,
                        ) + "\n\n"

                response = StreamingResponse(
                    event_gen(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
                if new_cookie:
                    response.set_cookie(
                        EDU_SESSION_COOKIE,
                        sid,
                        max_age=EDU_SESSION_MAX_AGE,
                        httponly=True,
                        samesite="lax",
                    )
                return response

            @self._app.post("/api/edu/transcribe")
            async def edu_transcribe(file: UploadFile = File(...)):
                """语音转文字（点一下说话）：前端 MediaRecorder 录制的音频
                原样透传给硅基流动 SenseVoice 转录，返回识别文本。

                成功响应 {"text": "识别出的文字"}；
                识别空结果时返回 {"text": ""}，由前端提示重说。
                """
                if self.education_service is None:
                    return JSONResponse(status_code=503, content={"error": "语音服务未启用"})
                audio_data = await file.read()
                if not audio_data:
                    return JSONResponse(status_code=400, content={"error": "未收到音频数据"})

                provider = self.education_service.provider
                api_url = f"{provider.base_url.rstrip('/')}/audio/transcriptions"
                try:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        form = aiohttp.FormData()
                        # 浏览器产物（webm/ogg/wav）按原扩展名与 MIME 转发，
                        # 服务端靠扩展名识别封装格式
                        form.add_field(
                            "file",
                            audio_data,
                            filename=file.filename or "recording.webm",
                            content_type=file.content_type or "audio/webm",
                        )
                        form.add_field("model", self.stt_model)
                        async with session.post(
                            api_url,
                            headers={"Authorization": f"Bearer {provider.api_key}"},
                            data=form,
                        ) as resp:
                            body = await resp.json(content_type=None)
                            if resp.status != 200:
                                print(f"[WebChannel] STT 接口返回 {resp.status}: {body}")
                                return JSONResponse(status_code=502, content={"error": "语音识别失败，请重试"})
                            return {"text": (body.get("text") or "").strip()}
                except Exception as e:
                    print(f"[WebChannel] 语音转录接口异常: {e}")
                    return JSONResponse(status_code=502, content={"error": "语音识别失败，请检查网络后重试"})

            @self._app.post("/api/edu/tts")
            async def edu_tts(request: TTSRequest):
                """文字转语音（🔊 朗读）：把学伴回复清洗为口语文本后交给
                CosyVoice2 合成 mp3，二进制音频直接回给前端 <audio> 播放。
                """
                if self.education_service is None:
                    return JSONResponse(status_code=503, content={"error": "语音服务未启用"})
                plain = _md_to_plain(request.text)
                if not plain:
                    return JSONResponse(status_code=400, content={"error": "没有可朗读的内容"})
                if len(plain) > TTS_MAX_CHARS:
                    plain = plain[:TTS_MAX_CHARS]

                provider = self.education_service.provider
                api_url = f"{provider.base_url.rstrip('/')}/audio/speech"
                try:
                    timeout = aiohttp.ClientTimeout(total=120)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        payload = {
                            "model": self.tts_model,
                            "input": plain,
                            # 预置音色的完整格式是 "<model>:<voice>"
                            "voice": f"{self.tts_model}:{self.tts_voice}",
                            "response_format": "mp3",
                            "sample_rate": 32000,
                        }
                        async with session.post(
                            api_url,
                            headers={
                                "Authorization": f"Bearer {provider.api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        ) as resp:
                            if resp.status != 200:
                                err_body = await resp.text()
                                print(f"[WebChannel] TTS 接口返回 {resp.status}: {err_body[:300]}")
                                return JSONResponse(status_code=502, content={"error": "语音合成失败，请重试"})
                            audio_bytes = await resp.read()
                    return Response(content=audio_bytes, media_type="audio/mpeg")
                except Exception as e:
                    print(f"[WebChannel] 朗读接口异常: {e}")
                    return JSONResponse(status_code=502, content={"error": "语音合成失败，请检查网络后重试"})

            @self._app.get("/api/edu/history")
            async def edu_history(raw_request: Request):
                """按浏览器会话 cookie 返回学伴聊天记录（页面重开时回显）。

                可选 query 参数 username：登录用户请求自己的历史；
                cookie 会话必须归属该用户，否则视为无历史（多账号隔离）。
                """
                try:
                    await self._ensure_edu_storage()

                    # 与 chat 的用户定位规则保持一致：username 优先，回退访客
                    user_id = self._edu_user_id
                    username = raw_request.query_params.get("username")
                    if username and username.strip():
                        row = await self._db.get_user_by_username(username.strip())
                        if row is not None:
                            user_id = row["id"]

                    sid = raw_request.cookies.get(EDU_SESSION_COOKIE)
                    messages = []
                    if sid:
                        session = await self._db.get_session(sid)
                        if session is not None and session["user_id"] == user_id:
                            rows = await self._db.get_messages(sid)
                            messages = [
                                # ts: Unix 秒时间戳，前端用于渲染 >10 分钟间隔的时间分隔线
                                {"role": r["role"], "content": r["content"], "ts": r["timestamp"]}
                                for r in rows
                            ]
                    return JSONResponse(content={"messages": messages})
                except Exception as e:
                    print(f"[WebChannel] 学伴历史接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            # ============ 错题集：云端存储（wrong_questions 表） ============

            async def _wrongbook_user_id(username: str | None) -> int:
                """定位错题归属用户：username 优先，回退访客（与学伴对话规则一致）。"""
                if username and username.strip():
                    row = await self._db.get_user_by_username(username.strip())
                    if row is not None:
                        return row["id"]
                return self._edu_user_id

            def _wrongbook_payload(r) -> dict:
                """wrong_questions 行 → 前端条目（字段与旧 localStorage 版结构对齐）。"""
                return {
                    "id": r["id"],
                    "time": r["create_time"],  # epoch 秒，前端格式化展示
                    "question": r["question"],
                    "verdict": r["verdict"],
                    "markdown": r["markdown"],
                    "subject": r["subject"] if "subject" in r.keys() else "其他",
                }

            @self._app.get("/api/edu/wrongbook")
            async def wrongbook_list(raw_request: Request):
                """返回当前用户的错题列表（最新在前，最多 100 条）。

                可选 query 参数 subject：按学科过滤（如 "数学"），缺省/"全部"不过滤。
                """
                try:
                    await self._ensure_edu_storage()
                    user_id = await _wrongbook_user_id(raw_request.query_params.get("username"))
                    subject = raw_request.query_params.get("subject") or None
                    rows = await self._db.list_wrong_questions(user_id, subject=subject)
                    return JSONResponse(content={"items": [_wrongbook_payload(r) for r in rows]})
                except Exception as e:
                    print(f"[WebChannel] 错题集列表接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            @self._app.post("/api/edu/wrongbook")
            async def wrongbook_add(request: WrongBookAdd):
                """批量收录错题（前端从学伴批改结果中提取后整批提交）。

                学科规则：条目自带合法 subject 时直接使用；否则整批一次
                LLM 调用自动分类（失败降级关键词兜底），分类不阻塞收录。
                """
                try:
                    await self._ensure_edu_storage()
                    user_id = await _wrongbook_user_id(request.username)
                    items = [it.model_dump() for it in request.items]

                    # 补齐学科：仅对未带 subject 的条目批量自动分类
                    valid_subjects = set(WRONGBOOK_SUBJECTS)
                    need_classify = [it for it in items if it.get("subject") not in valid_subjects]
                    if need_classify:
                        subjects = await self.education_service.classify_subjects(
                            [it["question"] for it in need_classify]
                        )
                        for it, subj in zip(need_classify, subjects):
                            it["subject"] = subj

                    await self._db.add_wrong_questions(user_id, items)
                    return JSONResponse(content={"ok": True, "added": len(items)})
                except Exception as e:
                    print(f"[WebChannel] 错题集收录接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            @self._app.get("/api/edu/wrongbook/analysis")
            async def wrongbook_analysis(raw_request: Request):
                """生成指定学科的错题薄弱点分析（错题数据变化时自动重新生成）。

                经总线 → 网关 → EduAgentLoop._run_wrongbook（Agent 可自主
                调 rag_search 检索教材佐证复习建议），与对话/批改/计划
                同一核心骨架；classify_subjects 仍为收录时的轻量预处理。

                query 参数：username（定位用户）、subject（学科名或"全部"）。
                """
                try:
                    await self._ensure_edu_storage()
                    user_id = await _wrongbook_user_id(raw_request.query_params.get("username"))
                    subject = raw_request.query_params.get("subject") or "全部"
                    rows = await self._db.list_wrong_questions(user_id, subject=subject)
                    entries = [
                        {
                            "question": r["question"],
                            "verdict": r["verdict"],
                            "markdown": r["markdown"],
                        }
                        for r in rows
                    ]

                    # 数据签名（条数 + 最新收录时间）未变化时直接用缓存
                    signature = (len(entries), rows[0]["create_time"] if rows else 0.0)
                    cached = self._wrong_analysis_cache.get((user_id, subject))
                    if cached and cached[0] == signature:
                        return JSONResponse(content={"ok": True, "subject": subject, "analysis": cached[1]})

                    result: dict = {"error": "生成出现问题，请稍后重试", "http_status": 502}
                    async for kind, payload in self._drive_edu_request(
                        sender_id=self._edu_sender_id(raw_request.query_params.get("username")),
                        edu_meta={
                            "edu_kind": "edu_wrongbook",
                            "edu_content": f"生成{subject}错题薄弱点分析",
                            "subject": subject,
                            "entries": entries,
                        },
                    ):
                        if kind in ("result", "error"):
                            result = payload
                            break

                    if result.get("error") or result.get("cancelled"):
                        return JSONResponse(
                            status_code=int(result.get("http_status", 502)),
                            content={"error": result.get("error", "生成被打断，请重试")},
                        )

                    analysis = result.get("markdown", "")
                    self._wrong_analysis_cache[(user_id, subject)] = (signature, analysis)
                    return JSONResponse(content={"ok": True, "subject": subject, "analysis": analysis})
                except EducationServiceError as e:
                    return JSONResponse(status_code=e.http_status, content={"error": str(e)})
                except Exception as e:
                    print(f"[WebChannel] 错题分析接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            @self._app.delete("/api/edu/wrongbook/{question_id}")
            async def wrongbook_delete(question_id: int, raw_request: Request):
                """删除单条错题（限定归属用户，防止越权）。"""
                try:
                    await self._ensure_edu_storage()
                    user_id = await _wrongbook_user_id(raw_request.query_params.get("username"))
                    deleted = await self._db.delete_wrong_question(user_id, question_id)
                    return JSONResponse(content={"ok": deleted})
                except Exception as e:
                    print(f"[WebChannel] 错题集删除接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            @self._app.delete("/api/edu/wrongbook")
            async def wrongbook_clear(raw_request: Request):
                """清空当前用户全部错题。"""
                try:
                    await self._ensure_edu_storage()
                    user_id = await _wrongbook_user_id(raw_request.query_params.get("username"))
                    cleared = await self._db.clear_wrong_questions(user_id)
                    return JSONResponse(content={"ok": True, "cleared": cleared})
                except Exception as e:
                    print(f"[WebChannel] 错题集清空接口异常: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})

            # ============ 专注模式：YOLO 坐姿/注意力监督 ============

            @self._app.post("/api/edu/focus/start")
            async def focus_start(raw_request: Request):
                """开启专注监督循环（YOLO 每隔 interval 秒检测一轮）。

                请求体可选 {"username": "登录用户名"}，仅作记录用途。
                单摄像头独占：已在运行时返回 409 冲突。
                """
                try:
                    body = await raw_request.json()
                except Exception:
                    body = {}
                username = str(body.get("username") or "").strip()
                if self._focus is None:
                    self._focus = FocusMonitor()
                result = await self._focus.start(username)
                if not result.get("ok"):
                    return JSONResponse(status_code=409, content=result)
                print(f"[WebChannel] 专注监督开启: user={username or 'guest'}")
                return result

            @self._app.post("/api/edu/focus/stop")
            async def focus_stop():
                """停止专注监督循环，返回本次监督汇总。"""
                if self._focus is None:
                    return {"ok": True, "rounds": 0, "alert_count": 0}
                result = await self._focus.stop()
                print(f"[WebChannel] 专注监督结束: rounds={result['rounds']} alerts={result['alert_count']}")
                return result

            @self._app.get("/api/edu/focus/status")
            async def focus_status():
                """专注监督当前状态（前端开启期间每几秒轮询一次）。"""
                if self._focus is None:
                    return {
                        "running": False, "rounds": 0, "alert_count": 0,
                        "latest": None, "events": [], "last_error": None,
                    }
                return self._focus.status()

        # ============ RAG 知识库管理接口（注入 rag_retriever 后启用） ============
        # 阻塞操作（Embedding API 调用 / FAISS 索引读写）一律丢进线程池，
        # 避免拖慢事件循环上的其他会话
        if self.rag_retriever is not None:

            @self._app.get("/api/rag/config")
            async def rag_config():
                """返回检索默认参数（阈值 / top_k），供管理页展示。"""
                return {
                    "threshold": self.rag_retriever.default_threshold,
                    "top_k": getattr(self.rag_retriever, "default_top_k", 3),
                }

            @self._app.get("/api/rag/documents")
            async def rag_list_documents():
                """列出所有已导入文档来源及块数统计。"""
                sources = await asyncio.to_thread(self.rag_retriever.get_all_sources)
                total = await asyncio.to_thread(self.rag_retriever.count)
                return {"documents": sources, "total_chunks": total}

            @self._app.get("/api/rag/document")
            async def rag_document_preview(raw_request: Request):
                """预览指定来源文档的整篇内容（管理页点击文档时调用）。

                query 参数 source：文档来源名（与导入时 metadata.source 精确匹配）。
                各分块按序拼接为整篇返回，块间重叠自动去除。
                """
                source = (raw_request.query_params.get("source") or "").strip()
                if not source:
                    return JSONResponse(status_code=400, content={"error": "缺少 source 参数"})
                try:
                    preview = await asyncio.to_thread(
                        self.rag_retriever.get_source_preview, source
                    )
                except Exception as e:
                    print(f"[RAG] 文档预览失败 {source}: {e}")
                    return JSONResponse(status_code=500, content={"error": "服务器内部错误，请稍后重试"})
                if not preview["chunk_count"]:
                    return JSONResponse(status_code=404, content={"error": "未找到该文档的内容"})
                return {
                    "source": source,
                    "chunk_count": preview["chunk_count"],
                    "full_text": preview["full_text"],
                }

            @self._app.post("/api/rag/add")
            async def rag_add_document(request: Request):
                """添加教材文本到知识库（自动分块 + 向量化 + 持久化）。"""
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
                content = body.get("content", "")
                source = body.get("source", "manual")
                metadata = body.get("metadata", {}) or {}
                metadata["source"] = source
                if not content.strip():
                    return JSONResponse(status_code=400, content={"error": "内容不能为空"})
                try:
                    added = await asyncio.to_thread(
                        self.rag_retriever.add_documents,
                        [{"content": content, "metadata": metadata}],
                    )
                except Exception as e:
                    print(f"[RAG] 添加失败: {e}")
                    return JSONResponse(status_code=502, content={"error": f"添加失败: {e}"})
                return {
                    "message": f"添加成功，新增 {added} 个块",
                    "added": added,
                    "total_chunks": self.rag_retriever.count(),
                }

            @self._app.post("/api/rag/upload")
            async def rag_upload(files: list[UploadFile] = File(...), subject: str = Form("")):
                """批量上传教材文件（multipart），服务端解析多格式后入库。

                支持 PDF / Word(.doc .docx) / PPT(.ppt .pptx) / Excel(.xls .xlsx) /
                TXT / MD / CSV / HTML / 图片 OCR（JPG PNG BMP WEBP）。
                单文件失败不影响其他文件，逐文件返回结果。
                """
                results = []
                for f in files:
                    fname = f.filename or "unnamed"
                    try:
                        data = await f.read()
                        text = await asyncio.to_thread(parse_document, fname, data)
                        metadata = {"source": fname, "subject": subject} if subject else {"source": fname}
                        added = await asyncio.to_thread(
                            self.rag_retriever.add_documents,
                            [{"content": text, "metadata": metadata}],
                        )
                        results.append({
                            "filename": fname, "ok": True, "added": added,
                            "chars": len(text),
                        })
                        print(f"[RAG] 上传导入成功: {fname}（{len(text)} 字 → {added} 块）")
                    except DocParseError as e:
                        results.append({"filename": fname, "ok": False, "error": str(e)})
                        print(f"[RAG] 上传解析失败: {fname}: {e}")
                    except Exception as e:
                        results.append({"filename": fname, "ok": False, "error": f"导入失败: {e}"})
                        print(f"[RAG] 上传导入失败: {fname}: {e}")
                return {"results": results, "total_chunks": self.rag_retriever.count()}

            @self._app.post("/api/rag/delete")
            async def rag_delete_document(request: Request):
                """删除指定来源的全部块（向量重组，不重复调用 Embedding）。"""
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
                source = body.get("source", "")
                if not source:
                    return JSONResponse(status_code=400, content={"error": "source 不能为空"})
                removed = await asyncio.to_thread(self.rag_retriever.delete_source, source)
                if removed == 0:
                    return JSONResponse(status_code=404, content={"error": f"未找到来源: {source}"})
                return {
                    "message": f"已删除「{source}」，共移除 {removed} 个块",
                    "removed": removed,
                    "total_chunks": self.rag_retriever.count(),
                }

            @self._app.delete("/api/rag/clear")
            async def rag_clear():
                """清空整个知识库（不可恢复）。"""
                await asyncio.to_thread(self.rag_retriever.clear)
                return {"message": "已清空所有数据", "total_chunks": 0}

            @self._app.post("/api/rag/retrieve")
            async def rag_retrieve(request: Request):
                """本地向量检索测试：返回最相关的教材片段（含相似度）。"""
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
                query = body.get("query", "")
                top_k = body.get("top_k", 3)
                threshold = body.get("threshold")
                if not query:
                    return JSONResponse(status_code=400, content={"error": "查询不能为空"})
                try:
                    results = await asyncio.to_thread(
                        self.rag_retriever.retrieve, query, top_k=top_k, threshold=threshold
                    )
                except Exception as e:
                    print(f"[RAG] 检索失败: {e}")
                    return JSONResponse(status_code=502, content={"error": f"检索失败: {e}"})
                return {"results": results}

            @self._app.post("/api/rag/ask")
            async def rag_ask(request: Request):
                """RAG 问答：先检索教材片段，再让主模型依据片段作答。"""
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
                query = body.get("query", "")
                top_k = body.get("top_k", 3)
                threshold = body.get("threshold")
                if not query:
                    return JSONResponse(status_code=400, content={"error": "查询不能为空"})

                try:
                    results = await asyncio.to_thread(
                        self.rag_retriever.retrieve, query, top_k=top_k, threshold=threshold
                    )
                except Exception as e:
                    return JSONResponse(status_code=502, content={"error": f"检索失败: {e}"})
                if not results:
                    return {"answer": "知识库中未找到相关内容，无法回答。", "sources": []}

                context = "\n\n".join([
                    f"【来源：{r['source']}】\n{r['content']}"
                    for r in results
                ])

                if self.education_service:
                    try:
                        response = await self.education_service.provider.chat(
                            messages=[
                                {"role": "system", "content": "你是一位老师。请根据提供的教材片段准确回答问题。如果片段信息不足，请诚实说明。"},
                                {"role": "user", "content": f"参考信息：\n{context}\n\n问题：{query}\n请回答。"},
                            ],
                            extra_params={"temperature": 0.3, "max_tokens": 2048},
                        )
                        answer = response.content or "无法生成答案"
                    except Exception as e:
                        answer = f"生成答案失败: {e}"
                else:
                    answer = "问答服务未配置"
                sources = [{"source": r["source"], "score": r["score"]} for r in results]
                return {"answer": answer, "sources": sources}

        # WebSocket /ws
        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 连接处理。"""
            # 接收连接
            await websocket.accept()

            # 生成 client_id：每个标签页独立 uuid，作为会话唯一标识；
            # 后续回复凭它从连接表找回连接，多标签页/多用户互不串扰
            client_id = str(uuid.uuid4())
            self._connections[client_id] = websocket

            print(f"[WebChannel] 新连接: client_id={client_id}")

            try:
                # 循环接收消息
                while True:
                    data = await websocket.receive_text()

                    # 跳过空消息
                    if not data.strip():
                        continue

                    print(f"[WebChannel] 收到消息: client_id={client_id} content={data[:50]}...")

                    # 构造入站消息（chat_id = client_id，用于找回连接发送回复）
                    message = InboundMessage(
                        channel="web",
                        sender_id=client_id,
                        chat_id=client_id,
                        content=data,
                        raw={
                            "client_id": client_id,
                        },
                    )

                    # 发布到总线
                    await self.bus.publish_inbound(message)

            except Exception as e:
                # 连接断开
                print(f"[WebChannel] 连接断开: client_id={client_id} reason={e}")

            finally:
                # 清理连接
                self._connections.pop(client_id, None)

    async def send(self, message: OutboundMessage) -> None:
        """发送出站消息：教育响应投回等待中的 HTTP 请求，其余走 WebSocket。

        教育链路的回复（流式增量 / 最终结果 / 错误）凭 meta 标记识别，
        投入 _edu_waiters 中该请求的队列，由 _drive_edu_request 消费后
        变成 SSE 事件或 JSON 响应——教育回复与普通聊天一样走总线，
        但绝不混入 WebSocket 通道。

        Args:
            message: 出站消息实例
        """
        meta = message.meta or {}
        if meta.get("edu_delta") or "edu_result" in meta or "edu_error" in meta:
            waiter = self._edu_waiters.get(message.chat_id)
            if waiter is None:
                # 等待方已消失（学生打断 SSE / 请求超时收尾），迟到回复丢弃
                print(f"[WebChannel] 教育响应无等待方，已丢弃: chat_id={message.chat_id}")
                return
            await waiter.put(message)
            return

        # 通过 chat_id 找回 WebSocket 连接
        ws = self._connections.get(message.chat_id)

        if ws is None:
            # 连接不存在（用户已断开）
            print(f"[WebChannel] 连接不存在: chat_id={message.chat_id}")
            return

        try:
            # 发送文本消息
            await ws.send_text(message.content)
            print(f"[WebChannel] 消息已发送: chat_id={message.chat_id}")

        except Exception as e:
            # 发送失败说明连接已失效（用户关闭页面/网络中断等），
            # 立即从连接表移除，避免后续回复继续往死连接上写
            print(f"[WebChannel] 发送失败: chat_id={message.chat_id} reason={e}")
            self._connections.pop(message.chat_id, None)

    async def stop(self) -> None:
        """停止 Web 服务。

        先通知 uvicorn 优雅退出，再逐一关闭活跃 WebSocket 连接，
        避免服务停止后仍有连接悬挂。
        """
        if self._server is not None:
            # 通知 server 停止：should_exit 是 uvicorn 的优雅停机标志，
            # 置 True 后 serve() 循环会在下一轮检查时自行退出
            self._server.should_exit = True
            self._server = None

        # 关闭所有 WebSocket 连接（先拷贝快照再遍历，
        # 避免遍历过程中字典被修改导致 RuntimeError）
        for client_id, ws in list(self._connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()

        # 注销所有等待中的教育请求（服务停止后它们的回复不会再产生）
        self._edu_waiters.clear()

        # 停止专注监督循环（若开启过），释放摄像头占用
        if self._focus is not None:
            try:
                await self._focus.stop()
            except Exception:
                pass

        # 关闭学伴对话的 SQLite 连接（若教育功能已启用过）
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                pass
            self._db = None

        print("[WebChannel] 服务已停止")
