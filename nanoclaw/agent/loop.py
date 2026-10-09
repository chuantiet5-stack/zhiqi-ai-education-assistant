"""Agent 主循环模块

本模块实现 NanoClaw Agent 的核心推理循环（Agentic Loop）。

核心职责：
- 将用户消息组装进上下文，循环调用 LLM 获取响应
- 当 LLM 请求工具调用时，执行工具并把结果以 tool 消息的形式
  追加回对话，再次请求 LLM，直到 LLM 给出最终文本回复
- 通过签名统计 + 滑动窗口检测重复调用，防止工具被无限循环调用（防爆机制）

在 Agent 架构中的角色：
    用户输入 → ContextBuilder 组装上下文 → AgentLoop 循环
        ├─ LLM 返回工具调用 → 执行工具 → 结果回填 → 进入下一轮推理
        └─ LLM 返回纯文本   → 保存历史 → 返回最终响应

整合对接层（文件末尾）：
    面向 Web 渠道整合预留的稳定异步函数（run_chat_round 等），
    对接 SQLite 持久层与 MemoryManager，产品定位为中小学生
    引导式助教；不依赖 FastAPI / 网页接口 / 向量服务。
"""

import json

from providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agent.tools.registry import ToolRegistry
from agent.context import ContextBuilder
from session.manager import SessionManager


class GenerationCancelled(Exception):
    """生成被请求方主动取消（如学生打断 SSE 连接）。

    注意：刻意继承 Exception 而非模拟 asyncio.CancelledError——
    本异常在网关的会话 worker 协程内抛出，若用 CancelledError 会被
    worker 的"任务被取消则退出"分支误杀，导致该会话队列永久停摆。
    """

class AgentLoop:
    """Agent 执行循环

    管理 LLM 与工具之间的交互循环，包含防爆检测机制。

    设计意图：
    - 单次 run() 内部可能经历多轮"推理 → 工具执行 → 结果回填"，
      直到 LLM 不再请求工具调用、给出最终回答为止
    - max_iterations 为整个循环设置安全上限，防止 LLM 陷入
      死循环后无限消耗 Token 与时间
    - 所有消息（用户 / assistant / tool）在追加回内存列表的同时
      经 SessionManager 持久化，保证会话可在进程重启后恢复
    """

    # 防爆阈值常量
    FUSE_THRESHOLD = 20  # 熔断阈值
    WARNING_THRESHOLD = 10  # 警告阈值
    WINDOW_SIZE = 30  # 滑动窗口大小

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        context: ContextBuilder,
        session_manager: SessionManager,
        model: str | None = None,
        max_iterations: int = 32,
        session_key: str = "cli:direct"
    ):
        """初始化 AgentLoop

        Args:
            provider: LLM 提供方实例
            tools: 工具注册表
            context: 上下文构建器
            session_manager: 会话管理器
            model: 模型名称（可选）
            max_iterations: 最大迭代次数
            session_key: 会话标识符，默认 "cli:direct"
        """
        self.provider = provider
        self.tools = tools
        self.context = context
        self.session_manager = session_manager
        self.session_key = session_key
        self.model = model
        self.max_iterations = max_iterations
        # 压缩器实例由外部按需注入（构造 AgentLoop 时压缩器可能尚未
        # 创建），故先置 None，运行时判断存在才触发压缩
        self.consolidator = None  # 在整合压缩器之前先设为 None

        # 内部状态
        # 滑动窗口只保留最近 WINDOW_SIZE 次调用签名：
        # 既能捕捉"短期密集重复"的循环模式，又避免历史无限增长
        self._tool_call_history: list[str] = []  # 滑动窗口
        # 本轮 run 的工具调用轨迹（name/arguments/result）：
        # 教育链路据此把工具产出的结构化数据（视频列表/百科链接/笔记路径）
        # 回传给 Web 层组装前端面板；CLI 路径不消费，仅记录
        self.tool_trace: list[dict] = []
        # 本轮 run 的中间轮正文：请求工具调用前模型已输出的文本片段。
        # 多轮工具循环里实质内容常出现在工具调用之前的轮次（如先讲解
        # 再搜视频），若只取最后一轮文本会丢失讲解——教育链路把
        # 中间轮与最终轮拼接为完整回复
        self.round_texts: list[str] = []
        # 从会话管理器恢复历史
        self._session_history: list[dict] = session_manager.get_history(session_key)

    async def run(
        self,
        user_message: str,
        meta: dict | None = None,
        on_delta=None,
    ) -> str:
        """执行 Agent 主循环

        Args:
            user_message: 用户输入消息
            meta: 附加元数据（可选）。CLI/飞书渠道传 None 或渠道原始数据；
                教育链路（EduAgentLoop 子类）通过它传入 cancel_event、
                模型覆盖、allowed_tools 等请求级参数
            on_delta: 文本增量回调（可选）。传入时每轮流式执行，
                文本增量经 await on_delta(piece) 实时产出（打字机效果）；
                工具调用轮的增量由 Provider 层缓冲归并，不泄露半截内容

        Returns:
            str: Agent 的最终响应
        """
        # 1. 构建初始 messages
        # 使用 __init__ 时从 SessionManager 恢复的内存历史，
        # 避免每次对话都重复读盘
        messages = self.context.build_messages(
            history=self._session_history,
            current_message=user_message
        )

        # 保存用户消息（session_manager 为 None 时不落盘，
        # 教育链路的持久化由 Web 层的 SQLite 负责）
        user_msg = {"role": "user", "content": user_message}
        if self.session_manager:
            self.session_manager.save_message(self.session_key, user_msg)

        # 2. 执行推理-工具循环
        return await self._execute_loop(messages, meta=meta, on_delta=on_delta)

    async def _execute_loop(
        self,
        messages: list[dict],
        meta: dict | None = None,
        on_delta=None,
    ) -> str:
        """推理-工具循环体（CLI 与教育链路共用）

        每轮：调用 LLM（tools 定义来自 ToolRegistry）→ 有工具调用则
        执行并回填 tool 消息进入下一轮，无则返回最终文本。
        防爆熔断 / 压缩 / 持久化钩子全部在此层统一生效。

        Args:
            messages: 已组装好的消息列表（就地追加中间消息）
            meta: 请求级元数据（cancel_event / model / allowed_tools / edu_kind）
            on_delta: 流式增量回调，为 None 时走非流式 chat()

        Returns:
            str: 最终回复文本；熔断时返回熔断提示
        """
        meta = meta or {}
        self.tool_trace = []  # 每轮请求重置工具轨迹
        self.round_texts = []  # 每轮请求重置中间轮正文

        # 请求级参数：取消事件 / 模型覆盖（视觉轮）/ 工具白名单 / 生成参数
        cancel_event = meta.get("cancel_event")
        model = meta.get("model") or self.model
        extra_params = meta.get("extra_params")
        tools_defs = self.tools.get_definitions() if self.tools else []
        allowed = meta.get("allowed_tools")
        if allowed is not None:
            allow_set = set(allowed)
            tools_defs = [
                d for d in tools_defs if d["function"]["name"] in allow_set
            ]

        def _cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        # max_iterations 是循环的安全上限：LLM 每轮都可能继续请求
        # 工具调用，不设上限的话一旦模型陷入死循环会无限消耗
        # Token 和时间；达到上限后由 for-else 兜底返回提示
        for iteration in range(self.max_iterations):
            # 学生打断检查：轮与轮之间也要响应取消
            if _cancelled():
                raise GenerationCancelled()

            # 压缩必须发生在调用 LLM 之前：若带着超预算的 messages
            # 发起请求，可能直接被模型端的上下文长度限制拒绝
            if hasattr(self, 'consolidator') and self.consolidator:
                pre_compress_len = len(messages)
                messages = await self.consolidator.maybe_consolidate(messages)
                if len(messages) < pre_compress_len:
                    # 发生了压缩：快照从"[system]+旧历史+新消息"变成了
                    # "[system, 摘要]+保留尾部"，_save_to_history 基于
                    # 长度差的增量推算随之失效（会漏记甚至错记本轮消息，
                    # 内存历史与持久层永久漂移）——改为以压缩后快照整体
                    # 重建内存历史（剔除摘要消息），保证两边始终一致
                    self._sync_history_after_consolidation(messages)

            # a. 调用 LLM（流式 / 非流式两条等价路径）
            print(f"  💭 思考中... (第 {iteration + 1} 轮)", end="", flush=True)
            if on_delta is not None:
                content, tool_calls, finish = await self._stream_round(
                    messages, tools_defs, model, on_delta, cancel_event, extra_params
                )
                if finish == "error":
                    return "抱歉，发生了错误，请稍后重试。"
            else:
                response = await self.provider.chat(
                    messages=messages,
                    tools=tools_defs or None,
                    model=model,
                    extra_params=extra_params,
                )
                print(" ✓")  # 思考完成

                # 打印 reasoning_content（思考过程）
                if response.tool_calls and response.tool_calls[0].reasoning_content:
                    reasoning = response.tool_calls[0].reasoning_content
                    # 截断显示，避免刷屏
                    preview = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                    print(f"  🧠 思考过程:\n{preview}")

                # 错误响应不携带有效的 content / tool_calls，必须先于
                # 工具调用分支处理，避免把异常响应当正常结果回填对话
                if response.finish_reason == "error":
                    return f"抱歉，发生了错误，请稍后重试。"

                content = response.content
                tool_calls = response.tool_calls
                finish = response.finish_reason

            # c. 处理工具调用
            if tool_calls:
                # 记录中间轮正文（请求工具前已输出的内容）：多轮工具循环
                # 里实质讲解常在工具调用之前，教育链路依赖它拼接完整回复，
                # 避免讲解只进流式增量、不进持久化正文
                if content:
                    self.round_texts.append(content)

                # 构造 assistant 消息（含 tool_calls）
                assistant_msg = self._build_assistant_message(LLMResponse(
                    content=content, tool_calls=tool_calls, finish_reason=finish,
                ))
                # 必须先把声明 tool_calls 的 assistant 消息追加回对话，
                # 再追加对应的 tool 结果消息——OpenAI 协议要求 tool 消息
                # 紧跟在含对应 tool_call_id 的 assistant 消息之后，
                # 顺序颠倒会被 API 以消息顺序非法为由拒绝
                messages.append(assistant_msg)

                # 保存 assistant 消息（含 tool_calls）
                if self.session_manager:
                    self.session_manager.save_message(self.session_key, assistant_msg)

                # 遍历每个 tool_call
                for tc in tool_calls:
                    # 获取参数 JSON 字符串
                    args_json = json.dumps(tc.arguments, ensure_ascii=False)

                    # ★ 打印工具调用过程，让用户看到 Agent 在干什么
                    print(f"\n  🛠️  调用工具: {tc.name}({args_json})")

                    # 防爆检测
                    check_result = self._check_tool_loop(tc.name, args_json)

                    if check_result and "熔断" in check_result:
                        # 熔断 - 直接返回
                        print(f"  🚨 {check_result}")
                        return check_result
                    elif check_result and "警告" in check_result:
                        # 警告 - append SYSTEM_ERROR 消息，skip 执行
                        # 用一条伪造的 tool 消息代替真实执行结果：既满足
                        # "tool_call 必须有配对 tool 消息"的协议要求，
                        # 又把警告内容反馈给 LLM，引导其改用其他方案
                        print(f"  ⚠️  {check_result}")
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"系统警告：{check_result}"
                        }
                        messages.append(tool_msg)
                        if self.session_manager:
                            self.session_manager.save_message(self.session_key, tool_msg)
                    else:
                        # 防爆通过 - 执行工具
                        result = await self.tools.execute(tc.name, tc.arguments)
                        # 记录工具轨迹（供教育链路组装前端面板数据）
                        self.tool_trace.append({
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": result,
                        })
                        # 打印工具结果（截断显示，避免刷屏）
                        preview = result[:200] + "..." if len(result) > 200 else result
                        print(f"  ✅ 结果: {preview}")
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        }
                        messages.append(tool_msg)
                        if self.session_manager:
                            self.session_manager.save_message(self.session_key, tool_msg)

                # continue 回到循环顶部
                # 所有工具结果均已回填 messages，直接进入下一轮推理，
                # 由 LLM 根据工具返回值决定继续调用工具还是给出最终回答
                continue

            # d. 没有工具调用 - 保存历史并返回
            # LLM 不再请求工具意味着本次任务到达终态，
            # 把本轮累计的新消息同步进内存历史供下次对话使用
            if self.session_manager:
                self._save_to_history(messages)

                # 保存最终的 assistant 消息（不含 tool_calls）
                if content:
                    assistant_final_msg = {"role": "assistant", "content": content}
                    self.session_manager.save_message(self.session_key, assistant_final_msg)

            # 教育链路：输出被 max_tokens 截断时给用户明确提示，
            # 避免误以为内容完整（CLI 路径保持原有行为不变）
            final_text = content or ""
            if finish == "length" and meta.get("edu_kind"):
                final_text += "\n\n> ⚠️ 输出被截断，建议减少一次批改的题目数量后重试。"

            return final_text

        # 3. for-else: 超过最大次数
        # 只有每轮都在请求工具调用、从未走到 return 时才会抵达这里，
        # 说明 LLM 在上限轮数内未能完成任务
        return "已达到最大迭代次数，任务未完成。"

    async def _stream_round(
        self,
        messages: list[dict],
        tools_defs: list[dict],
        model: str | None,
        on_delta,
        cancel_event=None,
        extra_params: dict | None = None,
    ) -> tuple[str, list[ToolCallRequest], str]:
        """流式执行一轮推理

        文本增量实时经 on_delta 产出（打字机效果）；工具调用增量由
        Provider 层按 index 归并缓冲，流结束后一次性产出完整
        ToolCallRequest 列表（避免半截函数参数泄露给前端）。

        Returns:
            (content, tool_calls, finish_reason) 三元组，语义与
            非流式 LLMResponse 的同名字段完全一致
        """
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        finish = "stop"

        async for event_type, payload in self.provider.chat_stream_ex(
            messages, tools=tools_defs or None, model=model, extra_params=extra_params
        ):
            # 学生打断检查：流式每个 chunk 都响应取消，中断即时生效
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled()

            if event_type == "delta":
                content_parts.append(payload)
                if on_delta is not None:
                    await on_delta(payload)
            elif event_type == "tool_calls":
                tool_calls = payload
            elif event_type == "done":
                finish = payload

        return "".join(content_parts), tool_calls, finish

    def _build_assistant_message(self, response: LLMResponse) -> dict:
        """构建 assistant 消息（含 tool_calls）

        严格按照 OpenAI 格式构建，reasoning_content 放在顶层。

        Args:
            response: LLM 响应

        Returns:
            dict: assistant 消息字典
        """
        assistant_msg: dict = {
            "role": "assistant",
            "content": response.content
        }

        # 构建 tool_calls 数组
        # 注意 arguments 必须序列化为 JSON 字符串而非 dict——这是
        # OpenAI 协议对 tool_calls 字段类型的硬性要求
        tool_calls = []
        for tc in response.tool_calls:
            tool_call_item = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                }
            }
            tool_calls.append(tool_call_item)

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        # reasoning_content 放在顶层（不能放在 tool_calls 里）
        if response.tool_calls and response.tool_calls[0].reasoning_content:
            assistant_msg["reasoning_content"] = response.tool_calls[0].reasoning_content

        return assistant_msg

    def _check_tool_loop(self, tool_name: str, tool_args_json: str) -> str | None:
        """防爆检测

        检测工具调用的重复频率，防止无限循环。

        Args:
            tool_name: 工具名称
            tool_args_json: 工具参数的 JSON 字符串

        Returns:
            str | None: 熔断/警告消息，或 None 表示放行
        """
        # 以 "工具名+参数" 组成唯一签名：只有完全相同的调用才计为重复，
        # 参数不同的合法调用不会被误判
        signature = f"{tool_name}:{tool_args_json}"

        # 统计签名出现次数
        count = self._tool_call_history.count(signature)

        if count >= self.FUSE_THRESHOLD:
            return f"检测到工具 '{tool_name}' 重复调用已达熔断阈值（{self.FUSE_THRESHOLD}次），已自动终止执行。请尝试其他方案。"
        elif count >= self.WARNING_THRESHOLD:
            return f"警告：工具 '{tool_name}' 重复调用已达 {self.WARNING_THRESHOLD} 次，请检查是否有循环调用问题。"

        # 放行 - 添加到历史
        self._tool_call_history.append(signature)

        # 维护滑动窗口大小
        # 弹出最旧的签名，使统计范围始终限于最近 WINDOW_SIZE 次：
        # 窗口外的"低频重复"视为正常的长间隔调用，不参与熔断计数
        if len(self._tool_call_history) > self.WINDOW_SIZE:
            self._tool_call_history.pop(0)

        return None

    def _sync_history_after_consolidation(self, compressed_messages: list[dict]) -> None:
        """压缩发生后，用压缩后的快照整体重建内存会话历史

        压缩后的快照结构为 [system, 历史摘要, 保留的尾部...]：剔除摘要
        消息后剩余部分即"此刻应有的完整历史"（含本轮已产生的中间消息）。
        直接重建可避免 _save_to_history 的长度差推算在列表长度骤减后
        漏记/错记消息；此后的增量保存仍照常按长度差追加。

        Args:
            compressed_messages: maybe_consolidate 返回的压缩后消息列表
        """
        self._session_history = [
            m for m in compressed_messages[1:]
            if not (m.get("role") == "system"
                    and str(m.get("content", "")).startswith("[历史摘要]"))
        ]

    def _save_to_history(self, messages_snapshot: list[dict]) -> None:
        """保存本轮新增消息到会话历史

        注意：消息已通过 session_manager.save_message 持久化，
        此方法仅更新内存中的 _session_history，供下次对话使用。

        Args:
            messages_snapshot: 当前完整的消息列表
        """
        # 计算新增消息（跳过 system 消息）
        # 新消息数量 = 总消息数 - 已有历史消息数 - 1 (system)
        existing_count = len(self._session_history)

        # 新增的消息（不包含 system）
        new_messages = messages_snapshot[1 + existing_count:]

        # 添加到历史（仅更新内存状态）
        self._session_history.extend(new_messages)

    def clear_history(self) -> None:
        """清空对话历史和工具调用历史

        同时清空内存中的调用签名窗口、内存会话历史与磁盘会话文件，
        三处状态必须一起重置，否则会出现"清空不彻底"的残留会话。
        """
        self._tool_call_history.clear()
        self._session_history.clear()
        # 清空会话文件
        self.session_manager.clear(self.session_key)


# ======================================================================
# 整合对接层（Web 渠道专用，稳定 API）
# ----------------------------------------------------------------------
# 产品定位：中小学生引导式助教——引导思考，不直接给出完整答案
# （人设由 workspace/identity.md 承载，此处只负责流程编排）
#
# 本层为后续前后端整合预留的稳定异步函数：
#   1. run_chat_round(user_id, session_id, user_input) -> str
#   2. create_new_session(user_id, title="新对话") -> str
#   3. list_user_sessions(user_id) -> list[dict]
#   4. load_session_messages(session_id) -> list[dict]
# 以及两个向量占位 Stub（固定签名，队友的向量服务就绪后替换实现）：
#   - extract_long_term_memory(user_id, dialogue_text)
#   - recall_long_term_memory(user_id, query, top_k=3)
#
# 单轮对话流程（run_chat_round）：
#   加载助教人设(system) → 读取 SQLite 历史 → MemoryConsolidator
#   压缩上下文 → 请求 LLM（含工具循环）→ 保存对话入库 → 执行记忆钩子
#
# 约束：本层不写 FastAPI / 网页接口，不做任何向量逻辑；
# 原有 AgentLoop（CLI 渠道）保持原样，两条路径互不影响。
# ======================================================================

from db.database import Database
from agent.memory import MemoryConsolidator
from session.persistent_memory import MemoryManager, PersistentSessionMemory


class _IntegrationRuntime:
    """整合层共享运行时容器

    持有 Web 路径所需的全部共享组件（全应用各一份），
    由 init_integration_runtime() 创建，模块级 _runtime 单例持有。

    Attributes:
        provider: LLM 提供方
        tools: 工具注册表（助教可用 bili_video_search 等）
        context: 上下文构建器（读取 identity.md 助教人设）
        db: SQLite 持久层
        memory_manager: 全局会话记忆池
        model: 默认模型名
        max_iterations: 工具循环安全上限
        _fuse_histories: 每个会话独立的工具防爆滑动窗口
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        context: ContextBuilder,
        db: Database,
        memory_manager: MemoryManager,
        model: str | None = None,
        max_iterations: int = 32,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.context = context
        self.db = db
        self.memory_manager = memory_manager
        self.model = model
        self.max_iterations = max_iterations
        # key = session_id，value = 工具调用签名列表（滑动窗口）
        self._fuse_histories: dict[str, list[str]] = {}


# 模块级单例：Web 集成时调用 init_integration_runtime() 赋值一次
_runtime: _IntegrationRuntime | None = None


def _require_runtime() -> _IntegrationRuntime:
    """获取整合层运行时，未初始化时给出明确报错。"""
    if _runtime is None:
        raise RuntimeError(
            "整合层尚未初始化：请先调用 await init_integration_runtime(...)"
        )
    return _runtime


async def init_integration_runtime(
    provider: LLMProvider,
    tools: ToolRegistry,
    context_builder: ContextBuilder,
    db: Database,
    model: str | None = None,
    workspace: str = ".",
    max_iterations: int = 32,
    token_budget: int = 16000,
) -> None:
    """初始化整合层运行时（Web 集成入口，进程启动时调用一次）

    组装 MemoryConsolidator 与 MemoryManager，并绑定全局单例。
    重复调用会重建运行时（用于测试），生产环境只调用一次。

    Args:
        provider: LLM 提供方实例
        tools: 工具注册表
        context_builder: 上下文构建器（加载 identity.md 助教人设）
        db: 已 connect + init_db 的数据库实例
        model: 默认模型名
        workspace: 工作区路径（压缩历史 HISTORY.md 写入位置）
        max_iterations: 工具循环安全上限
        token_budget: 上下文 Token 预算（超出触发压缩）
    """
    global _runtime

    consolidator = MemoryConsolidator(
        provider=provider,
        workspace=workspace,
        token_budget=token_budget,
    )
    memory_manager = MemoryManager(
        db=db,
        consolidator=consolidator,
        context_builder=context_builder,
    )
    _runtime = _IntegrationRuntime(
        provider=provider,
        tools=tools,
        context=context_builder,
        db=db,
        memory_manager=memory_manager,
        model=model,
        max_iterations=max_iterations,
    )
    print("[整合层] 运行时初始化完成：助教人设 + SQLite 记忆 + Token 压缩已就绪")


# ----------------------------------------------------------------------
# 向量占位 Stub（固定签名，仅打印日志）
# ----------------------------------------------------------------------
# 扩展点说明：队友的向量服务就绪后，直接替换下面两个函数的实现，
# 固定签名不变，调用方（run_chat_round）零改动。
# 注入位置已在 run_chat_round 中用「向量注入扩展点」注释标出。

async def extract_long_term_memory(user_id: int, dialogue_text: str) -> None:
    """[向量占位 Stub] 从对话中抽取长期记忆并向量化入库

    预期实现（队友开发）：调用 LLM 抽取值得长期保存的信息 →
    embedding → 写入向量库（按 user_id 隔离）。

    Args:
        user_id: 用户 id（向量库按用户隔离）
        dialogue_text: 本轮完整对话文本（学生问 + 助教答）
    """
    print(
        f"[向量占位] extract_long_term_memory: user_id={user_id}, "
        f"文本长度={len(dialogue_text)} 字符（向量服务未接入，仅记录日志）"
    )


async def recall_long_term_memory(
    user_id: int, query: str, top_k: int = 3
) -> list[str]:
    """[向量占位 Stub] 按语义召回该用户的长期记忆片段

    预期实现（队友开发）：query → embedding → 向量库相似度检索 →
    返回 top_k 条记忆文本。

    Args:
        user_id: 用户 id
        query: 检索查询（通常是学生本轮的提问）
        top_k: 最多召回几条

    Returns:
        list[str]: 召回的记忆片段列表（未接入向量服务时恒为空列表）
    """
    print(
        f"[向量占位] recall_long_term_memory: user_id={user_id}, "
        f"query={query[:50]!r}, top_k={top_k}（向量服务未接入，返回空列表）"
    )
    return []


# ----------------------------------------------------------------------
# 稳定对接函数 1~4
# ----------------------------------------------------------------------

async def run_chat_round(
    user_id: int, session_id: str, user_input: str
) -> str:
    """单轮对话主入口（Web 渠道调用此函数即可完成一次完整对话）

    流程：归属校验 → 向量召回占位 → 用户消息入库 → 加载助教人设并
    组装压缩上下文 → LLM 工具循环 → 助教回复入库 → 记忆抽取钩子。

    Args:
        user_id: 用户 id
        session_id: 会话 UUID
        user_input: 学生本轮输入

    Returns:
        str: 助教回复文本；LLM/网络异常时返回友好提示，
             会话归属校验失败（跨用户访问）抛 ValueError

    Raises:
        ValueError: session_id 不存在或不属于该 user_id
    """
    rt = _require_runtime()

    # 1. 获取会话记忆体（内部做会话归属校验，防止跨用户读取）
    memory: PersistentSessionMemory = await rt.memory_manager.get_memory(
        user_id, session_id
    )

    # 2. 【向量注入扩展点 · 召回侧】
    #    向量服务就绪后：此处用学生提问召回该用户的长期记忆，
    #    若召回结果非空，把记忆片段拼进 system prompt（如
    #    "## 关于该学生的长期记忆"小节），实现个性化辅导。
    #    当前为占位 Stub，恒返回空列表，不影响流程。
    recalled = await recall_long_term_memory(user_id, user_input)

    # 3. 用户消息入库（写库 + 内存缓存同步）
    await memory.append_message("user", user_input)

    # 4. 加载助教人设（identity.md）→ 组装 system + SQLite 历史，
    #    超出 Token 预算时由 MemoryConsolidator 自动压缩
    #    （system 消息固定在列表头部，受压缩器保护不被删除）
    messages = await memory.build_context(
        system_prompt=rt.context.build_system_prompt()
    )

    # 5. LLM 工具循环（带防爆检测，逻辑与 AgentLoop 对齐）
    try:
        reply = await _run_tool_loop(rt, session_id, messages)
    except Exception as e:
        # 对接层兜底：LLM/网络/工具异常不向 Web 层抛裸异常，
        # 转成学生可理解的友好提示（归属校验等调用方错误仍向上抛）
        print(f"[整合层] 对话异常: {e}")
        return "抱歉，助教这边暂时出了点小状况，请稍后再试一次。"

    # 6. 助教最终回复入库
    await memory.append_message("assistant", reply)

    # 7. 【向量注入扩展点 · 抽取侧】
    #    向量服务就绪后：把本轮完整对话交给抽取钩子，提炼值得
    #    长期保存的信息（学习偏好、薄弱知识点等）入库，
    #    供下次 recall_long_term_memory 个性化召回。
    await extract_long_term_memory(
        user_id, f"学生：{user_input}\n助教：{reply}"
    )

    return reply


async def _run_tool_loop(
    rt: _IntegrationRuntime, session_id: str, messages: list[dict]
) -> str:
    """整合层内部的 LLM 工具调用循环

    与 AgentLoop.run 的循环逻辑对齐（防爆阈值直接复用其类常量），
    区别在于：不经过 SessionManager/JSONL 持久化——工具调用的
    中间消息只在本轮内存中流转，最终回复由 run_chat_round 入库。

    Args:
        rt: 整合层运行时
        session_id: 会话 id（防爆窗口按会话隔离）
        messages: 已组装好的上下文（system + 历史），就地追加

    Returns:
        str: LLM 最终文本回复

    Raises:
        RuntimeError: 达到最大迭代次数仍未给出最终回答
    """
    # 每个会话独立的防爆滑动窗口
    fuse_history = rt._fuse_histories.setdefault(session_id, [])

    for iteration in range(rt.max_iterations):
        print(f"  💭 思考中... (第 {iteration + 1} 轮)", end="", flush=True)
        response = await rt.provider.chat(
            messages=messages,
            tools=rt.tools.get_definitions(),
            model=rt.model,
        )
        print(" ✓")

        # 错误响应不携带有效 content / tool_calls
        if response.finish_reason == "error":
            raise RuntimeError("LLM 返回错误响应")

        # 工具调用分支：回填 assistant + tool 消息后进入下一轮推理
        if response.has_tool_calls:
            # 协议要求：tool 消息必须紧跟在含对应 tool_call_id 的
            # assistant 消息之后，顺序不能颠倒
            assistant_msg: dict = {
                "role": "assistant",
                "content": response.content,
            }
            tool_calls = []
            for tc in response.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(
                            tc.arguments, ensure_ascii=False
                        ),
                    },
                })
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                args_json = json.dumps(tc.arguments, ensure_ascii=False)
                print(f"\n  🛠️  调用工具: {tc.name}({args_json})")

                # 防爆检测（阈值复用 AgentLoop 常量，窗口按会话隔离）
                signature = f"{tc.name}:{args_json}"
                count = fuse_history.count(signature)
                if count >= AgentLoop.FUSE_THRESHOLD:
                    raise RuntimeError(
                        f"工具 '{tc.name}' 重复调用达熔断阈值，已终止"
                    )
                fuse_history.append(signature)
                if len(fuse_history) > AgentLoop.WINDOW_SIZE:
                    fuse_history.pop(0)

                result = await rt.tools.execute(tc.name, tc.arguments)
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  ✅ 结果: {preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            continue

        # 纯文本回复：本轮对话到达终态
        return response.content or ""

    raise RuntimeError(f"已达到最大迭代次数（{rt.max_iterations}），任务未完成")


async def create_new_session(user_id: int, title: str = "新对话") -> str:
    """创建新会话（Web 端"新建对话"按钮调用）

    Args:
        user_id: 用户 id
        title: 会话标题，默认"新对话"

    Returns:
        str: 新会话的 UUID（session_id）
    """
    return await _require_runtime().db.create_session(user_id, title)


async def list_user_sessions(user_id: int) -> list[dict]:
    """获取用户的会话列表（Web 端侧边栏渲染用）

    Args:
        user_id: 用户 id

    Returns:
        list[dict]: 会话列表，按创建时间倒序，每项含
            id / user_id / title / create_time
    """
    rows = await _require_runtime().db.list_sessions(user_id)
    return [dict(row) for row in rows]


async def load_session_messages(session_id: str) -> list[dict]:
    """获取会话的全部聊天记录（Web 端进入会话时回显历史）

    Args:
        session_id: 会话 UUID

    Returns:
        list[dict]: 消息列表（时间正序），每项含
            id / session_id / role / content / timestamp
    """
    rows = await _require_runtime().db.get_messages(session_id)
    return [dict(row) for row in rows]


# ----------------------------------------------------------------------
# 本地调试入口
# ----------------------------------------------------------------------

# 调试专用固定账号：debug_demo 自动创建/复用，无需前端登录
DEBUG_USERNAME = "debug_student"


async def debug_demo() -> None:
    """本地调试入口：无需前端登录页面，直接在控制台与助教对话

    用法：
        python -m agent.loop
    或在代码中：
        from agent.loop import debug_demo; await debug_demo()

    自动完成：加载配置 → 创建 LLM Provider → 注册工具（B 站视频搜索）
    → 初始化 SQLite → 创建/复用调试账号 → 新建会话 → 控制台 REPL。
    """
    import asyncio
    import sys

    from config import load_config
    from providers.openai_compat import OpenAICompatProvider
    from agent.tools.bili_video import BiliVideoSearchTool

    # 1. 配置与 API 密钥校验
    config = load_config()
    if not config.api_key:
        print("错误: 未配置 API 密钥（config.json 的 api_key 或环境变量 NANOCLAW_API_KEY）")
        return

    # 2. 创建 Provider 与工具（调试环境只注册教育场景核心工具，
    #    完整工具注册见 main.py 的 build_agent）
    provider = OpenAICompatProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )
    tools = ToolRegistry()
    tools.register(BiliVideoSearchTool())

    # 3. 上下文构建器（读取 identity.md 助教人设）
    context = ContextBuilder(
        workspace=config.workspace,
        identity_file=config.identity_file,
    )

    # 4. 初始化 SQLite 持久层
    db = Database("nanoclaw.db")
    await db.connect()
    await db.init_db()

    # 5. 初始化整合层运行时
    await init_integration_runtime(
        provider=provider,
        tools=tools,
        context_builder=context,
        db=db,
        model=config.model,
        workspace=config.workspace,
        max_iterations=config.max_iterations,
    )

    # 6. 创建/复用内置测试账号（无需前端登录）
    user = await db.get_user_by_username(DEBUG_USERNAME)
    if user is None:
        user_id = await db.create_user(DEBUG_USERNAME, "debug-only-not-a-login")
        print(f"[调试] 已创建测试账号: {DEBUG_USERNAME} (user_id={user_id})")
    else:
        user_id = user["id"]
        print(f"[调试] 复用测试账号: {DEBUG_USERNAME} (user_id={user_id})")

    # 7. 新建调试会话
    session_id = await create_new_session(user_id, title="调试会话")
    print(f"[调试] 会话已创建: session_id={session_id}")
    print("=" * 60)
    print(f"中小学生引导式助教（{DEBUG_USERNAME}）| /new 新会话  /exit 退出")
    print("=" * 60)

    # 8. 控制台 REPL（input 是阻塞调用，扔线程池避免卡住事件循环）
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\n学生 > ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/new":
            session_id = await create_new_session(user_id, title="调试会话")
            print(f"[调试] 新会话: session_id={session_id}")
            continue

        reply = await run_chat_round(user_id, session_id, user_input)
        print(f"\n助教 > {reply}")

    # 9. 收尾
    await db.close()
    print("[调试] 已退出，对话记录已保存在 nanoclaw.db")


if __name__ == "__main__":
    import asyncio
    import sys

    # Windows 控制台的 UTF-8 输出（避免 emoji/中文乱码）
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(debug_demo())


# ======================================================================
# 改动说明（增量修改，2026-09-04）
# ----------------------------------------------------------------------
# 1. 保留原有功能：AgentLoop 类（CLI 渠道主循环）一行未动。
# 2. 新增「整合对接层」（文件末尾），面向 Web 渠道整合：
#    - _IntegrationRuntime + init_integration_runtime()：共享运行时
#      （Provider/工具/人设/SQLite/MemoryManager 单例组装）
#    - 4 个稳定对接函数：run_chat_round / create_new_session /
#      list_user_sessions / load_session_messages
#    - 2 个向量占位 Stub（固定签名仅打印日志）：
#      extract_long_term_memory / recall_long_term_memory，
#      注入位置在 run_chat_round 中以「向量注入扩展点」注释标出
#    - _run_tool_loop()：整合路径的 LLM 工具循环（防爆阈值复用
#      AgentLoop 常量，窗口按会话隔离；中间消息不入库，只落最终回复）
#    - debug_demo()：内置 DEBUG_USERNAME 测试账号，控制台 REPL
#      本地调试，无需前端登录；python -m agent.loop 直接运行
# 3. 单轮对话流程：加载 identity.md 助教人设 → 读取 SQLite 历史 →
#    MemoryConsolidator 压缩上下文 → 请求 LLM → 保存对话入库 →
#    执行记忆钩子（抽取/召回占位）。
# 4. 产品定位落地：人设由 identity.md 承载（引导式助教，不直接给
#    完整答案）；对接层异常统一转友好提示，不向 Web 层抛裸异常。
# 5. 明确不做：无 FastAPI / 网页接口代码，无任何向量逻辑。
# ======================================================================