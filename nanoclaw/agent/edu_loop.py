"""
教育 Agent 主循环

EduAgentLoop 继承 AgentLoop，复用其推理-工具循环体（_execute_loop：
function calling、防爆熔断、流式输出、取消响应），把教育助手的四大
核心功能全部纳入 NanoClaw 核心骨架：

    edu.html → WebChannel → MessageBus（入站）→ Gateway → EduAgentLoop
        ├─ edu_chat      学伴对话（人设 system + SQLite 历史 + 工具循环）
        ├─ edu_grade     题目批改（文本/图片多模态，走视觉模型）
        ├─ edu_plan      周学习计划
        └─ edu_wrongbook 错题集薄弱点分析（rag_search 佐证复习建议）

与基类 AgentLoop 的区别：
- 消息组装：基类从本地 identity.md + JSONL 会话历史构建；本类从
  请求元数据（年级/科目/图片）+ SQLite 对话历史组装
- 持久化：基类经 SessionManager 写 JSONL；本类不落盘（session_manager
  为 None），对话持久化由 Web 层统一写 SQLite chat_messages
- 返回值：基类返回纯文本；本类返回 dict（markdown + 知识点/视频/
  笔记面板数据），由 Gateway 包装进 OutboundMessage.meta 下发

工具调用（取代旧版输出标记方案）：
    bili_video_search  B 站教学视频 → "推荐教学视频"面板
    knowledge_link     知识点百科链接 → "知识点补充"面板
    save_note          保存学伴笔记 → 本地文件 + 保存提示
    rag_search         检索教材（Agent 自主决策调用，教材优先策略由此执行）
"""

import json
import re

from agent.loop import AgentLoop, GenerationCancelled
from agent.tools.edu_tools import _loads_result
from edu_service import (
    EducationServiceError,
    COMPANION_SYSTEM_PROMPT,
    GRADING_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    WRONGBOOK_ANALYSIS_PROMPT,
    CHAT_EXTRA_PARAMS,
    LLM_EXTRA_PARAMS,
    MAX_IMAGE_DATA_URL_LENGTH,
    SUBJECTS_BY_GRADE,
    resolve_grade_label,
    validate_grade_subject,
    strip_box_markers,
)
from providers.base import LLMProvider

# 学伴对话可用工具白名单（rag_search 未注册时过滤后自然为空，无害）
EDU_CHAT_TOOLS = ["rag_search", "bili_video_search", "knowledge_link", "save_note"]


class EduAgentLoop(AgentLoop):
    """
    教育 Agent：学伴对话 / 题目批改 / 学习计划三条链路共用一个实例。

    Attributes:
        provider: OpenAI 兼容 Provider（主模型）
        vision_model: 视觉模型名（图片批改/图片对话），None 时图片请求报 502
        max_iterations: 工具循环上限（教育场景 6 轮足够：教材检索+视频+
            百科+笔记通常一轮并发调用，防呆余量即可，无需基类默认的 32）

    教材检索（RAG）不经服务端预注入：Agent 在推理-工具循环中自主决策
    是否调用 rag_search（何时查、用什么关键词都由模型判断），教材优先
    答题策略写在 COMPANION_SYSTEM_PROMPT 的【查教材】规则里。
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools,
        model: str | None = None,
        vision_model: str | None = None,
        max_iterations: int = 6,
    ) -> None:
        # 刻意不调用 super().__init__：教育链路没有本地会话文件与
        # identity.md 上下文，基类协作所需属性在此显式声明对齐
        self.provider = provider
        self.tools = tools
        self.model = model
        self.vision_model = vision_model
        self.max_iterations = max_iterations

        # 基类协作属性：教育链路不做本地持久化（SQLite 由 Web 层负责）
        self.session_manager = None
        self.context = None
        self.consolidator = None
        self.session_key = "edu"
        self._tool_call_history: list[str] = []
        self.tool_trace: list[dict] = []
        self._session_history: list[dict] = []

    # ------------------------------------------------------------------
    # 入口分派
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        meta: dict | None = None,
        on_delta=None,
    ) -> dict:
        """教育请求入口（覆盖基类签名，返回 dict 结果）

        Args:
            user_message: 学生最新消息文本（日志/兜底用，实际内容在 meta.history）
            meta: 请求元数据（edu_kind / grade_level / subject / history /
                image_data_url / cancel_event 等，由 Web 层组装）
            on_delta: 流式增量回调（仅 edu_chat 流式路径传入）

        Returns:
            dict: edu_chat → {markdown, knowledge, videos, saved_file}
                  edu_plan / edu_grade / edu_wrongbook → {markdown}
                  失败 → {error, http_status}；被打断 → {cancelled: True}
        """
        meta = meta or {}
        kind = meta.get("edu_kind")
        try:
            if kind == "edu_chat":
                return await self._run_chat(meta, on_delta)
            if kind == "edu_plan":
                return await self._run_plan(meta)
            if kind == "edu_grade":
                return await self._run_grade(meta)
            if kind == "edu_wrongbook":
                return await self._run_wrongbook(meta)
            return {"error": f"未知的教育请求类型: {kind}", "http_status": 400}
        except EducationServiceError as e:
            # 参数/输入类 400 与上游模型失败 502，由 Web 层转 HTTP 状态码
            return {"error": str(e), "http_status": e.http_status}
        except GenerationCancelled:
            # 学生打断：回复未完成，Web 层不落库
            return {"cancelled": True, "markdown": ""}
        except Exception as e:
            print(f"[EduAgentLoop] 处理异常: {e}")
            return {"error": "生成出现问题，请稍后重试", "http_status": 502}

    # ------------------------------------------------------------------
    # 学伴对话
    # ------------------------------------------------------------------

    async def _run_chat(self, meta: dict, on_delta) -> dict:
        """学伴对话：人设 + SQLite 历史 + 工具循环（教材检索由 Agent 决策）。"""
        grade_level = meta.get("grade_level")
        subject = meta.get("subject") or "通用"
        history = meta.get("history") or []

        # 顶栏科目选择已移除：subject="通用" 只校验年级；
        # 显式科目（如计划弹窗上下文）按白名单校验
        if subject == "通用":
            if grade_level not in SUBJECTS_BY_GRADE:
                raise EducationServiceError(
                    f"年级编号 {grade_level} 不支持，范围为 1-9（小学 1-6、初中 7-9）"
                )
        else:
            validate_grade_subject(grade_level, subject)
        if not history or history[-1].get("role") != "user":
            raise EducationServiceError("对话历史最后一条必须是用户消息")

        # 只保留最近 16 条，防止上下文超长导致费用与延迟暴涨
        recent = history[-16:]

        # 只有"当前这条消息"带图才走视觉模型：历史更早的图片全部降级为
        # 占位文本。若沿用旧的"历史任一轮带图就走视觉模型"，学生发一张图
        # 后的每次纯文本追问都会把几 MB 的 base64 图重发一遍，视觉模型
        # 处理多模态长上下文极慢（这正是"图片问问题之后输出特别慢"的根因）。
        # 追问若需再看图，模型会提示学生重新发送
        has_image = bool(recent[-1].get("image_data_url"))
        if has_image and not self.vision_model:
            raise EducationServiceError(
                "图片对话需要视觉模型，请在 config.json 的 models.vision 中填写",
                http_status=502,
            )

        llm_messages: list[dict] = [{
            "role": "system",
            "content": COMPANION_SYSTEM_PROMPT.format(
                grade_label=resolve_grade_label(grade_level),
                subject=subject,
            ),
        }]

        for idx, msg in enumerate(recent):
            role = msg.get("role")
            if role == "assistant":
                if msg.get("text"):
                    llm_messages.append({"role": "assistant", "content": msg["text"]})
                continue
            # 用户消息：仅当前最后一条消息带图时构造多模态 content；
            # 历史消息中的图片降级为占位文本（避免每轮重发大 base64）
            image_url = msg.get("image_data_url")
            text = msg.get("text") or ""
            if image_url and idx == len(recent) - 1:
                self._validate_image(image_url)
                llm_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": text or "（学生发来一张图片，请先判断是不是作业，再按规则回应）"},
                    ],
                })
            elif image_url:
                llm_messages.append({
                    "role": "user",
                    "content": text or "（学生此前发过一张图片，如需再讨论请让学生重新发送）",
                })
            else:
                llm_messages.append({"role": "user", "content": text or "（学生发来一张图片）"})

        # 教材检索（RAG）不在此预注入：是否检索、检索关键词均由 Agent
        # 在推理-工具循环中自主决策（rag_search 工具），教材优先答题策略
        # 由 COMPANION_SYSTEM_PROMPT 的【查教材】规则约束

        # 图片轮走视觉模型：部分视觉模型不支持 function calling，
        # 此时禁用工具（无图轮才启用完整教育工具集）
        run_meta = dict(meta)
        if has_image:
            run_meta["model"] = self.vision_model
            run_meta["allowed_tools"] = []
        else:
            run_meta["allowed_tools"] = list(EDU_CHAT_TOOLS)
        run_meta["extra_params"] = CHAT_EXTRA_PARAMS

        reply = await self._execute_loop(llm_messages, meta=run_meta, on_delta=on_delta)

        # 完整回复 = 工具调用前的中间轮正文 + 最后一轮收尾。
        # 多轮工具循环里实质讲解通常先于工具调用输出，只取最后一轮
        # 会把讲解丢成"仅存在于流式增量"的过眼云烟（落库/历史恢复缺失）。
        # 但部分模型（GLM 系列）会在工具执行后把已输出的讲解原样重写，
        # 直接拼接会出现整段重复——合并时自动去重
        full_reply = self._merge_round_texts(self.round_texts, reply)

        return {
            "markdown": strip_box_markers(full_reply),
            **self._panels_from_trace(),
        }

    # ------------------------------------------------------------------
    # 题目批改 / 学习计划（一次性生成，无流式）
    # ------------------------------------------------------------------

    async def _run_grade(self, meta: dict) -> dict:
        """题目识别批改：文本走主模型，图片走视觉模型（无需年级科目）。"""
        question_text = meta.get("question_text")
        image_data_url = meta.get("image_data_url")
        if not question_text and not image_data_url:
            raise EducationServiceError("请上传题目图片或输入题目文本")

        if image_data_url:
            self._validate_image(image_data_url)
            if not self.vision_model:
                raise EducationServiceError(
                    "题目识别服务未配置视觉模型，请在 config.json 的 models.vision 中填写",
                    http_status=502,
                )
            user_content = [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": "请批改这份作业，先判定作答情况，错误的和未作答的题目直接给出正确答案并附讲解。"},
            ]
            user_message: dict = {"role": "user", "content": user_content}
            model = self.vision_model
        else:
            user_message = {
                "role": "user",
                "content": f"请批改以下题目与作答：\n\n{question_text}",
            }
            model = None

        messages = [
            {"role": "system", "content": GRADING_SYSTEM_PROMPT},
            user_message,
        ]
        # 批改是独立单轮任务：不提供工具，保证输出就是纯批改结果
        reply = await self._execute_loop(messages, meta={
            "edu_kind": "edu_grade",
            "model": model,
            "allowed_tools": [],
            "extra_params": LLM_EXTRA_PARAMS,
        })
        return {"markdown": strip_box_markers(reply)}

    async def _run_plan(self, meta: dict) -> dict:
        """构建周学习计划（年级 + 科目 + 薄弱点 + 每周时长）。"""
        grade_level = meta.get("grade_level")
        subject = meta.get("subject") or ""
        validate_grade_subject(grade_level, subject)

        grade_label = resolve_grade_label(grade_level)
        weakness = meta.get("weakness")
        weakness_text = weakness.strip() if weakness and weakness.strip() \
            else "（学生未提供，请按该年级该科目的常见薄弱点分析）"
        weekly_hours = meta.get("weekly_hours", 4)

        user_text = (
            f"请为{grade_label}学生制定{subject}科目的周学习计划。\n"
            f"每周可用时间：约 {weekly_hours} 小时。\n"
            f"薄弱点（学生/家长自述）：{weakness_text}"
        )
        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        reply = await self._execute_loop(messages, meta={
            "edu_kind": "edu_plan",
            "allowed_tools": [],
            "extra_params": LLM_EXTRA_PARAMS,
        })
        return {"markdown": strip_box_markers(reply)}

    async def _run_wrongbook(self, meta: dict) -> dict:
        """错题集薄弱点分析（Web 层从 SQLite 取错题后经总线送入）。

        与旧版 EducationService.analyze_wrong_book 直调 Provider 的区别：
        走 AgentLoop 推理-工具循环——Agent 可自主调用 rag_search 检索
        教材中薄弱知识点的讲解，让"复习建议"有教材依据而非泛泛而谈。
        """
        subject = meta.get("subject") or "全部"
        entries = meta.get("entries") or []
        if not entries:
            return {"markdown": f"暂无「{subject}」的错题记录，继续保持！多做练习巩固已学知识哦～"}

        # 组装错题清单（截断到 40 条防止超长；从批改 markdown 提取知识点标注）
        lines: list[str] = []
        for i, e in enumerate(entries[:40]):
            km = re.search(r"知识点\s*[:：]\s*(.+)", e.get("markdown", ""))
            kp = km.group(1).strip()[:40] if km else ""
            lines.append(
                f"{i + 1}. [{e.get('verdict', '')}] {e.get('question', '')}"
                + (f"（{kp}）" if kp else "")
            )
        user_text = WRONGBOOK_ANALYSIS_PROMPT.format(
            subject=subject, n=len(entries), entries="\n".join(lines)
        )
        messages = [{"role": "user", "content": user_text}]
        reply = await self._execute_loop(messages, meta={
            "edu_kind": "edu_wrongbook",
            # 仅放行教材检索：错题分析面向错题面板（纯 Markdown 渲染，
            # 无知识点/视频面板），百科/视频工具没有展示载体
            "allowed_tools": ["rag_search"],
            "extra_params": {"temperature": 0.3, "max_tokens": 1200},
        })
        return {"markdown": strip_box_markers(reply)}

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image_data_url: str) -> None:
        """校验图片 data URL 格式与大小（400/413）。"""
        if not image_data_url.startswith("data:image/"):
            raise EducationServiceError("图片格式不正确，请上传 JPG/PNG 格式的图片")
        if len(image_data_url) > MAX_IMAGE_DATA_URL_LENGTH:
            raise EducationServiceError("图片过大，请压缩后重试", http_status=413)

    @staticmethod
    def _norm_text(s: str) -> str:
        """规范化文本用于重复比较：仅保留字母/数字/汉字，忽略空白、
        Markdown 符号与标点——格式微调不影响重复判定。"""
        return "".join(ch for ch in s if ch.isalnum())

    @classmethod
    def _merge_round_texts(cls, round_texts: list[str], final_text: str) -> str:
        """合并工具调用前的中间轮正文与最终轮正文，自动去重。

        场景一（保留拼接）：模型先输出完整讲解再调工具，最后一轮只补
          短收尾（如"面板已整理好啦"）——两段内容互补，拼接正确。
        场景二（必须去重）：模型在工具执行后把已输出的讲解原样重写
          一遍（GLM 系列常见行为）——直接拼接会出现整段重复。

        判定规则：两段规范化后，短段的字符基本被长段覆盖（包含关系
        或重合度 ≥ 0.8）视为重复，保留更长（更完整）的一段；否则拼接。

        Returns:
            str: 去重后的完整回复文本
        """
        texts = [t.strip() for t in [*round_texts, final_text] if t and t.strip()]
        if not texts:
            return final_text or ""
        if len(texts) == 1:
            return texts[0]

        def overlap(shorter: str, longer: str) -> float:
            """字符级重合度：交集字符数 / 较短段长度（0~1）。"""
            from collections import Counter
            cs, cl = Counter(shorter), Counter(longer)
            inter = sum((cs & cl).values())
            return inter / max(1, len(shorter))

        merged: list[str] = []
        for cur in texts:
            cur_n = cls._norm_text(cur)
            is_dup = False
            for i, prev in enumerate(merged):
                prev_n = cls._norm_text(prev)
                if not cur_n or not prev_n:
                    continue
                shorter_n, longer_n = (
                    (cur_n, prev_n) if len(cur_n) <= len(prev_n) else (prev_n, cur_n)
                )
                # 包含关系或重合度极高 → 同一内容，保留更完整的一段
                if shorter_n in longer_n or overlap(shorter_n, longer_n) >= 0.8:
                    if len(cur) > len(prev):
                        merged[i] = cur
                    is_dup = True
                    break
            if not is_dup:
                merged.append(cur)
        return "\n\n".join(merged)

    def _panels_from_trace(self) -> dict:
        """从本轮工具轨迹解析面板数据（取每类工具最后一次成功调用）。

        工具结果为 JSON 字符串（edu_tools 约定），解析失败按未调用处理，
        绝不影响正文回复。
        """
        knowledge: list = []
        videos: list = []
        saved_file: dict | None = None

        for entry in reversed(self.tool_trace):
            data = _loads_result(entry.get("result", ""))
            if not isinstance(data, dict):
                continue
            name = entry.get("name")
            if name == "knowledge_link" and not knowledge:
                knowledge = data.get("knowledge") or []
            elif name == "bili_video_search" and not videos:
                videos = data.get("videos") or []
            elif name == "save_note" and saved_file is None:
                saved_file = data

        return {"knowledge": knowledge, "videos": videos, "saved_file": saved_file}
