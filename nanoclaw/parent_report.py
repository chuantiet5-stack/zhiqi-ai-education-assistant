"""
家长报告模块

将孩子在使用学伴过程中的学习情况定期总结后推送给家长：
- 家长在飞书私聊机器人发送「绑定 孩子用户名」建立绑定关系
  （绑定关系持久化在 SQLite parent_bindings 表）
- 家长发「今日总结 / 本周总结」即时生成学习总结
- 每天固定时间（默认 8:00）自动把当日总结推送给所有绑定的家长

数据来源（全部已在库中，无需额外埋点）：
- chat_messages：孩子与学伴的全部对话（跨该用户所有会话汇总）
- wrong_questions：批改收录的错题

推送路径与 Agent 回复完全一致：
ParentReportService 构造 OutboundMessage → bus.publish_outbound →
Gateway 分发 → 对应渠道 channel.send()。飞书可随时主动推送。

拦截机制：main.py 把 handle_message 注册为 Gateway 的 inbound_hook，
家长指令在进入通用 Agent 之前被截获处理，不会污染 Agent 会话历史。
"""

import asyncio
import re
from datetime import datetime, timedelta

from bus.queue import InboundMessage, MessageBus, OutboundMessage
from providers.base import LLMProvider

# 家长报告只在"人对人私信"性质的渠道启用（web 是孩子自己用，CLI 是开发者）
PARENT_CHANNELS = {"feishu"}

# 绑定引导文案（未绑定时发总结指令的回复）
BIND_GUIDE = (
    "👋 您好！我是知契的家长通知助手。\n"
    "请先绑定孩子：发送「绑定 孩子的用户名」"
    "（用户名是孩子在学伴网页注册的账号）。\n"
    "绑定成功后可以随时发：\n"
    "• 今日总结 —— 看孩子今天学了什么\n"
    "• 本周总结 —— 看最近一周的学习情况\n"
    "• 解绑 —— 解除绑定"
)


def _grade_label(grade_level: int | None) -> str:
    """年级编号 → 可读标签（与 edu_service 口径一致）。"""
    if not grade_level or grade_level < 1:
        return "未设置"
    if grade_level <= 6:
        return f"小学{grade_level}年级"
    return f"初中{grade_level - 6}年级"


class ParentReportService:
    """
    家长报告服务

    职责：
    - handle_message：作为 Gateway 入站钩子，截获家长指令（绑定/解绑/总结）
    - daily_push_task：常驻定时任务，到点给所有绑定家长推送当日总结
    - generate_summary：取库中对话与错题，交给模型生成面向家长的总结
    """

    def __init__(
        self,
        provider: LLMProvider,
        db,
        bus: MessageBus,
        model: str | None = None,
        fallback_model: str | None = None,
        push_hour: int = 20,
        push_minute: int = 0,
    ):
        """
        Args:
            provider: OpenAI 兼容 Provider（总结生成用，建议传 cheap 模型）
            db: Database 实例（绑定关系与学习记录的读写）
            bus: 消息总线（推送经出站队列交给渠道发送）
            model: 总结使用的模型名；None 时用 provider 默认模型
            fallback_model: cheap 模型生成失败/为空时的备用模型（一般传主模型）
            push_hour / push_minute: 每日定时推送的时刻（本地时间）
        """
        self.provider = provider
        self.db = db
        self.bus = bus
        self.model = model
        self.fallback_model = fallback_model
        self.push_hour = push_hour
        self.push_minute = push_minute

    # ------------------------------------------------------------------
    # 入站指令处理（Gateway inbound_hook）
    # ------------------------------------------------------------------

    async def handle_message(self, msg: InboundMessage) -> bool:
        """
        截获家长指令。

        Returns:
            bool: True 表示该消息已处理（不进 Agent）；False 放行
        """
        if msg.channel not in PARENT_CHANNELS:
            return False

        text = (msg.content or "").strip()
        if not text:
            return False

        # ---- 绑定：如「绑定 xiaoming」「绑定：小明同学」----
        m = re.match(r"^绑定[:：,\s]*(.+)$", text)
        if m:
            await self._handle_bind(msg, m.group(1).strip())
            return True

        # ---- 解绑 ----
        if text in ("解绑", "取消绑定", "解除绑定"):
            await self._handle_unbind(msg)
            return True

        # ---- 帮助 ----
        if text in ("帮助", "家长帮助", "使用帮助"):
            await self._send(msg.channel, msg.chat_id, BIND_GUIDE, msg)
            return True

        # ---- 总结：今日总结 / 本周总结 / 学习报告 等 ----
        if "总结" in text or "报告" in text:
            days = 7 if ("周" in text or "七" in text) else 1
            await self._handle_summary(msg, days)
            return True

        # 其余文本放行给通用 Agent（家长也可以正常和机器人聊天）
        return False

    async def _handle_bind(self, msg: InboundMessage, child_username: str) -> None:
        """处理绑定指令：校验用户名存在后落库，回复确认。"""
        if not child_username:
            await self._send(msg.channel, msg.chat_id, BIND_GUIDE, msg)
            return

        user = await self.db.get_user_by_username(child_username)
        if user is None:
            await self._send(
                msg.channel,
                msg.chat_id,
                f"😔 没找到用户名为「{child_username}」的账号。\n"
                "请确认孩子是否已在学伴网页注册（账号区分大小写），或让孩子先注册一个～",
                msg,
            )
            return

        await self.db.bind_parent(msg.channel, msg.sender_id, msg.chat_id, user["id"])
        nickname = user["nickname"] or user["username"]
        await self._send(
            msg.channel,
            msg.chat_id,
            f"✅ 绑定成功！孩子：{nickname}（{_grade_label(user['grade_level'])}）\n"
            "之后每天会自动收到学习总结；也可随时发「今日总结」或「本周总结」查看。\n"
            f"如需更换孩子，重新发送「绑定 新用户名」即可；发「解绑」解除绑定。",
            msg,
        )
        print(f"[家长报告] 绑定成功: {msg.channel}:{msg.sender_id} -> {user['username']}")

    async def _handle_unbind(self, msg: InboundMessage) -> None:
        """处理解绑指令。"""
        removed = await self.db.unbind_parent(msg.channel, msg.sender_id)
        if removed:
            await self._send(msg.channel, msg.chat_id, "已解除绑定，不再接收学习总结。", msg)
        else:
            await self._send(msg.channel, msg.chat_id, "您还没有绑定过孩子哦～" + BIND_GUIDE, msg)

    async def _handle_summary(self, msg: InboundMessage, days: int) -> None:
        """处理总结指令：未绑定则引导，已绑定则生成并发送总结。"""
        binding = await self.db.get_parent_binding(msg.channel, msg.sender_id)
        if binding is None:
            await self._send(msg.channel, msg.chat_id, BIND_GUIDE, msg)
            return
        summary = await self.generate_summary(binding, days=days)
        await self._send(msg.channel, msg.chat_id, summary, msg)

    # ------------------------------------------------------------------
    # 总结生成
    # ------------------------------------------------------------------

    async def generate_summary(self, binding, days: int = 1) -> str:
        """
        生成面向家长的学习总结。

        Args:
            binding: parent_bindings 行（需含 JOIN 出的 child_* 字段）
            days: 统计最近几天的记录（今日=1，本周=7）

        Returns:
            str: 总结文本（可直接推送给家长）
        """
        child_label = binding["child_nickname"] or binding["child_username"]
        now = datetime.now()
        if days <= 1:
            since = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            range_label = "今天"
        else:
            since = (now - timedelta(days=days)).timestamp()
            range_label = f"最近{days}天"

        msgs = await self.db.get_messages_since(binding["child_user_id"], since)
        wrongs = await self.db.get_wrong_since(binding["child_user_id"], since)

        if not msgs and not wrongs:
            return (
                f"📭 {child_label}{range_label}还没有学习记录哦～\n"
                "孩子还没有在学伴网页提问或批改作业，鼓励 TA 试试「讲个知识点」或「批改作业」吧！"
            )

        material = self._build_material(child_label, range_label, msgs, wrongs)
        system_prompt = (
            "你是「知契」的家庭报告助手，负责把孩子的 AI 学伴使用记录总结给家长看。"
            "请按以下结构写一份简短总结：\n"
            "📚 学习情况：孩子最近在学什么、问了哪些问题（合并同类项，不要逐条罗列）\n"
            "✍️ 错题与薄弱点：指出错题反映的知识点方向；没有错题就说暂无错题记录\n"
            "💡 给家长的建议：一两句具体可操作的建议（如让孩子复述、陪做口算）\n"
            "要求：中文；语气温暖亲切，像班主任留言；全文不超过 250 字；"
            "只基于记录内容，不要编造孩子没做过的学习行为。"
        )

        # max_tokens 从 800 提高：cheap 模型（如 Qwen3.5-4B）是思考模型，
        # 会先消耗 reasoning_content 再输出正文，预算太小会导致正文为空
        extra = {"temperature": 0.4, "max_tokens": 2048}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": material},
        ]

        try:
            response = await self.provider.chat(messages, model=self.model, extra_params=extra)
            content = (response.content or "").strip()

            # Provider 出错不抛异常而是返回 finish_reason="error"，需在此甄别；
            # 内容为空（如思考耗尽 token 预算）也触发换备用模型重试一次
            if not content or response.finish_reason == "error":
                print(
                    f"[家长报告] 总结首次生成失败: finish_reason={response.finish_reason} "
                    f"model={self.model or '默认'}，尝试备用模型重试"
                )
                response = await self.provider.chat(
                    messages, model=self.fallback_model, extra_params=extra
                )
                content = (response.content or "").strip()

            if content and response.finish_reason != "error":
                return content
            print(
                f"[家长报告] 总结重试后仍失败: finish_reason={response.finish_reason} "
                f"model={self.fallback_model or '默认'}"
            )
            return "⚠️ 总结生成结果为空，请稍后再试。"
        except Exception as e:
            print(f"[家长报告] 总结生成失败: {e}")
            return "⚠️ 学习总结生成失败（模型请求异常），请稍后再试。"

    @staticmethod
    def _build_material(
        child_label: str, range_label: str, msgs: list, wrongs: list
    ) -> str:
        """把库中记录压缩成给模型的素材文本（控制长度防撑爆 prompt）。"""
        lines = [f"【孩子】{child_label}", f"【统计范围】{range_label}"]

        questions = [
            m["content"].replace("\n", " ")[:80]
            for m in msgs
            if m["role"] == "user" and m["content"].strip()
        ]
        answers = sum(1 for m in msgs if m["role"] == "assistant")
        lines.append(f"【对话规模】孩子发言 {len(questions)} 条，学伴回复 {answers} 条")
        if questions:
            shown = questions[:40]
            lines.append("【孩子提问/发送的内容】（最多展示 40 条，每条截断）")
            lines.extend(f"- {q}" for q in shown)
            if len(questions) > len(shown):
                lines.append(f"…（另有 {len(questions) - len(shown)} 条未展示）")

        lines.append(f"【错题记录】共 {len(wrongs)} 道")
        lines.extend(f"- ({w['verdict']}) {w['question'][:60]}" for w in wrongs)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 定时推送
    # ------------------------------------------------------------------

    async def daily_push_task(self) -> None:
        """
        常驻定时任务：每天 push_hour:push_minute 给所有绑定家长推送当日总结。

        实现为"睡到下一个推送时刻 → 推送 → 继续睡"的循环；
        服务随主进程退出（asyncio 任务被取消）自然终止。
        """
        while True:
            try:
                now = datetime.now()
                target = now.replace(
                    hour=self.push_hour, minute=self.push_minute,
                    second=0, microsecond=0,
                )
                if target <= now:
                    target += timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                print(
                    f"[家长报告] 下次定时推送: {target.strftime('%Y-%m-%d %H:%M')}"
                    f"（约 {wait_seconds / 3600:.1f} 小时后）"
                )
                await asyncio.sleep(wait_seconds)
                await self.push_daily_reports()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # 定时循环绝不能死：任何异常记录后 1 分钟再试
                print(f"[家长报告] 定时任务异常: {e}")
                await asyncio.sleep(60)

    async def push_daily_reports(self) -> None:
        """给所有绑定家长推送当日总结（单个失败不影响其他家长）。"""
        bindings = await self.db.list_parent_bindings()
        if not bindings:
            print("[家长报告] 今日推送：暂无绑定家长，跳过")
            return

        ok = 0
        for b in bindings:
            try:
                summary = await self.generate_summary(b, days=1)
                await self._send(b["channel"], b["chat_id"], summary)
                ok += 1
                print(f"[家长报告] 已推送 -> {b['channel']}（孩子: {b['child_username']}）")
            except Exception as e:
                print(f"[家长报告] 推送失败 -> {b['channel']}（孩子: {b['child_username']}）: {e}")
        print(f"[家长报告] 当日推送完成: {ok}/{len(bindings)}")

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------

    async def _send(
        self, channel: str, chat_id: str, content: str, src_msg: InboundMessage | None = None
    ) -> None:
        """
        经消息总线把文本发给指定渠道会话。

        Args:
            channel: 目标渠道名
            chat_id: 目标会话
            content: 文本内容
            src_msg: 触发本次回复的入站消息（有则携带 reply_to）
        """
        reply_to = None
        if src_msg is not None and isinstance(src_msg.raw, dict):
            reply_to = src_msg.raw.get("message_id")
        await self.bus.publish_outbound(OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
        ))
