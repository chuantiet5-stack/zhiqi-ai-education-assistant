"""
持久化会话记忆模块

在 SessionManager（JSONL 文件存储）和 MemoryConsolidator（Token 压缩）
的基础上，提供对接 SQLite 持久层（db.database.Database）的会话记忆能力，
面向 Web 渠道的多用户场景：

- PersistentSessionMemory    单个会话的记忆体：加载历史、保存消息、
                             组装上下文、触发 Token 压缩
- MemoryManager              全局会话池：按 (user_id, session_id) 缓存
                             记忆体实例，提供 get_memory / release

职责边界（重要）：
- 本模块不实现任何向量 / embedding / 相似度检索逻辑，向量能力由
  队友的向量服务提供，后续以增强方式接入，不改变本模块对外接口
- 本模块不依赖向量服务即可完整运行（规则：向量模块为可选依赖）

上下文组装约定：
- system 身份消息永远放在消息列表头部（messages[0]），
  MemoryConsolidator.maybe_consolidate 固定保护第一条消息不被压缩，
  二者配合保证 Agent 的人设约束在任何压缩轮次后都完好

使用示例：
    db = Database("nanoclaw.db")
    await db.connect()
    await db.init_db()

    manager = MemoryManager(db=db, consolidator=consolidator, context_builder=builder)
    memory = await manager.get_memory(user_id=1, session_id="uuid...")
    await memory.append_message("user", "你好")
    messages = await memory.build_context()      # system 在头，超预算自动压缩
    ...
    manager.release(user_id=1, session_id="uuid...")
"""

from collections import OrderedDict
from typing import Any

from agent.context import ContextBuilder
from agent.memory import MemoryConsolidator
from db.database import Database


class PersistentSessionMemory:
    """
    持久化会话记忆体

    持有单个会话（session_id）的记忆状态，负责：
    - 从 SQLite 加载历史消息到内存缓存
    - append_message 同时写库与更新缓存（写库为主，缓存为镜像）
    - build_context 组装 system + 历史 的完整消息列表，
      超出 Token 预算时调用 consolidator 压缩旧消息
    - 压缩只影响内存缓存，数据库始终保留全量原始历史，
      因此重新 load 即可"恢复完整记忆"，压缩是无损可逆的

    Attributes:
        session_id: 会话 UUID（sessions 表主键）
        history: 内存中的历史消息列表（OpenAI messages 格式，不含 system）
    """

    def __init__(
        self,
        session_id: str,
        db: Database,
        consolidator: MemoryConsolidator,
        context_builder: ContextBuilder = None,
    ) -> None:
        """
        Args:
            session_id: 会话 UUID
            db: 共享的数据库实例（全应用一个连接）
            consolidator: 共享的 Token 压缩器
            context_builder: 上下文构建器，用于生成 system 身份消息；
                也可在 build_context 中直接传入 system_prompt 字符串
        """
        self.session_id = session_id
        self.db = db
        self.consolidator = consolidator
        self.context_builder = context_builder

        # 内存缓存：OpenAI messages 格式（不含 system 消息）
        # 系统消息每次组装时动态生成（内含当前时间），不参与缓存与持久化
        self.history: list[dict[str, Any]] = []

    async def load(self) -> None:
        """
        从数据库加载会话历史到内存缓存

        覆盖式加载（幂等）：无论缓存当前状态如何（可能已被压缩），
        都以数据库中的全量历史为准重建缓存。
        """
        rows = await self.db.get_messages(self.session_id)
        self.history = [
            {"role": row["role"], "content": row["content"]} for row in rows
        ]

    async def append_message(self, role: str, content: str) -> None:
        """
        追加一条消息：先写数据库，再更新内存缓存

        写库失败时异常向上抛出，保证"缓存里有的一定在库里"；
        缓存与库的顺序不可颠倒，否则会出现内存有、库里无的漂移。

        Args:
            role: 角色，user / assistant / system（库层有 CHECK 约束）
            content: 消息正文
        """
        await self.db.add_message(self.session_id, role, content)
        self.history.append({"role": role, "content": content})

    async def build_context(
        self, system_prompt: str = None
    ) -> list[dict[str, Any]]:
        """
        组装发送给 LLM 的完整消息列表，并按需做 Token 压缩

        组装顺序（约定）：system 身份消息固定在列表头部，
        其后为历史消息；MemoryConsolidator 保护第一条消息，
        因此无论压缩多少轮，人设约束始终完好。

        压缩结果会回写到内存缓存（去掉头部 system 消息后的部分），
        避免下一轮对同样的旧消息重复调用 LLM 摘要；
        数据库中的全量历史不受影响。

        Args:
            system_prompt: 直接指定的 system 消息内容；
                为 None 时用 context_builder 动态构建（含实时时间）

        Returns:
            list[dict]: 完整 messages 列表，可直接传给 LLM

        Raises:
            ValueError: 既未提供 system_prompt 也未注入 context_builder
        """
        # 解析 system 消息来源
        if system_prompt is None:
            if self.context_builder is None:
                raise ValueError(
                    "需要提供 system_prompt 参数，或在初始化时注入 context_builder"
                )
            system_prompt = self.context_builder.build_system_prompt()

        # system 永远在头部（规则约束）
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ] + self.history

        # 超预算时压缩（内部保护 messages[0]，即 system 消息）
        compressed = await self.consolidator.maybe_consolidate(messages)

        # 回写缓存：去掉头部 system 消息（它不参与持久化与缓存）
        self.history = compressed[1:]

        return compressed


class MemoryManager:
    """
    全局会话记忆池

    以 (user_id, session_id) 为键缓存 PersistentSessionMemory 实例，
    让 Web 渠道的多用户多会话各自持有独立的记忆体，
    避免每条消息都重新从数据库加载历史。

    缓存策略：
    - 容量上限 max_size，超出时按 LRU（最近最少使用）淘汰，
      被淘汰的实例数据已全部落库，随取随建，无丢失风险

    Attributes:
        db: 共享数据库实例
        consolidator: 共享 Token 压缩器
        context_builder: 共享上下文构建器
    """

    def __init__(
        self,
        db: Database,
        consolidator: MemoryConsolidator,
        context_builder: ContextBuilder = None,
        max_size: int = 128,
    ) -> None:
        """
        Args:
            db: 共享数据库实例
            consolidator: 共享 Token 压缩器
            context_builder: 共享上下文构建器（生成 system 消息）
            max_size: 缓存容量上限，超出按 LRU 淘汰
        """
        self.db = db
        self.consolidator = consolidator
        self.context_builder = context_builder
        self.max_size = max_size

        # OrderedDict 支持move_to_end，实现轻量 LRU
        self._pool: OrderedDict[tuple[int, str], PersistentSessionMemory] = (
            OrderedDict()
        )

    async def get_memory(
        self, user_id: int, session_id: str
    ) -> PersistentSessionMemory:
        """
        获取会话记忆体（不存在则创建并加载历史）

        Args:
            user_id: 用户 id，用于校验会话归属与构造缓存键
            session_id: 会话 UUID

        Returns:
            PersistentSessionMemory: 就绪的记忆体（历史已加载）

        Raises:
            ValueError: 会话不存在，或不属于该用户
        """
        key = (user_id, session_id)

        # 缓存命中：移到队尾标记"最近使用"
        if key in self._pool:
            self._pool.move_to_end(key)
            return self._pool[key]

        # 归属校验：防止跨用户读取他人会话历史（越权防线放在数据访问层）
        session = await self.db.get_session(session_id)
        if session is None:
            raise ValueError(f"会话不存在: {session_id}")
        if session["user_id"] != user_id:
            raise ValueError(f"会话 {session_id} 不属于用户 {user_id}")

        # 创建记忆体并从数据库加载全量历史
        memory = PersistentSessionMemory(
            session_id=session_id,
            db=self.db,
            consolidator=self.consolidator,
            context_builder=self.context_builder,
        )
        await memory.load()

        # 入池；超容量时淘汰队首（最久未使用）
        self._pool[key] = memory
        while len(self._pool) > self.max_size:
            self._pool.popitem(last=False)

        return memory

    def release(self, user_id: int, session_id: str) -> None:
        """
        释放会话记忆体（从缓存移除）

        仅做缓存淘汰：消息在 append_message 时已实时落库，
        release 不需要任何落盘动作。之后再次 get_memory 会
        重新从数据库加载全量历史。

        Args:
            user_id: 用户 id
            session_id: 会话 UUID
        """
        self._pool.pop((user_id, session_id), None)

    def clear(self) -> None:
        """清空整个会话池（用于进程退出或测试）。"""
        self._pool.clear()


if __name__ == "__main__":
    # 自测脚本：python -m session.persistent_memory
    # 用 :memory: 数据库 + 假 LLM provider，不依赖外部服务

    class FakeProvider:
        """假 LLM provider，仅满足 consolidator 的接口需求"""

        async def chat(self, messages, tools=None):
            class Resp:
                content = "（假摘要）用户与助手互相问候。"

            return Resp()

    class FakeContextBuilder:
        """假上下文构建器，返回固定 system 消息"""

        @staticmethod
        def build_system_prompt() -> str:
            return "你是 NanoClaw。"

    async def _test() -> None:
        from agent.memory import MemoryConsolidator

        async with Database(":memory:") as db:
            user_id = await db.create_user("alice", "pwd123")
            session_id = await db.create_session(user_id)

            consolidator = MemoryConsolidator(
                provider=FakeProvider(), workspace=".", token_budget=50
            )
            manager = MemoryManager(
                db=db,
                consolidator=consolidator,
                context_builder=FakeContextBuilder(),
            )

            # 1. get_memory：首次创建并加载（此时历史为空）
            memory = await manager.get_memory(user_id, session_id)
            assert memory.history == [], "新会话历史应为空"

            # 2. append_message：写库 + 缓存同步
            await memory.append_message("user", "你好")
            await memory.append_message("assistant", "你好，有什么可以帮你？")
            assert len(memory.history) == 2

            # 3. build_context：system 在头部
            messages = await memory.build_context()
            assert messages[0]["role"] == "system", "system 必须在列表头部"
            assert messages[0]["content"] == "你是 NanoClaw。"
            assert len(messages) == 3, "system + 2 条历史"
            print("build_context OK:", [(m["role"], m["content"][:20]) for m in messages])

            # 4. 压缩回写：token_budget=50，两条短消息不足以触发压缩（<8条消息）
            assert len(memory.history) == 2, "未触发压缩时缓存不变"

            # 5. 缓存复用：再次 get_memory 返回同一实例
            memory2 = await manager.get_memory(user_id, session_id)
            assert memory2 is memory, "缓存应复用同一实例"

            # 6. release 后重新加载：数据来自数据库（全量）
            manager.release(user_id, session_id)
            memory3 = await manager.get_memory(user_id, session_id)
            assert memory3 is not memory
            assert len(memory3.history) == 2, "release 后应从库中重新加载历史"
            print("reload from db OK:", memory3.history)

            # 7. 归属校验：他人会话拒绝访问
            user2 = await db.create_user("bob", "pwd456")
            try:
                await manager.get_memory(user2, session_id)
                assert False, "应当拒绝跨用户访问"
            except ValueError as e:
                print("归属校验 OK:", e)

            # 8. 持久化验证：绕过缓存直查数据库
            rows = await db.get_messages(session_id)
            assert len(rows) == 2, "消息应已实时落库"
            print("db persist OK:", [(r["role"], r["content"]) for r in rows])

    import asyncio

    asyncio.run(_test())
