"""
异步 SQLite 持久层模块

基于 aiosqlite 提供用户、会话、聊天消息三张表的异步访问，
供 Web 渠道（注册/登录、会话管理、聊天记录）使用。

设计要点：
- 单连接 + WAL 模式：aiosqlite 连接内部跑独立线程，配合 WAL
  允许读写并发；整个应用共享一个 Database 实例即可
- 外键约束：SQLite 默认不启用外键，连接时显式 PRAGMA 开启，
  并对两张子表使用 ON DELETE CASCADE（删用户连带删会话，
  删会话连带删消息）
- 密码存储：bcrypt.hashpw 返回 bytes，直接以 BLOB 落库，
  校验时用 checkpw 原样比对，无需编解码

表结构：
    users            用户表（id 自增主键 / username 唯一 / bcrypt 密码哈希）
    sessions         会话表（UUID 字符串主键，归属某用户）
    chat_messages    聊天消息表（归属某会话，role 限定三种取值）
    wrong_questions  错题集表（归属某用户，批改出的错题/未作答题云端保存）
"""

import asyncio
import time
import uuid
from pathlib import Path

import aiosqlite
import bcrypt

# 默认数据库文件路径（项目根目录下）
DEFAULT_DB_PATH = "nanoclaw.db"


class Database:
    """
    异步 SQLite 数据库封装

    持有一条 aiosqlite 连接，提供三张表的增删改查方法。
    使用方式：

        db = Database("nanoclaw.db")
        await db.connect()
        await db.init_db()          # 首次运行建表
        ...                         # 业务调用
        await db.close()

    也支持异步上下文管理器（自动 connect + init_db + close）：

        async with Database("nanoclaw.db") as db:
            ...
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """
        Args:
            db_path: SQLite 数据库文件路径，父目录不存在时自动创建
        """
        self.db_path = db_path
        self._conn: aiosqlite.Connection = None

    # ------------------------------------------------------------------
    # 连接与建表
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """建立数据库连接并启用外键约束与 WAL 模式。"""
        # 确保父目录存在（如传入 "data/nanoclaw.db"）
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self.db_path)
        # 查询结果默认以 sqlite3.Row 返回，支持 row["字段名"] 访问
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode = WAL")

    async def init_db(self) -> None:
        """创建三张表及索引（使用 IF NOT EXISTS，可重复调用）。"""
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                nickname      TEXT,
                grade_level   INTEGER,
                avatar        TEXT,
                create_time   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                user_id     INTEGER NOT NULL
                            REFERENCES users(id) ON DELETE CASCADE,
                title       TEXT NOT NULL DEFAULT '新对话',
                create_time REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL
                           REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL
                           CHECK (role IN ('user', 'assistant', 'system')),
                content    TEXT NOT NULL,
                timestamp  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wrong_questions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL
                            REFERENCES users(id) ON DELETE CASCADE,
                question    TEXT NOT NULL,
                verdict     TEXT NOT NULL,
                markdown    TEXT NOT NULL,
                subject     TEXT NOT NULL DEFAULT '其他',
                create_time REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS parent_bindings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                channel       TEXT NOT NULL,
                sender_id     TEXT NOT NULL,
                chat_id       TEXT NOT NULL,
                child_user_id INTEGER NOT NULL
                              REFERENCES users(id) ON DELETE CASCADE,
                create_time   REAL NOT NULL,
                -- 同一渠道同一家长只绑一个孩子，重新绑定即换绑
                UNIQUE(channel, sender_id)
            );

            -- 外键查询是最高频路径，建索引避免全表扫描
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_wrong_questions_user_id
                ON wrong_questions(user_id);
            CREATE INDEX IF NOT EXISTS idx_parent_bindings_child
                ON parent_bindings(child_user_id);
        """)

        # 旧库迁移：users 表补充个人资料字段（昵称/年级/头像）。
        # SQLite 不支持 ADD COLUMN IF NOT EXISTS，列已存在时抛
        # OperationalError("duplicate column name")，直接忽略即可
        for column_def in ("nickname TEXT", "grade_level INTEGER", "avatar TEXT"):
            try:
                await self._conn.execute(f"ALTER TABLE users ADD COLUMN {column_def}")
            except aiosqlite.OperationalError:
                pass

        # 旧库迁移：wrong_questions 表补充学科分类列（历史错题归入"其他"）
        try:
            await self._conn.execute(
                "ALTER TABLE wrong_questions ADD COLUMN subject TEXT NOT NULL DEFAULT '其他'"
            )
        except aiosqlite.OperationalError:
            pass

        await self._conn.commit()

    async def close(self) -> None:
        """提交未完成事务并关闭连接。"""
        if self._conn is not None:
            await self._conn.commit()
            await self._conn.close()
            self._conn = None

    # 支持异步上下文管理器用法
    async def __aenter__(self) -> "Database":
        await self.connect()
        await self.init_db()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # users 表
    # ------------------------------------------------------------------

    async def create_user(
        self,
        username: str,
        password: str,
        nickname: str | None = None,
        grade_level: int = 3,
        avatar: str = "🧑‍🎓",
    ) -> int:
        """
        注册新用户（bcrypt 加密密码后落库）。

        Args:
            username: 用户名（需唯一）
            password: 明文密码
            nickname: 昵称，缺省用用户名
            grade_level: 年级编号（1-9），默认 3（小学三年级）
            avatar: 头像（emoji 或图片 data URL），默认 🧑‍🎓

        Returns:
            int: 新用户的 id

        Raises:
            sqlite3.IntegrityError: 用户名已存在
        """
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cur = await self._conn.execute(
            "INSERT INTO users (username, password_hash, nickname, grade_level, avatar, create_time) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, password_hash, nickname or username, grade_level, avatar, time.time()),
        )
        await self._conn.commit()
        return cur.lastrowid

    async def get_user_by_username(self, username: str) -> aiosqlite.Row:
        """按用户名查询用户，不存在时返回 None。"""
        cur = await self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        )
        return await cur.fetchone()

    async def verify_password(self, username: str, password: str) -> aiosqlite.Row:
        """
        校验用户名密码。

        Returns:
            aiosqlite.Row: 验证通过返回用户行，否则返回 None
        """
        user = await self.get_user_by_username(username)
        if user is None:
            return None
        if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
            return user
        return None

    async def update_user_profile(
        self, username: str, nickname: str, grade_level: int, avatar: str
    ) -> aiosqlite.Row | None:
        """
        更新用户资料（昵称/年级/头像），返回更新后的用户行。

        Args:
            username: 用户名（定位用户，不可改）
            nickname: 昵称
            grade_level: 年级编号（1-9）
            avatar: 头像（emoji 或图片 data URL）
        """
        await self._conn.execute(
            "UPDATE users SET nickname = ?, grade_level = ?, avatar = ? WHERE username = ?",
            (nickname, grade_level, avatar, username),
        )
        await self._conn.commit()
        return await self.get_user_by_username(username)

    # ------------------------------------------------------------------
    # sessions 表
    # ------------------------------------------------------------------

    async def create_session(self, user_id: int, title: str = "新对话") -> str:
        """
        创建新会话。

        Args:
            user_id: 归属用户 id
            title: 会话标题，默认"新对话"

        Returns:
            str: 新会话的 UUID 主键
        """
        session_id = str(uuid.uuid4())
        await self._conn.execute(
            "INSERT INTO sessions (id, user_id, title, create_time) VALUES (?, ?, ?, ?)",
            (session_id, user_id, title, time.time()),
        )
        await self._conn.commit()
        return session_id

    async def get_session(self, session_id: str) -> aiosqlite.Row:
        """按 id 查询会话，不存在时返回 None。"""
        cur = await self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        return await cur.fetchone()

    async def list_sessions(self, user_id: int) -> list[aiosqlite.Row]:
        """列出某用户的全部会话，按创建时间倒序（最新的在前）。"""
        cur = await self._conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY create_time DESC",
            (user_id,),
        )
        return await cur.fetchall()

    async def update_session_title(self, session_id: str, title: str) -> None:
        """更新会话标题（如用首条消息自动命名）。"""
        await self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
        )
        await self._conn.commit()

    async def delete_session(self, session_id: str) -> None:
        """删除会话（其下所有聊天消息由外键级联删除）。"""
        await self._conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # chat_messages 表
    # ------------------------------------------------------------------

    async def add_message(self, session_id: str, role: str, content: str) -> int:
        """
        向会话追加一条消息。

        Args:
            session_id: 会话 UUID
            role: 角色，只能是 user / assistant / system
            content: 消息正文

        Returns:
            int: 新消息的 id

        Raises:
            sqlite3.IntegrityError: role 非法或 session_id 不存在
        """
        cur = await self._conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        await self._conn.commit()
        return cur.lastrowid

    async def get_messages(
        self, session_id: str, limit: int = None
    ) -> list[aiosqlite.Row]:
        """
        查询会话的聊天记录，按时间正序（早的在前）。

        Args:
            session_id: 会话 UUID
            limit: 最多返回几条，None 表示全部；指定时取最近的 N 条
                   （返回顺序仍为正序，方便直接拼进对话上下文）
        """
        if limit is None:
            cur = await self._conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            return await cur.fetchall()
        # 先倒序取最近 N 条，再翻回正序
        cur = await self._conn.execute(
            "SELECT * FROM (SELECT * FROM chat_messages WHERE session_id = ? "
            "ORDER BY timestamp DESC, id DESC LIMIT ?) ORDER BY timestamp, id",
            (session_id, limit),
        )
        return await cur.fetchall()

    # ------------------------------------------------------------------
    # wrong_questions 表（错题集：批改出的错题/未作答题云端保存）
    # ------------------------------------------------------------------

    async def add_wrong_questions(
        self, user_id: int, items: list[dict]
    ) -> list[int]:
        """
        批量收录错题（一次批改可能出多道错题，整批一次事务落库）。

        Args:
            user_id: 归属用户 id
            items: 错题列表，每项含 question（题干摘要）/ verdict（判定，
                   如 "❌ 错误"）/ markdown（完整批改讲解块）/ subject（学科，
                   缺省按"其他"入库）

        Returns:
            list[int]: 按传入顺序返回各条新记录的 id
        """
        now = time.time()
        ids = []
        # executemany 不回填 lastrowid，逐条 INSERT 才能收集每条的 id
        for it in items:
            cur = await self._conn.execute(
                "INSERT INTO wrong_questions (user_id, question, verdict, markdown, subject, create_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    it["question"],
                    it["verdict"],
                    it["markdown"],
                    it.get("subject") or "其他",
                    now,
                ),
            )
            ids.append(cur.lastrowid)
        await self._conn.commit()
        return ids

    async def list_wrong_questions(
        self, user_id: int, limit: int = 100, subject: str | None = None
    ) -> list[aiosqlite.Row]:
        """列出某用户的错题，按收录时间倒序（最新的在前），最多 limit 条。

        Args:
            subject: 学科过滤（如 "数学"）；None / "全部" 时不过滤
        """
        if subject and subject != "全部":
            cur = await self._conn.execute(
                "SELECT * FROM wrong_questions WHERE user_id = ? AND subject = ? "
                "ORDER BY create_time DESC, id DESC LIMIT ?",
                (user_id, subject, limit),
            )
        else:
            cur = await self._conn.execute(
                "SELECT * FROM wrong_questions WHERE user_id = ? "
                "ORDER BY create_time DESC, id DESC LIMIT ?",
                (user_id, limit),
            )
        return await cur.fetchall()

    async def delete_wrong_question(self, user_id: int, question_id: int) -> bool:
        """
        删除单条错题（限定归属用户，防止越权删他人错题）。

        Returns:
            bool: 是否确实删除了一行
        """
        cur = await self._conn.execute(
            "DELETE FROM wrong_questions WHERE id = ? AND user_id = ?",
            (question_id, user_id),
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def clear_wrong_questions(self, user_id: int) -> int:
        """
        清空某用户全部错题。

        Returns:
            int: 删除的条数
        """
        cur = await self._conn.execute(
            "DELETE FROM wrong_questions WHERE user_id = ?", (user_id,)
        )
        await self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # parent_bindings 表（家长报告：飞书/QQ 家长账号与孩子的绑定关系）
    # ------------------------------------------------------------------

    async def bind_parent(
        self, channel: str, sender_id: str, chat_id: str, child_user_id: int
    ) -> None:
        """
        绑定家长与孩子。

        同一渠道同一家长（sender_id）只保留一条绑定记录，重复绑定
        视为换绑（INSERT OR REPLACE 覆盖旧记录并刷新 chat_id，
        家长换群/重开会话后 chat_id 可能变化）。

        Args:
            channel: 渠道名（feishu / qq / wecom）
            sender_id: 家长在渠道内的用户标识
            chat_id: 家长与机器人的会话标识（推送目的地）
            child_user_id: 孩子的 users.id
        """
        await self._conn.execute(
            "INSERT OR REPLACE INTO parent_bindings "
            "(channel, sender_id, chat_id, child_user_id, create_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (channel, sender_id, chat_id, child_user_id, time.time()),
        )
        await self._conn.commit()

    async def unbind_parent(self, channel: str, sender_id: str) -> bool:
        """
        解除家长绑定。

        Returns:
            bool: 是否确实存在并删除了一条绑定
        """
        cur = await self._conn.execute(
            "DELETE FROM parent_bindings WHERE channel = ? AND sender_id = ?",
            (channel, sender_id),
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def get_parent_binding(
        self, channel: str, sender_id: str
    ) -> aiosqlite.Row | None:
        """查询家长绑定（JOIN users 带出孩子用户名/昵称/年级），未绑定时返回 None。"""
        cur = await self._conn.execute(
            "SELECT b.*, u.username AS child_username, u.nickname AS child_nickname, "
            "u.grade_level AS child_grade "
            "FROM parent_bindings b JOIN users u ON b.child_user_id = u.id "
            "WHERE b.channel = ? AND b.sender_id = ?",
            (channel, sender_id),
        )
        return await cur.fetchone()

    async def list_parent_bindings(self) -> list[aiosqlite.Row]:
        """列出全部家长绑定（定时推送用），带出孩子信息。"""
        cur = await self._conn.execute(
            "SELECT b.*, u.username AS child_username, u.nickname AS child_nickname, "
            "u.grade_level AS child_grade "
            "FROM parent_bindings b JOIN users u ON b.child_user_id = u.id "
            "ORDER BY b.create_time ASC"
        )
        return await cur.fetchall()

    async def get_messages_since(
        self, user_id: int, since_ts: float, limit: int = 200
    ) -> list[aiosqlite.Row]:
        """
        取某用户自 since_ts 起的学伴对话消息（跨该用户全部会话）。

        Args:
            user_id: 用户 id
            since_ts: 起始时间戳（含）
            limit: 最多返回条数（防止超长对话撑爆总结 prompt）

        Returns:
            list: [{role, content, timestamp}]，按时间正序
        """
        cur = await self._conn.execute(
            "SELECT m.role, m.content, m.timestamp "
            "FROM chat_messages m JOIN sessions s ON m.session_id = s.id "
            "WHERE s.user_id = ? AND m.timestamp >= ? AND m.role != 'system' "
            "ORDER BY m.timestamp ASC LIMIT ?",
            (user_id, since_ts, limit),
        )
        return await cur.fetchall()

    async def get_wrong_since(
        self, user_id: int, since_ts: float, limit: int = 20
    ) -> list[aiosqlite.Row]:
        """取某用户自 since_ts 起收录的错题（题干+判定），最新在前。"""
        cur = await self._conn.execute(
            "SELECT question, verdict, create_time FROM wrong_questions "
            "WHERE user_id = ? AND create_time >= ? "
            "ORDER BY create_time DESC LIMIT ?",
            (user_id, since_ts, limit),
        )
        return await cur.fetchall()


if __name__ == "__main__":
    # 自测脚本：python -m db.database
    async def _test() -> None:
        async with Database(":memory:") as db:
            uid = await db.create_user("alice", "secret123")
            print("user id:", uid)
            user = await db.verify_password("alice", "secret123")
            print("login ok:", user["username"] == "alice")
            print("wrong pwd:", await db.verify_password("alice", "bad") is None)

            sid = await db.create_session(uid)
            await db.add_message(sid, "user", "你好")
            await db.add_message(sid, "assistant", "你好，有什么可以帮你？")
            msgs = await db.get_messages(sid)
            print("messages:", [(m["role"], m["content"]) for m in msgs])
            print("sessions:", len(await db.list_sessions(uid)))

            await db.delete_session(sid)
            print("after delete:", await db.get_messages(sid))

    asyncio.run(_test())
