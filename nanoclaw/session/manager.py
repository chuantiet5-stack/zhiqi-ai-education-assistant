"""
会话管理器模块

SessionManager 负责持久化存储 Agent 的对话历史，包括：
- 按会话 key 分组存储消息（JSONL 格式）
- 加载历史对话记录
- 清除指定会话
- 列出所有已保存的会话

会话文件存储在 workspace/sessions 目录下，每个会话一个 .jsonl 文件。
文件名由 session_key 转换而来（":" 替换为 "_"）。

使用示例：
    manager = SessionManager()

    # 保存消息
    manager.save_message("cli:direct", {"role": "user", "content": "你好"})

    # 加载历史
    history = manager.get_history("cli:direct")

    # 清除会话
    manager.clear("cli:direct")
"""

import json
import os
from datetime import datetime
from typing import Any


class SessionManager:
    """
    会话管理器

    持久化存储 Agent 的对话历史，支持多会话管理。

    所有数据文件存储在 workspace/ 目录下，避免污染项目根目录。
    """

    def __init__(self, sessions_dir: str = "workspace/sessions") -> None:
        """
        Args:
            sessions_dir: 会话文件存储目录，默认 "workspace/sessions"
        """
        self.sessions_dir = sessions_dir

        # 自动创建目录
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _get_session_path(self, session_key: str) -> str:
        """
        获取会话文件路径

        把 session_key 中的 ":" 替换成 "_"，拼接为文件路径。
        例如 "cli:direct" → "workspace/sessions/cli_direct.jsonl"

        Args:
            session_key: 会话标识符

        Returns:
            str: 会话文件的完整路径
        """
        # 替换特殊字符
        safe_key = session_key.replace(":", "_")

        # 拼接路径
        return os.path.join(self.sessions_dir, f"{safe_key}.jsonl")

    def save_message(self, session_key: str, message: dict[str, Any]) -> None:
        """
        保存消息到会话文件

        给 message 加上 "timestamp" 字段（当前时间 ISO 格式），
        以 append 模式打开 JSONL 文件，写入一行 JSON。

        Args:
            session_key: 会话标识符
            message: 消息字典，格式为 {"role": "...", "content": "..."}
        """
        # 获取文件路径
        file_path = self._get_session_path(session_key)

        # 添加时间戳
        message_with_timestamp = message.copy()
        message_with_timestamp["timestamp"] = datetime.now().isoformat()

        # 写入 JSONL 文件（append 模式）：每条消息独立成行，进程崩溃最多
        # 损坏最后一行，不会像整体 JSON 数组那样导致整个历史文件不可读
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                # ensure_ascii=False 支持中文
                json_line = json.dumps(message_with_timestamp, ensure_ascii=False)
                f.write(json_line + "\n")
        except Exception as e:
            # 写入失败不影响主流程，打印警告
            print(f"警告: 保存消息失败 - {e}")

    def get_history(self, session_key: str) -> list[dict[str, Any]]:
        """
        加载会话历史

        读取对应的 JSONL 文件，逐行解析 JSON，跳过空行。
        返回消息列表（去掉 timestamp 字段）。

        Args:
            session_key: 会话标识符

        Returns:
            list[dict]: 消息列表，文件不存在时返回空列表
        """
        # 获取文件路径
        file_path = self._get_session_path(session_key)

        # 文件不存在则返回空列表
        if not os.path.isfile(file_path):
            return []

        # 读取并解析
        history: list[dict[str, Any]] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # 跳过空行
                    if not line.strip():
                        continue

                    # 解析 JSON
                    try:
                        message = json.loads(line)

                        # 去掉 timestamp 字段（OpenAI API 不认识）
                        message_copy = message.copy()
                        message_copy.pop("timestamp", None)

                        history.append(message_copy)

                    except json.JSONDecodeError:
                        # 跳过无法解析的行
                        continue

        except Exception as e:
            print(f"警告: 加载历史失败 - {e}")
            return []

        return history

    def clear(self, session_key: str) -> None:
        """
        清除指定会话

        删除对应的 JSONL 文件（如果存在）。

        Args:
            session_key: 会话标识符
        """
        # 获取文件路径
        file_path = self._get_session_path(session_key)

        # 删除文件（如果存在）
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"警告: 清除会话失败 - {e}")

    def list_sessions(self) -> list[str]:
        """
        列出所有已保存的会话

        扫描 sessions_dir 下的所有 .jsonl 文件，
        返回 session_key 列表（文件名还原为 session_key）。

        Returns:
            list[str]: session_key 列表
        """
        sessions: list[str] = []

        # 检查目录是否存在
        if not os.path.isdir(self.sessions_dir):
            return sessions

        # 扫描 .jsonl 文件
        try:
            for filename in os.listdir(self.sessions_dir):
                # 只处理 .jsonl 文件
                if not filename.endswith(".jsonl"):
                    continue

                # 还原 session_key：去掉扩展名，"_" 替换回 ":"
                # （这是 _get_session_path 中替换的逆操作，要求 session_key 本身不含 "_"）
                safe_key = filename[:-6]  # 去掉 ".jsonl"
                session_key = safe_key.replace("_", ":")

                sessions.append(session_key)

        except Exception:
            pass

        return sessions