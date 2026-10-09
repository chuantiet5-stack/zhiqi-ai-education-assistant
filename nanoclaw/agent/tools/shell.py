"""
Shell 命令执行工具模块

提供安全的 Shell 命令执行能力，包含：
- 危险命令拦截（rm -rf、sudo、fork 炸弹等）
- 执行超时限制（60秒）
- 输出长度截断（10000字符）
- 工作目录限制

使用 asyncio.create_subprocess_shell 异步执行命令。
"""

import asyncio
import os
import re
from typing import Any

from agent.tools.base import Tool


class ExecTool(Tool):
    """
    Shell 命令执行工具

    在指定工作目录下安全执行 Shell 命令，内置危险命令拦截机制。

    安全防护：
    - 拦截递归删除（rm -rf、rmdir /s）
    - 拦截格式化磁盘（format、mkfs）
    - 拦截关机重启（shutdown、reboot）
    - 拦截权限提升（sudo、su）
    - 拦截危险权限（chmod 777）
    - 拦截设备覆写（> /dev/）
    - 拦截下载执行脚本（wget | sh、curl | bash）
    - 拦截网络后门（nc -l、ncat -l）
    - 拦截磁盘覆写（dd if=）
    - 拦截 Fork 炸弹

    使用示例：
        tool = ExecTool(workspace="/path/to/project")
        result = await tool.execute(command="ls -la")
    """

    # 危险命令正则模式列表（忽略大小写）。
    # 采用黑名单而非白名单：LLM 需要的合法命令种类无法穷举，
    # 只能拦截已知高危模式，属于"防误删/防破坏"的软防护而非严格沙箱
    DENY_PATTERNS = [
        # 递归删除
        r"rm\s+.*-r",
        r"rm\s+-rf",
        r"rmdir\s+/s",
        # 格式化磁盘
        r"format\s+",
        r"mkfs",
        # 关机重启
        r"shutdown",
        r"reboot",
        # 权限提升
        r"sudo\s+",
        r"\bsu\b",
        # 危险权限修改
        r"chmod\s+777",
        # 覆盖设备文件
        r">\s*/dev/",
        # 下载执行恶意脚本
        r"wget\s+.*\|\s*sh",
        r"curl\s+.*\|\s*bash",
        # 开网络后门
        r"nc\s+-l",
        r"ncat\s+-l",
        # 磁盘镜像覆写
        r"dd\s+if=",
        # Fork 炸弹
        r":\(\)\{.*\}",
    ]

    # 执行超时（秒）
    TIMEOUT = 60

    # 输出最大长度
    MAX_OUTPUT_LENGTH = 10000

    def __init__(self, workspace: str = ".") -> None:
        """
        Args:
            workspace: 工作目录，命令在此目录下执行，默认当前目录
        """
        self.workspace = os.path.abspath(workspace)
        # 预编译所有拦截正则（忽略大小写，因为 LLM 可能写出 RM -RF 等大写形式）；
        # 构造时编译一次并复用，避免每次执行命令都重复编译的开销
        self._deny_regexes = [re.compile(p, re.IGNORECASE) for p in self.DENY_PATTERNS]

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "执行 Shell 命令。参数 command 为要执行的命令字符串。内置安全防护，拦截危险命令。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 Shell 命令",
                }
            },
            "required": ["command"],
        }

    def _is_dangerous(self, command: str) -> str | None:
        """
        检测命令是否包含危险模式

        遍历 deny_patterns 正则列表，匹配命令字符串。
        如果命中危险模式，返回拦截信息；否则返回 None。

        Args:
            command: 要检测的命令字符串

        Returns:
            str | None: 拦截信息或 None（安全）
        """
        for pattern in self._deny_regexes:
            match = pattern.search(command)
            if match:
                return f"安全拦截：检测到危险命令模式 '{match.group()}'"
        return None

    async def execute(self, **kwargs: Any) -> str:
        """
        异步执行 Shell 命令

        流程：
        1. 安全检测，拦截危险命令
        2. 异步执行命令（在工作目录下）
        3. 超时控制（60秒）
        4. 捕获 stdout + stderr
        5. 输出截断（10000字符）
        6. 附带退出码信息

        Args:
            **kwargs: 工具参数，包含 command 字段

        Returns:
            str: 命令执行结果或错误信息
        """
        command = kwargs.get("command", "")

        if not command:
            return "错误：命令不能为空"

        # 1. 安全检测
        danger_msg = self._is_dangerous(command)
        if danger_msg:
            return danger_msg

        try:
            # 2. 异步执行命令
            # 使用 shell（而非直接 exec）：LLM 给出的命令天然含管道/重定向等
            # shell 语法，交给 shell 解析才符合调用预期；
            # cwd 限定在工作区内，防止命令污染其他目录
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
            )

            # 3. 超时控制
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.TIMEOUT,
                )
            except asyncio.TimeoutError:
                # 超时后必须 kill 并 wait 回收子进程，
                # 否则会留下僵尸进程持续占用系统资源
                process.kill()
                await process.wait()
                return f"命令执行超时（{self.TIMEOUT}秒），已终止"

            # 4. 解码输出
            # errors="replace"：命令输出可能是任意字节（非 UTF-8 文本甚至二进制），
            # 用替换符容错解码，保证解码环节永不抛异常
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            # 5. 拼接输出
            output = stdout_text
            if stderr_text:
                output += f"\n标准错误:\n{stderr_text}"

            # 6. 输出截断
            if len(output) > self.MAX_OUTPUT_LENGTH:
                output = output[:self.MAX_OUTPUT_LENGTH] + "\n...(输出过长，已截断)"

            # 7. 附带退出码：LLM 只能读到文本结果，显式附上退出码才能让它
            #    判断命令是否成功、失败时是否需要重试或换方案
            output += f"\n[退出码: {process.returncode}]"

            return output

        except Exception as e:
            return f"错误：命令执行失败 - {e}"