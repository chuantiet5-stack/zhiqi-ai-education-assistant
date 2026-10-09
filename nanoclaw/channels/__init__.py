"""渠道适配器模块。

集中导出渠道基类与各渠道实现，供 main.py 按配置选用。
各渠道线程模型差异概述（详见各模块 docstring）：
- CLIChannel：input() 经 asyncio.to_thread 放入线程池，非 TTY 时静默挂起；
- FeishuChannel：lark SDK 在子线程回调，需 run_coroutine_threadsafe 投回主循环；
- WebChannel：FastAPI + WebSocket，每个浏览器标签页独立 uuid 会话。

注意：飞书/Web 的实现仅在被显式导入时加载（如
`from channels.feishu import FeishuChannel`），避免未安装对应
SDK（lark-oapi）时影响 CLI 场景的正常启动。
"""

from channels.base import Channel
from channels.cli import CLIChannel

__all__ = ["Channel", "CLIChannel"]