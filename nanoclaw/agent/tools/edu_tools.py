"""
教育助手专用工具集

供 EduAgentLoop（教育 Agent 主循环）通过 ToolRegistry 以 function calling
方式调用的三个教育工具，取代旧版"模型输出标记 + 后处理解析"方案：

- EduVideoSearchTool（bili_video_search）：搜索 B 站教学视频，
  结果以 JSON 返回（正文展示 + 右侧"推荐教学视频"面板数据源）
- KnowledgeLinkTool（knowledge_link）：为知识点生成百科直达链接，
  结果以 JSON 返回（右侧"知识点补充"面板数据源）
- SaveNoteTool（save_note）：把笔记内容写入本地"学伴笔记"文件夹，
  返回保存路径（前端"笔记已保存"提示的数据源）

与 agent/tools/bili_video.py 中 BiliVideoSearchTool 的区别：
后者面向 CLI/飞书的通用聊天 Agent，返回 Markdown 人类可读文本；
本模块面向教育链路，返回 JSON 便于 Web 层解析出结构化面板数据。
"""

import asyncio
import json
import os
import re
from datetime import datetime
from urllib.parse import quote, urlparse

from agent.tools.base import Tool
from agent.tools.bili_video import search_bili_videos
from agent.tools.filesystem import WriteFileTool


def _loads_result(text: str) -> dict:
    """解析工具轨迹中的 JSON 结果字符串，失败返回空 dict。"""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


class EduVideoSearchTool(Tool):
    """
    B 站教学视频搜索工具（教育版）

    与通用版的差异：结果以 JSON 字符串返回，EduAgentLoop 从工具轨迹
    中解析出结构化视频列表，经 SSE done 事件下发给前端"推荐教学视频"
    面板；模型侧只会看到 JSON，也便于它理解搜索是否命中。
    """

    @property
    def name(self) -> str:
        return "bili_video_search"

    @property
    def description(self) -> str:
        return (
            "搜索B站教学视频。讲解完一个具体知识点后调用，"
            "系统会把搜索结果展示在右侧'推荐教学视频'面板中。"
            "正文里不要出现视频链接或'可观看视频'之类的文字。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，建议包含年级学科和知识点，如：初二数学 一元二次方程 讲解"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几个视频，默认 3",
                    "default": 3,
                },
            },
            "required": ["keyword"],
        }

    async def execute(self, **kwargs) -> str:
        keyword = kwargs.get("keyword")
        max_results = kwargs.get("max_results", 3)

        if not keyword or not str(keyword).strip():
            return json.dumps({"videos": [], "error": "搜索关键词不能为空"}, ensure_ascii=False)

        try:
            videos = await search_bili_videos(str(keyword).strip(), count=int(max_results))
        except Exception as e:
            # 搜索失败不阻塞主回复：返回空列表让模型知道未命中即可
            print(f"[EduVideoSearchTool] B站搜索失败: {e}")
            return json.dumps({"videos": [], "error": f"搜索失败: {e}"}, ensure_ascii=False)

        return json.dumps({"videos": videos}, ensure_ascii=False)


class KnowledgeLinkTool(Tool):
    """
    知识点百科链接工具

    为模型讲解中出现的知识点（≤3 个教材标准名词）生成"直达词条页"
    的百科链接，供前端"知识点补充"面板展示。

    两级解析策略（与旧版 EducationService._resolve_knowledge_links 同源）：
    1. 百度百科search/word 接口探测：词条存在时 302 到 /item/<词条>，
       返回词条页直链；
    2. 无该词条（302 到搜索页）或探测失败时，改用必应搜索
       "知识点 site:baike.baidu.com"兜底——永远有结果，不会空白页。
    """

    # 跟随重定向会直接落进结果页，无法判断词条是否存在，
    # 因此必须禁用自动重定向，用 302 Location 判断
    REDIRECT_CHECK_TIMEOUT = 5

    @property
    def name(self) -> str:
        return "knowledge_link"

    @property
    def description(self) -> str:
        return (
            "为讲解中出现的知识点生成百科词条链接。回复中讲解了新的知识点/"
            "概念/方法后调用，系统会把链接展示在右侧'知识点补充'面板中。"
            "正文中不要出现这些链接。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "知识点名称列表（教材标准名词，如'圆周率'），"
                        "每个不超过10个字，最多3个"
                    ),
                },
            },
            "required": ["names"],
        }

    @staticmethod
    def _absolute_redirect(location: str, base: str) -> str:
        """把 302 Location 补全为绝对 URL（协议相对/路径相对两种形式）。"""
        if location.startswith("http"):
            return location
        if location.startswith("//"):
            return f"https:{location}"
        if location.startswith("/"):
            parsed = urlparse(base)
            return f"{parsed.scheme}://{parsed.netloc}{location}"
        return location

    async def _check_redirect(self, url: str) -> str | None:
        """请求 URL 并返回 302 Location（不跟随重定向）；失败返回 None。"""
        import aiohttp

        try:
            timeout = aiohttp.ClientTimeout(total=self.REDIRECT_CHECK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url,
                    allow_redirects=False,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                ) as resp:
                    return resp.headers.get("Location")
        except Exception as e:
            print(f"[KnowledgeLinkTool] 重定向探测失败({url}): {e}")
            return None

    async def _resolve_one(self, name: str) -> dict:
        encoded = quote(name)
        location = await self._check_redirect(
            f"https://baike.baidu.com/search/word?word={encoded}"
        )
        if location and "/item/" in location:
            return {
                "name": name,
                "url": self._absolute_redirect(location, "https://baike.baidu.com/search/word"),
                "source": "百度百科",
            }
        # 兜底：必应站内搜百度百科，保证有结果
        return {
            "name": name,
            "url": f"https://cn.bing.com/search?q={encoded}+site%3Abaike.baidu.com",
            "source": "百科搜索",
        }

    async def execute(self, **kwargs) -> str:
        raw_names = kwargs.get("names")
        if isinstance(raw_names, str):
            raw_names = [raw_names]
        if not isinstance(raw_names, list):
            return json.dumps({"knowledge": []}, ensure_ascii=False)

        # 清洗：去空白与 Markdown 粗体残留、去空、去重、截断到 3 个
        names: list[str] = []
        for n in raw_names:
            if not isinstance(n, str):
                continue
            cleaned = n.strip().strip("*").strip()
            if cleaned and cleaned not in names:
                names.append(cleaned)
        names = names[:3]

        if not names:
            return json.dumps({"knowledge": []}, ensure_ascii=False)

        # 多个词条并发探测，整体失败也不阻塞主回复
        try:
            items = list(await asyncio.gather(*(self._resolve_one(n) for n in names)))
        except Exception as e:
            print(f"[KnowledgeLinkTool] 词条解析失败: {e}")
            items = []

        return json.dumps({"knowledge": items}, ensure_ascii=False)


class SaveNoteTool(Tool):
    """
    学伴笔记保存工具

    把学生请求保存的内容（比回复正文更系统的 Markdown 笔记）写入
    本地"学伴笔记"文件夹，文件名带时间戳避免同名覆盖。

    复用 WriteFileTool 执行写入（自带父目录创建与 workspace 路径
    穿越防护）；默认目录为 D 盘"学伴笔记"（不占 C 盘空间且容易找到）。
    """

    @property
    def name(self) -> str:
        return "save_note"

    @property
    def description(self) -> str:
        return (
            "把笔记内容保存为本地 Markdown 文件。当学生请求保存内容"
            "（如'帮我保存''记成笔记''保存到桌面'）时调用。"
            "保存结果由系统在回复后自动提示，正文严禁声称已保存或让学生去找文件。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "笔记文件名（简短，如'浮力知识点笔记'，不要带扩展名）"
                },
                "content": {
                    "type": "string",
                    "description": "要保存的完整笔记内容（Markdown 格式，系统化整理，适合日后复习）"
                },
            },
            "required": ["title", "content"],
        }

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清洗 Windows 非法字符与控制字符；清洗后为空则回退'笔记'。"""
        cleaned = re.sub(r'[\\/:*?"<>|\r\n\t\x00-\x1f]', "", name).strip().strip(".")
        return (cleaned or "笔记")[:50]

    def __init__(self, notes_dir: str | None = None) -> None:
        """
        Args:
            notes_dir: 笔记保存根目录，为 None 时默认 D 盘"学伴笔记"文件夹
        """
        self.notes_dir = os.path.abspath(notes_dir or r"D:\学伴笔记")
        self._file_writer = WriteFileTool(self.notes_dir)

    async def execute(self, **kwargs) -> str:
        title = str(kwargs.get("title") or "学伴笔记")
        content = str(kwargs.get("content") or "").strip()
        if not content:
            return json.dumps({"saved": False, "error": "笔记内容为空"}, ensure_ascii=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{ts}_{self._sanitize_filename(title)}.md"
        try:
            result = await self._file_writer.execute(file_path=file_name, content=content)
            if str(result).startswith("成功"):
                return json.dumps({
                    "saved": True,
                    "path": os.path.join(self.notes_dir, file_name),
                }, ensure_ascii=False)
            return json.dumps({"saved": False, "error": str(result)}, ensure_ascii=False)
        except Exception as e:
            print(f"[SaveNoteTool] 笔记保存失败: {e}")
            return json.dumps({"saved": False, "error": "写入本地文件失败"}, ensure_ascii=False)
