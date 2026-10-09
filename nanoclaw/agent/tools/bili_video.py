"""
B 站教学视频搜索模块

基于 bilibili-api-python 的搜索能力，根据关键词检索 B 站教学视频。
源自 D:\\QQ\\test.py 的调试脚本，精简为可复用的服务模块：
- 提供独立的 search_bili_videos() 协程，供教育服务（学伴对话）直接调用
- 提供 BiliVideoSearchTool，注册为 Agent 工具，供 QQ/企微/飞书等
  聊天渠道的机器人在讲解知识点后主动调用

容错约定：B 站接口变动或网络异常时由调用方决定降级行为，
本模块只负责抛出异常，不吞错误。
"""

import re
from typing import Any

from agent.tools.base import Tool


async def search_bili_videos(keyword: str, count: int = 3) -> list[dict]:
    """
    搜索 B 站教学视频。

    Args:
        keyword: 搜索关键词（如 "初二数学 一元二次方程 讲解"）
        count: 最多返回几个视频

    Returns:
        list[dict]: 视频列表，每项含 title / url / author，最多 count 个；
                    无结果时返回空列表

    Raises:
        Exception: B 站接口调用失败或网络异常时原样抛出
    """
    from bilibili_api.search import search as bili_search

    result = await bili_search(keyword=keyword, page=1)

    # B 站聚合搜索返回结构：result['result'] 是分组列表，
    # 每组含 result_type 字段，视频组为 result_type == 'video'，
    # 组内 data 字段才是视频对象数组（不同版本可能在 data.result 路径下）
    groups: list = []
    if isinstance(result, dict):
        if isinstance(result.get("data"), dict) and isinstance(result["data"].get("result"), list):
            groups = result["data"]["result"]
        elif isinstance(result.get("result"), list):
            groups = result["result"]

    videos: list[dict] = []
    for group in groups:
        if not isinstance(group, dict) or group.get("result_type") != "video":
            continue
        for video in group.get("data") or []:
            if not isinstance(video, dict):
                continue

            # 标题里的关键词会被 <em class="keyword"> 包裹，需清洗
            title = re.sub(r"<[^>]+>", "", video.get("title", "")).strip()

            # 优先用 bvid 构造链接，其次退回 aid
            bvid = video.get("bvid")
            aid = video.get("aid")
            if bvid:
                url = f"https://www.bilibili.com/video/{bvid}"
            elif aid:
                url = f"https://www.bilibili.com/video/av{aid}"
            else:
                continue

            videos.append({
                "title": title,
                "url": url,
                "author": video.get("author", ""),
            })
            if len(videos) >= count:
                return videos

    return videos


class BiliVideoSearchTool(Tool):
    """
    B 站教学视频搜索工具。

    供聊天渠道的 Agent 调用：讲解知识点后搜索相关教学视频，
    把返回的链接附在回复末尾发给用户。
    """

    @property
    def name(self) -> str:
        """工具名称"""
        return "bili_video_search"

    @property
    def description(self) -> str:
        """工具功能描述"""
        return (
            "搜索B站教学视频。当你为学生讲解完一个知识点、或用户想要"
            "相关的学习视频时调用此工具，并把返回的视频链接附在你的回复末尾。"
        )

    @property
    def parameters(self) -> dict:
        """工具参数定义（JSON Schema 格式）"""
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
                    "default": 3
                }
            },
            "required": ["keyword"]
        }

    async def execute(self, **kwargs: Any) -> str:
        """
        执行 B 站视频搜索。

        Args:
            keyword: 搜索关键词
            max_results: 最多返回几个视频，默认 3

        Returns:
            str: Markdown 格式的视频列表（标题 + 链接 + UP主）
        """
        keyword = kwargs.get("keyword")
        max_results = kwargs.get("max_results", 3)

        if not keyword:
            return "错误：搜索关键词不能为空"

        try:
            videos = await search_bili_videos(keyword, count=int(max_results))
        except Exception as e:
            return f"视频搜索出错: {str(e)}"

        if not videos:
            return f"未搜索到与「{keyword}」相关的视频"

        lines = []
        for index, video in enumerate(videos, 1):
            author = f"（UP: {video['author']}）" if video.get("author") else ""
            lines.append(f"{index}. [{video['title']}]({video['url']}){author}")
        return "\n".join(lines)
