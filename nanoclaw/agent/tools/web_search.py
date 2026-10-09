"""
互联网搜索工具模块

实现思路：直接抓取 Bing 搜索结果页 HTML，用正则解析出标题、链接、摘要。

为何弃用 ddgs 改用直接抓取 Bing 结果页：
- ddgs 的默认后端（Google/Brave/DuckDuckGo 等）在国内网络环境均不可达，
  且其版本迭代频繁、接口不稳定，经常整体拉不到结果；
- Bing（www.bing.com）是国内可直连的搜索引擎中结果质量较好的一个，
  直接抓取结果页无需第三方搜索依赖，也不需要 API key，
  在目标部署环境（国内服务器/个人电脑）下开箱即用。

超时与容错：httpx 请求设置 12 秒超时防止 Bing 慢响应拖死工具调用；
解析按结果块切分并做数量冗余，个别块解析失败不影响整体返回。
"""

import asyncio
import html
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from agent.tools.base import Tool


class WebSearchTool(Tool):
    """
    互联网搜索工具。

    抓取 Bing 搜索结果页（国内网络可达）并解析出标题、链接、摘要。
    """

    # 输出最大长度：搜索结果直接回传 LLM，截断以控制 token 消耗
    MAX_OUTPUT_LENGTH = 8000

    # 请求超时（秒）：Bing 偶尔响应缓慢，超时上限防止工具调用被无限拖住
    TIMEOUT = 12

    # 常见浏览器 User-Agent，避免被 Bing 返回简化页面
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    @property
    def name(self) -> str:
        """工具名称"""
        return "web_search"

    @property
    def description(self) -> str:
        """工具功能描述"""
        return "搜索互联网获取最新信息。当你需要查询实时信息、最新新闻或不确定的知识时使用。"

    @property
    def parameters(self) -> dict:
        """工具参数定义（JSON Schema 格式）"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几条结果",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, **kwargs: Any) -> str:
        """
        执行互联网搜索。

        Args:
            query: 搜索关键词
            max_results: 最多返回几条结果，默认为 5

        Returns:
            格式化的搜索结果字符串
        """
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return "错误：搜索关键词不能为空"

        try:
            # 执行搜索（同步 HTTP 请求，用 asyncio.to_thread 包装）
            # _search_sync 内部用的是 httpx 同步客户端，若在事件循环里直接调用
            # 会阻塞整个 Agent 的事件循环（期间所有协程都无法推进），
            # 因此丢进线程池执行，保持事件循环可响应
            results = await asyncio.to_thread(
                self._search_sync, query, max_results
            )

            # 处理搜索结果
            if not results:
                return "未找到相关结果"

            # 格式化输出
            output_parts = []
            for index, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                href = result.get("href", "无链接")
                body = result.get("body", "无描述")

                formatted_result = f"### {index}. {title}\n链接: {href}\n{body}\n"
                output_parts.append(formatted_result)

            output = "".join(output_parts)

            # 截断超长输出
            if len(output) > self.MAX_OUTPUT_LENGTH:
                output = output[:self.MAX_OUTPUT_LENGTH]

            return output

        except Exception as e:
            return f"搜索出错: {str(e)}"

    def _strip_tags(self, html_text: str) -> str:
        """
        去除 HTML 标签并反转义实体，得到纯文本。

        Args:
            html_text: 含标签的 HTML 片段

        Returns:
            str: 纯文本
        """
        text = re.sub(r"<[^>]+>", "", html_text)
        return html.unescape(text).strip()

    def _search_sync(self, query: str, max_results: int) -> list:
        """
        同步执行搜索：抓取 Bing 结果页并解析（内部方法）。

        Bing 每条自然结果位于 <li class="b_algo"> 块内，
        包含 <h2><a href="...">标题</a></h2> 和摘要 <p>。

        Args:
            query: 搜索关键词
            max_results: 最多返回几条结果

        Returns:
            list: 结果列表，每项含 title / href / body
        """
        # mkt=zh-CN 指定中文市场，优先返回中文结果；
        # quote_plus 对关键词做 URL 编码，支持中文和特殊字符
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&mkt=zh-CN"

        # 同步 GET：设超时防拖死；跟随重定向（Bing 可能跳转本地化域名）；
        # 带浏览器 UA，否则可能被下发简化版页面导致解析不到结果
        response = httpx.get(
            search_url,
            timeout=self.TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": self.USER_AGENT},
        )
        # HTTP 错误立即抛出，由外层 execute 统一捕获转为错误字符串
        response.raise_for_status()
        page_html = response.text

        # 按结果块切分：Bing 自然搜索结果位于 <li class="b_algo"> 块内，
        # 以该标记 split 后丢弃首段（结果区之前的页面骨架）
        blocks = page_html.split('<li class="b_algo"')[1:]
        results: list[dict[str, str]] = []

        # 多解析一倍数量的块（max_results * 2）：部分块可能是广告/推荐位，
        # 没有 h2>a 结构会被下面的 continue 跳过，冗余解析保证能凑够 max_results
        for block in blocks[: max_results * 2]:
            # 提取链接和标题
            link_match = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not link_match:
                continue

            href = link_match.group(1)
            title = self._strip_tags(link_match.group(2))
            if not href or not title:
                continue

            # 提取摘要（第一个 <p> 块）
            snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
            snippet = self._strip_tags(snippet_match.group(1)) if snippet_match else ""

            results.append({"title": title, "href": href, "body": snippet})

            if len(results) >= max_results:
                break

        return results