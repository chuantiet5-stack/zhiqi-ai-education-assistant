"""
网页抓取工具模块

抓取指定 URL 的网页内容，转换为纯文本返回。
支持 HTTP/HTTPS 协议，自动处理重定向和超时。
"""

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

from agent.tools.base import Tool


class WebFetchTool(Tool):
    """
    网页抓取工具

    抓取指定 URL 的网页内容，转换为 Markdown 格式纯文本。
    适用于阅读具体网页的详细内容。

    安全防护：
    - 只允许 http/https 协议
    - 15 秒请求超时
    - 自动跟随重定向

    使用示例：
        tool = WebFetchTool()
        result = await tool.execute(url="https://example.com/article")
    """

    # 输出最大长度
    MAX_OUTPUT_LENGTH = 12000

    # 请求超时（秒）
    TIMEOUT = 15

    # 常见浏览器 User-Agent
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "抓取指定 URL 的网页内容。当你需要阅读某个具体网页的详细内容时使用。通常配合 web_search 工具先搜索再抓取。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取的网页 URL",
                }
            },
            "required": ["url"],
        }

    def _validate_url(self, url: str) -> str | None:
        """
        验证 URL 安全性

        检查 URL 格式和协议，确保只允许 http/https。

        Args:
            url: 要验证的 URL

        Returns:
            str | None: 错误信息或 None（安全）
        """
        try:
            parsed = urlparse(url)

            # 检查协议
            if parsed.scheme not in ("http", "https"):
                return "安全拦截：只允许 http/https 协议"

            # 检查是否有有效的主机名
            if not parsed.netloc:
                return "错误：URL 格式无效，缺少主机名"

            return None

        except Exception as e:
            return f"错误：URL 解析失败 - {e}"

    async def execute(self, **kwargs: Any) -> str:
        """
        异步抓取网页内容

        流程：
        1. URL 安全验证
        2. 发送 HTTP GET 请求
        3. HTML 转 Markdown 纯文本
        4. 清理多余空行
        5. 截断超长输出

        Args:
            **kwargs: 工具参数，包含 url 字段

        Returns:
            str: 网页纯文本内容或错误信息
        """
        url = kwargs.get("url", "")

        if not url:
            return "错误：URL 不能为空"

        # 1. URL 安全检查
        validation_error = self._validate_url(url)
        if validation_error:
            return validation_error

        try:
            # 延迟导入：httpx/html2text 属于较重的运行期依赖，
            # 放在函数内可加快模块加载，也避免未安装时拖垮整个工具包的导入
            import httpx

            # 2. 发送 HTTP GET 请求
            # async with 确保请求结束后立即释放连接池；
            # follow_redirects=True：很多站点存在 http->https 或短链跳转，
            # 不跟随重定向只会拿到 3xx 状态码而非正文
            async with httpx.AsyncClient(
                timeout=self.TIMEOUT,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    url,
                    # 伪装常见浏览器 UA：无 UA 或脚本特征 UA 常被 WAF/CDN
                    # 拒绝服务或返回反爬验证页，导致抓不到正文
                    headers={"User-Agent": self.USER_AGENT},
                )

            # 检查状态码：非 2xx 说明页面不存在/被拒绝等，
            # 错误页没有解析价值，直接把状态码报给 LLM
            if response.status_code < 200 or response.status_code >= 300:
                return f"抓取失败：HTTP {response.status_code}"

            # 3. HTML 转纯文本
            html_content = response.text

            import html2text

            # HTML 转 Markdown：保留标题/列表等文本结构，便于 LLM 理解页面
            h = html2text.HTML2Text()
            h.ignore_links = False  # 保留链接：LLM 可能需要进一步抓取链接页
            h.ignore_images = True  # 忽略图片：文本 LLM 无法利用图片内容
            h.body_width = 0  # 不自动换行：避免换行符打断长 URL 和代码块

            text_content = h.handle(html_content)

            # 4. 清理多余空行：HTML 块级元素转换后常产生连续空行，
            #    压缩为最多一个空行以节省回传 LLM 的 token
            text_content = re.sub(r"\n{3,}", "\n\n", text_content)

            # 去除首尾空白
            text_content = text_content.strip()

            # 5. 截断超长输出
            if len(text_content) > self.MAX_OUTPUT_LENGTH:
                text_content = text_content[:self.MAX_OUTPUT_LENGTH] + "\n...(内容过长，已截断)"

            return text_content

        # 异常分支设计：按"最常见 → 最罕见"排列，前两个分支给出可指导
        # LLM 下一步动作的精确提示（超时可重试、连不上可换 URL 或检查网络），
        # 最后用 Exception 兜底承接其余所有异常（SSL 错误、解析库异常等），
        # 统一转为错误字符串返回而非抛出，保证 Agent 循环不中断
        except httpx.TimeoutException:
            return f"错误：请求超时（{self.TIMEOUT}秒）"

        except httpx.ConnectError:
            # 网络层连接失败（含 DNS 解析失败，httpx 无独立的 DNSFailureError）
            return "错误：无法连接到服务器，请检查 URL 和网络是否正常"

        except Exception as e:
            # 兜底分支：未知异常也转为错误文本，LLM 可据此决定重试或放弃
            return f"错误：抓取网页失败 - {e}"