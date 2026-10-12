"""
RAG 搜索工具

供 Agent 主动调用知识库检索教材片段。
"""

import asyncio

from agent.tools.base import Tool
from agent.rag.retriever import FAISSRetriever


class RAGSearchTool(Tool):
    """
    RAG 知识库搜索工具

    在已导入的教材/知识库中检索与问题相关的片段。
    """

    def __init__(self, retriever: FAISSRetriever):
        """
        Args:
            retriever: FAISS 检索器实例
        """
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "在教育知识库中搜索与问题相关的教材片段。"
            "当用户询问知识点、概念、定理、公式等学习内容时，"
            "回答前先调用此工具检索教材，优先依据教材内容作答，"
            "未检索到时再按自身知识回答。返回片段包含来源和相似度信息。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索问题或关键词，如 '一元二次方程的判别式'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数量，默认 3",
                    "default": 3
                },
                "threshold": {
                    "type": "number",
                    "description": "相似度阈值 0-1，默认 0.5，低于此值的片段将被过滤",
                    "default": 0.5
                }
            },
            "required": ["query"]
        }

    async def execute(self, **kwargs) -> str:
        """
        执行知识库检索

        Args:
            query: 搜索关键词
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            str: 格式化的检索结果
        """
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", getattr(self.retriever, "default_top_k", 3))
        threshold = kwargs.get("threshold", getattr(self.retriever, "default_threshold", 0.5))

        if not query:
            return "错误：查询内容不能为空"

        try:
            # retrieve 含同步网络请求（Embedding），放线程池执行避免阻塞事件循环
            results = await asyncio.to_thread(
                self.retriever.retrieve, query, top_k=top_k, threshold=threshold
            )
        except Exception as e:
            return f"错误：检索失败 - {e}"

        if not results:
            return f"未找到与「{query}」相关的知识点"

        output_lines = []
        for i, r in enumerate(results, 1):
            output_lines.append(
                f"【片段 {i}】来源：{r['source']} (相似度: {r['score']:.4f})\n{r['content']}"
            )
        return "\n\n".join(output_lines)