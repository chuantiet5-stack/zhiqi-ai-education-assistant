"""
硅基流动 Embedding 客户端

提供文本向量化能力，使用硅基流动的 BAAI/bge-m3 模型（1024维）。
"""

import time

import requests
from typing import List

# 网络瞬时故障（DNS 抖动 / 连接超时 / 限流）自动重试：
# 共尝试 3 次，间隔 2s、4s 递增
EMBED_RETRY_ATTEMPTS = 3
EMBED_RETRY_BACKOFF = 2

# 单次请求最大文本条数：超出会被 Embedding API 拒绝（400/413），
# 因此长文档入库时自动按此上限分批请求
EMBED_MAX_BATCH = 32


class SiliconFlowEmbedding:
    """
    硅基流动 Embedding API 客户端（同步）

    使用示例：
        client = SiliconFlowEmbedding(api_key="sk-xxx")
        vectors = client.embed(["文本1", "文本2"])
    """

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        """
        Args:
            api_key: 硅基流动 API Key
            base_url: API 基础地址，默认硅基流动官方地址
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = "BAAI/bge-m3"  # 1024 维

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本向量（自动按 EMBED_MAX_BATCH 分批请求）

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表（顺序与输入一致），每个向量维度为 1024

        Raises:
            Exception: API 调用失败时抛出
        """
        if not texts:
            return []
        if len(texts) <= EMBED_MAX_BATCH:
            return self._embed_once(texts)

        all_vectors: List[List[float]] = []
        total_batches = (len(texts) + EMBED_MAX_BATCH - 1) // EMBED_MAX_BATCH
        for i in range(0, len(texts), EMBED_MAX_BATCH):
            batch_no = i // EMBED_MAX_BATCH + 1
            print(f"[RAG] Embedding 分批请求 {batch_no}/{total_batches}（本批 {len(texts[i:i + EMBED_MAX_BATCH])} 条）")
            all_vectors.extend(self._embed_once(texts[i:i + EMBED_MAX_BATCH]))
        return all_vectors

    def _embed_once(self, texts: List[str]) -> List[List[float]]:
        """单次请求获取向量（带重试），texts 不能超过 EMBED_MAX_BATCH 条"""
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        last_error: Exception | None = None
        for attempt in range(1, EMBED_RETRY_ATTEMPTS + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
            except requests.exceptions.RequestException as e:
                # 网络层故障（DNS 解析失败 / 连接超时 / 连接复位）：可重试
                last_error = e
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 限流 / 服务端故障：可重试
                    last_error = Exception(f"Embedding API 错误 {resp.status_code}: {resp.text[:200]}")
                else:
                    # 请求本身有问题（鉴权失败 / 参数错误）：重试无意义，直接失败
                    raise Exception(f"Embedding API 错误 {resp.status_code}: {resp.text[:200]}")
            if attempt < EMBED_RETRY_ATTEMPTS:
                wait = EMBED_RETRY_BACKOFF * attempt
                print(f"[RAG] Embedding 请求失败（第 {attempt}/{EMBED_RETRY_ATTEMPTS} 次）: {last_error}，{wait} 秒后重试")
                time.sleep(wait)
        raise Exception(f"Embedding 请求失败（已重试 {EMBED_RETRY_ATTEMPTS - 1} 次）: {last_error}")
