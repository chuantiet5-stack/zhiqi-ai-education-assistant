"""
FAISS 检索器

基于 FAISS 的高效向量检索，支持：
- 文档分块与向量化存储
- 相似度检索（余弦相似度）
- 阈值过滤与 Top-K 控制
- 持久化（索引 + 元数据）
- 目录批量导入
"""

import os
import pickle
import hashlib
import threading
from typing import List, Dict, Optional

import numpy as np
import faiss

from .embedding_client import SiliconFlowEmbedding


class FAISSRetriever:
    """
    FAISS 向量检索器

    使用 FlatIP 索引（内积），配合 L2 归一化实现余弦相似度检索。

    使用示例：
        embed_client = SiliconFlowEmbedding(api_key)
        retriever = FAISSRetriever(embed_client)
        retriever.add_documents([{"content": "...", "metadata": {"source": "math.md"}}])
        results = retriever.retrieve("一元二次方程", top_k=3, threshold=0.4)
    """

    def __init__(
        self,
        embedding_client: SiliconFlowEmbedding,
        dimension: int = 1024,
        index_path: str = "./workspace/faiss_index/index.faiss",
        metadata_path: str = "./workspace/faiss_index/metadata.pkl",
        default_threshold: float = 0.4,
    ):
        """
        Args:
            embedding_client: Embedding 客户端实例
            dimension: 向量维度（默认 1024，对应 BAAI/bge-m3）
            index_path: FAISS 索引文件路径
            metadata_path: 元数据文件路径
            default_threshold: 默认相似度阈值
        """
        self.embedding_client = embedding_client
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.default_top_k = 3    # 默认返回 3 个块
        self.default_threshold = default_threshold  # 默认相似度阈值

        # 线程锁：索引与元数据可能被多线程并发访问
        # （Web 管理页上传/删除、学伴对话检索、Agent 工具检索都经 asyncio.to_thread 并行执行），
        # FAISS 索引和 metadata 列表必须一起原子地修改，否则会错位或写坏持久化文件。
        # Embedding 网络请求在锁外执行，不会因慢请求阻塞他人检索。
        self._lock = threading.RLock()

        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self._load()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 归一化，使内积等于余弦相似度"""
        faiss.normalize_L2(vectors)
        return vectors

    def _save(self):
        """保存索引和元数据到磁盘"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):
        """从磁盘加载索引和元数据"""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[RAG] 加载 FAISS 索引，共 {len(self.metadata)} 个块")

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
        """
        中文文本分块

        Args:
            text: 原始文本
            chunk_size: 每块最大字符数
            overlap: 块之间的重叠字符数

        Returns:
            List[str]: 分块后的文本列表
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # 在标点处断开，保持语义完整
                for sep in ["。", "！", "？", "\n\n", "\n", "；", "，"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + 1
                        break
            chunks.append(text[start:end].strip())
            # 确保每次循环 start 至少前进 1，防止 end - overlap <= start
            # 时（分隔符靠近块开头）start 倒退导致无限循环 → MemoryError
            next_start = end - overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start
        return [c for c in chunks if c]

    def add_documents(self, documents: List[Dict], verbose: bool = True) -> int:
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每项格式为 {"content": "...", "metadata": {...}}
            verbose: 是否打印详细信息

        Returns:
            int: 新增的块数
        """
        chunks = []
        with self._lock:
            existing_ids = {m.get("id") for m in self.metadata}
            for doc in documents:
                content = doc.get("content", "")
                meta = doc.get("metadata", {})
                if not content:
                    continue

                blocks = self._chunk_text(content)
                for idx, block in enumerate(blocks):
                    chunk_id = hashlib.md5(block.encode()).hexdigest()[:16]
                    # 去重：跳过库中已有的块，同批次内重复的块也只保留第一份
                    if chunk_id in existing_ids:
                        continue
                    existing_ids.add(chunk_id)
                    chunks.append({
                        "id": chunk_id,
                        "content": block,
                        "chunk_idx": idx,
                        **meta,
                    })

        if not chunks:
            return 0

        # 向量化（网络请求，放在锁外执行，避免长文档 Embedding 期间阻塞他人检索）
        texts = [c["content"] for c in chunks]
        vectors = self.embedding_client.embed(texts)
        vectors_np = np.array(vectors).astype(np.float32)
        self._normalize(vectors_np)

        # 原子入库：Embedding 期间其他线程可能已入库相同内容，锁内二次去重防竞态
        with self._lock:
            existing_ids = {m.get("id") for m in self.metadata}
            new_chunks = [c for c in chunks if c["id"] not in existing_ids]
            if not new_chunks:
                return 0
            if len(new_chunks) < len(chunks):
                # 有竞态重复，向量也要同步截掉重复部分（chunks 与 vectors 一一对应）
                keep_pos = [i for i, c in enumerate(chunks) if c["id"] in {n["id"] for n in new_chunks}]
                vectors_np = np.ascontiguousarray(vectors_np[keep_pos], dtype=np.float32)

            # 添加到索引
            self.index.add(vectors_np)
            self.metadata.extend(new_chunks)
            self._save()

        added = len(new_chunks)
        if verbose:
            print(f"[RAG] 新增 {added} 个块，总块数: {len(self.metadata)}")
        return added

    def load_from_directory(self, dir_path: str, extensions: tuple = (".txt", ".md")) -> int:
        """
        递归加载目录下所有文本文件

        Args:
            dir_path: 目录路径
            extensions: 文件扩展名列表

        Returns:
            int: 新增块数
        """
        if not os.path.isdir(dir_path):
            print(f"[RAG] 目录不存在: {dir_path}")
            return 0

        documents = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(extensions):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        if content.strip():
                            rel_path = os.path.relpath(filepath, dir_path)
                            documents.append({
                                "content": content,
                                "metadata": {
                                    "source": rel_path,
                                    "file": file,
                                    "dir": os.path.basename(root),
                                }
                            })
                    except Exception as e:
                        print(f"[RAG] 读取文件失败 {filepath}: {e}")

        if documents:
            return self.add_documents(documents)
        return 0

    def retrieve(self, query: str, top_k: int = 3, threshold: Optional[float] = None) -> List[Dict]:
        """
        检索与查询最相关的文档块

        Args:
            query: 查询文本
            top_k: 返回的最大块数
            threshold: 相似度阈值，默认使用 default_threshold

        Returns:
            List[Dict]: 检索结果列表，每项含 content, source, score
        """
        if len(self.metadata) == 0:
            return []

        thresh = self.default_threshold if threshold is None else threshold

        # 查询向量化（网络请求，放在锁外执行）
        query_vec = self.embedding_client.embed([query])
        query_np = np.array(query_vec).astype(np.float32)
        self._normalize(query_np)

        # 锁内搜索：保证候选数计算、search 与 metadata 映射读到一致的快照
        with self._lock:
            candidates = min(top_k * 2, len(self.metadata))
            scores, indices = self.index.search(query_np, candidates)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                if score < thresh:
                    continue
                meta = self.metadata[idx]
                results.append({
                    "content": meta["content"],
                    "source": meta.get("source", "unknown"),
                    "score": float(score),
                })
                if len(results) >= top_k:
                    break
        return results

    def count(self) -> int:
        """返回总块数"""
        with self._lock:
            return len(self.metadata)

    def clear(self):
        """清空所有数据"""
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            self._save()

    def delete_source(self, source: str) -> int:
        """
        删除指定来源的所有块

        IndexFlat 支持向量重建（reconstruct_n），先取出全部向量再按
        保留项重组新索引，避免删除后重新调用 Embedding API。

        Args:
            source: 文档来源名称（与导入时的 metadata.source 精确匹配）

        Returns:
            int: 删除的块数（来源不存在时为 0）
        """
        if not self.metadata:
            return 0

        # 锁内完成"取向量→重组→替换"全过程，避免与并发 add/retrieve 错位
        with self._lock:
            keep_idx = [i for i, m in enumerate(self.metadata) if m.get("source") != source]
            removed = len(self.metadata) - len(keep_idx)
            if removed == 0:
                return 0

            new_index = faiss.IndexFlatIP(self.dimension)
            if keep_idx:
                all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                kept = np.ascontiguousarray(all_vectors[keep_idx], dtype=np.float32)
                new_index.add(kept)

            self.index = new_index
            self.metadata = [self.metadata[i] for i in keep_idx]
            self._save()
        return removed

    def get_all_sources(self) -> List[Dict]:
        """获取所有文档来源及块数统计"""
        with self._lock:
            sources = {}
            for meta in self.metadata:
                src = meta.get("source", "unknown")
                if src not in sources:
                    sources[src] = {"source": src, "chunk_count": 0}
                sources[src]["chunk_count"] += 1
            return list(sources.values())

    def get_source_preview(self, source: str) -> Dict:
        """
        获取指定来源的整篇预览文本（供管理页点击文档时展示）

        按 chunk_idx 排序后将各块拼接为整篇：分块时相邻块有 overlap 重叠，
        拼接时依据"块 i+1 的开头 = 块 i 的结尾"匹配去掉重复部分。

        Args:
            source: 文档来源名称（与导入时的 metadata.source 精确匹配）

        Returns:
            Dict: {"chunk_count": 块数, "full_text": 拼接后的整篇文本}；来源不存在时块数为 0
        """
        with self._lock:
            chunks = [
                {"chunk_idx": m.get("chunk_idx", 0), "content": m.get("content", "")}
                for m in self.metadata
                if m.get("source") == source
            ]
        chunks.sort(key=lambda c: c["chunk_idx"])
        return {"chunk_count": len(chunks), "full_text": self._stitch_chunks(chunks)}

    @staticmethod
    def _stitch_chunks(chunks: List[Dict]) -> str:
        """
        将按序内容块拼接为整篇文本，去除相邻块的重叠部分

        分块策略是"下一块起点 = 上一块终点 - overlap"，因此优先按后缀/前缀
        匹配合并；匹配不上（如分块时被 strip 过或碎片化分块）则用换行连接，
        保证预览不丢内容。
        """
        parts = [c["content"] for c in chunks if c["content"]]
        if not parts:
            return ""
        out = parts[0]
        for nxt in parts[1:]:
            # 爬行分块的尾部碎片整块是已拼文本的后缀 → 内容已在，跳过
            if out.endswith(nxt):
                continue
            # 整块是已拼文本的前缀扩展（碎片化时可能出现）→ 直接扩展
            if nxt.startswith(out):
                out = nxt
                continue
            merged = False
            # 真实重叠不可能超过分块参数 overlap=30（与 _chunk_text 默认一致）；
            # 上界压到 30 防止周期性文本出现"更长的虚假匹配"而误删内容
            max_ov = min(len(out), len(nxt), 30)
            # 重叠通常为 overlap(30) 字符；从长到短试探，小于 8 字符的巧合
            # 匹配不可信，宁可换行连接也不误删内容
            for ov in range(max_ov, 8, -1):
                if out[-ov:] == nxt[:ov]:
                    out += nxt[ov:]
                    merged = True
                    break
            if not merged:
                out += "\n" + nxt
        return out