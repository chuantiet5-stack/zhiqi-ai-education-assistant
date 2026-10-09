"""RAG 检索包

提供基于 FAISS + 硅基流动 Embedding 的教材知识库检索能力：
- embedding_client: 硅基流动向量化客户端（BAAI/bge-m3，1024 维）
- retriever: FAISS 向量检索器（分块/入库/相似度检索/持久化）
"""
