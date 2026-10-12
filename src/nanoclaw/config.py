"""
NanoClaw 配置模块

提供配置加载和管理功能，支持：
- 从 JSON 文件读取配置
- 环境变量覆盖（NANOCLAW_API_KEY）
- 默认值填充

配置文件使用 config.json，包含 API 密钥、模型、工作区等参数。
"""

import json
import os
from dataclasses import dataclass


@dataclass
class NanoClawConfig:
    """
    NanoClaw 配置结构

    包含 Agent 运行所需的所有配置参数。

    Attributes:
        api_key: API 密钥，优先从环境变量 NANOCLAW_API_KEY 读取
        base_url: API 基础 URL，默认硅基流动地址
        model: 使用的模型名称，默认 Kimi-K2.5
        models: 多模型配置字典，键为角色（main/subagent/cheap），值为模型名，
            让子 Agent 可使用更便宜的模型
        workspace: 工作区路径，默认当前目录
        max_iterations: Agent 最大迭代次数，防止无限循环
        identity_file: 人设文件名，相对于 workspace
        feishu_app_id / feishu_app_secret: 飞书开放平台应用凭证，二者齐全才启用飞书渠道
        web_enabled: 是否启用内置 Web 渠道（无需外部凭证）
        web_host / web_port: Web 服务监听地址与端口
        mcp_servers: MCP Server 配置，键为服务名，值为 {"command": ..., "args": [...]}
    """

    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "deepseek-ai/DeepSeek-OCR"
    models: dict = None  # {"main": "...", "subagent": "...", "cheap": "..."}
    workspace: str = "."
    max_iterations: int = 32
    identity_file: str = "identity.md"
    # 飞书配置
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    # Web 配置
    web_enabled: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    # RAG 配置（向量依赖缺失或初始化失败时自动降级禁用）
    rag_enabled: bool = True
    rag_threshold: float = 0.4
    rag_top_k: int = 3
    rag_index_path: str = "./workspace/faiss_index"
    rag_knowledge_base: str = "./knowledge_base"  # 自动加载的教材目录
    # 语音对话配置（学伴语音输入 / 回复朗读）# 语音配置
    stt_model: str = "FunAudioLLM/SenseVoiceSmall"  # 语音转文字模型
    tts_model: str = "FunAudioLLM/CosyVoice2-0.5B"  # 文字转语音模型
    tts_voice: str = "anna"  # 默认音色（CosyVoice2 预置音色名）
    # 家长报告配置（每日定时推送学习总结的时刻，本地时间）
    parent_push_hour: int = 20
    parent_push_minute: int = 0
    # MCP Server 配置
    mcp_servers: dict = None  # {"server_name": {"command": "...", "args": [...]}}


def load_config(config_path: str = "config.json") -> NanoClawConfig:
    """
    加载配置

    从指定路径读取 JSON 配置文件，填充 NanoClawConfig 字段。
    环境变量 NANOCLAW_API_KEY 具有最高优先级，会覆盖配置文件中的值。

    Args:
        config_path: 配置文件路径，默认 config.json

    Returns:
        NanoClawConfig: 配置对象

    注意：
        - 配置文件不存在时返回默认配置
        - 环境变量 NANOCLAW_API_KEY 优先级最高
    """
    # 默认配置
    config = NanoClawConfig()

    # 从 JSON 文件读取（如果存在）
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 更新配置字段（仅更新存在的字段）
            if "api_key" in data:
                config.api_key = data["api_key"]
            if "base_url" in data:
                config.base_url = data["base_url"]
            if "model" in data:
                config.model = data["model"]
            if "workspace" in data:
                config.workspace = data["workspace"]
            if "max_iterations" in data:
                config.max_iterations = data["max_iterations"]
            if "identity_file" in data:
                config.identity_file = data["identity_file"]
            # 加载多模型配置
            if "models" in data:
                config.models = data["models"]
                # 如果没有单独的 model 字段，用 models.main
                if "model" not in data and "main" in config.models:
                    config.model = config.models["main"]
            # 加载飞书配置
            feishu = data.get("feishu", {})
            if feishu.get("app_id"):
                config.feishu_app_id = feishu["app_id"]
            if feishu.get("app_secret"):
                config.feishu_app_secret = feishu["app_secret"]
            # 加载 Web 配置
            web = data.get("web", {})
            if "enabled" in web:
                config.web_enabled = web["enabled"]
            if web.get("host"):
                config.web_host = web["host"]
            if web.get("port"):
                config.web_port = web["port"]
            # 加载 RAG 配置
            rag = data.get("rag", {})
            if "enabled" in rag:
                config.rag_enabled = rag["enabled"]
            if rag.get("threshold") is not None:
                config.rag_threshold = rag["threshold"]
            if rag.get("top_k"):
                config.rag_top_k = rag["top_k"]
            if rag.get("index_path"):
                config.rag_index_path = rag["index_path"]
            if rag.get("knowledge_base"):
                config.rag_knowledge_base = rag["knowledge_base"]
            # 加载语音对话配置
            voice = data.get("voice", {})
            if voice.get("stt_model"):
                config.stt_model = voice["stt_model"]
            if voice.get("tts_model"):
                config.tts_model = voice["tts_model"]
            if voice.get("tts_voice"):
                config.tts_voice = voice["tts_voice"]
            # 加载家长报告配置
            parent_report = data.get("parent_report", {})
            if "push_hour" in parent_report:
                config.parent_push_hour = int(parent_report["push_hour"])
            if "push_minute" in parent_report:
                config.parent_push_minute = int(parent_report["push_minute"])
            # 加载 MCP Server 配置
            if "mcp_servers" in data:
                config.mcp_servers = data["mcp_servers"]

        except json.JSONDecodeError:
            # JSON 解析错误，使用默认配置：
            # 宁可降级为默认值运行，也不让配置文件损坏导致程序完全无法启动
            pass
        except Exception:
            # 其他错误（如权限、编码问题）同样降级处理，保证主流程可用
            pass

    # 环境变量优先级最高
    env_api_key = os.environ.get("NANOCLAW_API_KEY")
    if env_api_key:
        config.api_key = env_api_key

    return config