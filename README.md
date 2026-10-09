# 知契 · 智能教育助手

基于 NanoClaw Agent 的智能教育助手，包含教育对话、教材 RAG 检索、教学视频与古诗词工具、专注监控和家长报告等模块。

## 功能与技术栈

- 教育 Agent 与多模态模型接口支持学习对话、作业批改和学习计划。
- 教材解析与 FAISS 向量检索为学习问答提供 RAG 上下文。
- 集成教学视频搜索与古诗词 MCP 工具。
- YOLO 姿态推理用于专注监控，并提供家长报告模块。
- 技术栈：Python、NanoClaw Agent、FastAPI、OpenAI 兼容模型接口、FAISS、MCP、SQLite、YOLO/OpenCV。

## 项目结构

- `nanoclaw/agent/`：Agent、教育工具与 RAG 检索。
- `nanoclaw/channels/`：Web 与飞书渠道及页面资源。
- `nanoclaw/mcp_servers/`：古诗词 MCP 服务。
- `nanoclaw/models/`：交付版本随附的 YOLO 姿态模型。
- `nanoclaw/db/`：本地 SQLite 持久层。

## 本地启动

在 Python 虚拟环境中执行：

```powershell
cd nanoclaw
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json
# 编辑 config.json，填入自己的模型 API 密钥；飞书凭证按需配置。
python main.py
```

Web 默认端口为 `8080`。配置示例中的 MCP 启动命令使用 `python`，请确保它指向当前虚拟环境。教材索引和用户数据库未随公开源码发布，需在本地运行时创建或导入。

MCP 接入说明见 `nanoclaw/MCP_INTEGRATION.md`。

## 发布说明

公开版本仅保留源码和运行所需资源。原始 API 密钥、飞书凭证、用户数据库、聊天记录、上传图片、教材索引、日志和缓存均未提交；视频、PPT 和原始压缩包不上传。本地 `config.json` 使用 `config.example.json` 创建。

本次整理验证了发布文件及 Python 语法，未安装完整依赖或执行联网业务测试。
