# 配置指南

## 模型接口

从 `src/nanoclaw/config.example.json` 创建同目录的 `config.json`。

| 字段 | 用途 |
| --- | --- |
| `api_key` | 模型服务密钥，可由 `NANOCLAW_API_KEY` 环境变量覆盖 |
| `base_url` | OpenAI 兼容模型接口地址 |
| `models.main` | 主对话模型 |
| `models.subagent / cheap` | 子任务与辅助任务模型 |
| `models.vision` | 图片理解与多模态任务模型 |
| `web.enabled / host / port` | Web 服务设置，默认示例端口为 `8080` |

模型名称需要与实际服务支持的模型一致。语音接口由教育服务调用模型服务的音频接口，对应能力也需由服务端支持。

## 教材检索

| 字段 | 用途 |
| --- | --- |
| `rag.enabled` | 是否启用教材检索 |
| `rag.threshold` | 相似度过滤阈值 |
| `rag.top_k` | 返回的教材片段数量 |
| `rag.index_path` | 本地向量索引目录 |

在 `http://127.0.0.1:8080/rag_admin` 导入自己的教材。文档经过解析、分块和向量化后进入 FAISS 索引；没有导入教材时，不应将普通模型回答当作教材检索结果。

教材、索引、上传文件及会话记录存放在本地工作目录，不提交到仓库。

## MCP 与飞书

`mcp_servers.poetry` 配置古诗词工具，示例通过 `python mcp_servers/poetry_server.py` 启动。运行环境需要安装 MCP 依赖，子进程的 `python` 应指向同一个虚拟环境。

飞书渠道需要配置 `feishu.app_id` 与 `feishu.app_secret`。未配置时保持为空；Web 学习工作台可独立于飞书使用。

## 本地视觉与学习记录

- 专注监控使用运行服务设备的摄像头，需要可用的 OpenCV 与 Ultralytics 环境。
- 本地模型位于 `src/nanoclaw/models/yolov8n-pose.pt`。
- 关键点分析反映坐姿及注意力的启发式判断，需要结合摄像头位置与实际场景解释。
- SQLite 保存本地用户、会话与学习记录；首次使用需注册账号。
- 家长报告模块依赖本地账号绑定与学习记录，飞书推送需要相应渠道配置。
