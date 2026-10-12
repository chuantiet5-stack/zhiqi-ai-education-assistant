# Zhiqi — 知契智能教育助手

结合教育 Agent、教材检索和本地视觉分析的学习辅助应用，提供学习对话、作业批改、学习计划和自习陪学功能。

## 功能

- 学习问答、多模态作业批改与学习计划。
- 教材解析、Embedding 和 FAISS 检索。
- B 站教学视频搜索与古诗词 MCP 工具。
- 基于 YOLO Pose 的坐姿和专注状态分析。
- 学习记录、持久会话、家长绑定与学习报告。
- Web 学习工作台、知识库管理和飞书渠道。

## 技术栈

Python · FastAPI · SQLite · FAISS · Ultralytics · OpenCV · OpenAI 兼容接口

## 项目结构

```text
zhiqi-ai-education-assistant/
├── src/
│   └── nanoclaw/
│       ├── agent/          # 教育 Agent、RAG 与工具
│       ├── channels/       # Web 页面与消息渠道
│       ├── db/             # 数据持久层
│       ├── models/         # 本地姿态模型
│       ├── config.example.json
│       └── main.py
├── docs/                  # 配置与开发文档
├── requirements.txt       # 项目依赖入口
├── run.py                 # 服务启动入口
├── .editorconfig
├── .gitignore
└── README.md
```

## 快速开始

```bash
git clone https://github.com/chuantiet5-stack/zhiqi-ai-education-assistant.git
cd zhiqi-ai-education-assistant
python -m venv .venv
```

激活虚拟环境（[操作说明](docs/development.md#开发环境)），然后安装依赖并创建配置：

```bash
python -m pip install -r requirements.txt
python run.py web --init-config
```

编辑 `src/nanoclaw/config.json`，填写模型接口密钥并确认文本与视觉模型配置。启动服务：

```bash
python run.py web
```

| 页面 | 地址 |
| --- | --- |
| 注册与登录 | <http://127.0.0.1:8080> |
| 学习工作台 | <http://127.0.0.1:8080/edu> |
| 教材知识库管理 | <http://127.0.0.1:8080/rag_admin> |

首次使用需创建本地账号并导入自己的教材。专注监控需要服务所在设备连接摄像头。

## 开发验证

```bash
python -m compileall -q src run.py
```

当前项目没有独立的自动化测试套件；业务和外部服务验证流程见[开发指南](docs/development.md)。

## 文档

- [配置与教材接入](docs/configuration.md)
- [开发与验证](docs/development.md)

真实凭证、账号、教材索引、上传内容和会话数据在本地生成，由 `.gitignore` 排除。
