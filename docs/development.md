# 开发与验证

## 开发环境

进入对应应用目录后，创建独立虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

安装该应用目录中的依赖文件后再运行。配置、数据库与会话均在本地生成。

## 依赖与能力

`src/nanoclaw/requirements.txt` 列出 Web、模型、RAG、数据库及视觉依赖。专注监控还需要可访问的本地摄像头；外部模型、Embedding、视频搜索与飞书依赖对应服务可用。

## 代码阅读入口

| 路径 | 职责 |
| --- | --- |
| `src/nanoclaw/main.py` | 模型、工具、教育服务与渠道组装 |
| `src/nanoclaw/edu_service.py` | 学习对话、作业与计划业务 |
| `src/nanoclaw/agent/edu_loop.py` | 教育 Agent 执行循环 |
| `src/nanoclaw/agent/rag/` | 教材解析、Embedding 与检索 |
| `src/nanoclaw/agent/tools/yolo_vision.py` | 姿态关键点与视觉工具 |
| `src/nanoclaw/focus_monitor.py` | 采样、窗口统计与提醒 |
| `src/nanoclaw/parent_report.py` | 家长绑定、摘要与推送 |
| `src/nanoclaw/session/persistent_memory.py` | SQLite 会话记忆 |

## 语法检查

在仓库根目录执行：

```bash
python -m compileall -q src run.py
```

该命令只检查 Python 源码编译，不代表联网功能或业务流程已经通过。当前仓库没有独立的 `tests/` 测试套件。

## 功能验证流程

1. 启动服务并注册本地账号，检查 `/edu` 的学习工作台。
2. 提交学习问题与作业图片，验证文本及视觉模型接口。
3. 在 `/rag_admin` 上传测试教材，再针对教材内容提问，检查检索片段。
4. 接入摄像头后验证专注监控，分别检查无摄像头和依赖缺失时的降级提示。
5. 配置家长绑定和飞书后，检查学习摘要与报告推送。

后续自动化测试可优先覆盖文档分块、检索结果过滤、会话归属和报告材料组装。涉及外部服务的验证使用独立测试账号与测试数据。

