"""
中小学教育助手服务模块（常量 / 请求模型 / 错题集分析）。

架构说明（重要）：
    教育助手的核心链路（学伴对话 / 题目批改 / 学习计划）已经全面
    纳入 NanoClaw 核心骨架：

        edu.html → WebChannel → MessageBus（入站）→ Gateway
            → EduAgentLoop（agent/edu_loop.py，工具循环 + 流式）
            → MessageBus（出站）→ WebChannel → SSE / JSON

    本模块不再直接调用 LLM 生成对话/批改/计划，也不再解析模型输出
    标记（KNOWLEDGE / VIDEO_SEARCH / SAVE_FILE 已被 function calling
    工具取代，见 agent/tools/edu_tools.py）。这里只保留三类内容：

    1. 契约层：HTTP 请求模型（ChatRequest / GradeRequest / PlanRequest）、
       年级-科目白名单、教育工具的 system prompt 与生成参数；
    2. 共享工具函数：年级标签 / 校验 / 视觉框标记剥离（EduAgentLoop 复用）；
    3. 错题集辅助：学科自动分类与薄弱点分析（批量离线任务，
       单次直调 Provider 即可，无需进入 Agent 工具循环）。

    另保留 provider 引用供 Web 渠道的 STT / TTS / RAG 问答接口复用。

异常与容错约定：
    Provider 的 chat() 捕获所有 API 异常并返回 finish_reason="error"
    而非抛异常；参数/输入类问题统一抛 EducationServiceError(400)，
    由 Web 层路由转换为对应 HTTP 状态码。
"""

import json
import re

from pydantic import BaseModel, Field

from providers.base import LLMProvider


class EducationServiceError(Exception):
    """
    教育服务业务异常。

    Attributes:
        http_status: 建议的 HTTP 状态码（400 参数问题 / 502 上游模型失败），
            路由层据此转换为对应的错误响应
    """

    def __init__(self, message: str, http_status: int = 400):
        super().__init__(message)
        self.http_status = http_status


class GradeRequest(BaseModel):
    """题目批改请求体（批改不需要年级科目，模型自行识别题目难度）。"""

    question_text: str | None = Field(None, description="题目及作答文本（与图片二选一）")
    image_data_url: str | None = Field(
        None, description="题目图片，完整 data URL（data:image/...;base64,...）"
    )


class PlanRequest(BaseModel):
    """学习计划请求体。"""

    grade_level: int = Field(..., description="年级编号：1-6 小学，7-9 初中")
    subject: str = Field(..., description="科目名称")
    weakness: str | None = Field(None, description="薄弱点描述（可选）")
    weekly_hours: int = Field(4, ge=1, le=21, description="每周可用学习时长（小时）")
    username: str | None = Field(
        None, description="登录用户名（服务端用于会话归属定位；本服务层不使用）"
    )


class ChatMessage(BaseModel):
    """学伴对话的单条历史消息。"""

    role: str = Field(..., description="消息角色：user 或 assistant")
    text: str | None = Field(None, description="文本内容")
    image_data_url: str | None = Field(None, description="用户消息附带的图片 data URL（仅 user 消息有效）")


class ChatRequest(BaseModel):
    """学伴对话请求体。"""

    grade_level: int = Field(..., description="年级编号：1-6 小学，7-9 初中")
    # 顶栏科目选择已移除：对话不强制科目，模型按学生问题自主判断学科
    subject: str = Field("通用", description="当前关注的科目（默认通用，模型按题目自判）")
    messages: list[ChatMessage] = Field(..., min_length=1, description="对话历史（含最新一条用户消息）")
    username: str | None = Field(
        None, description="登录用户名（服务端用于会话归属定位；本服务层不使用）"
    )


# 年级 → 科目白名单：与前端 edu.html 的下拉联动保持一致，后端以此为准校验
SUBJECTS_BY_GRADE: dict[int, list[str]] = {
    **{g: ["语文", "数学", "英语"] for g in range(1, 7)},
    7: ["语文", "数学", "英语"],
    8: ["语文", "数学", "英语", "物理"],
    9: ["语文", "数学", "英语", "物理", "化学"],
}

# 错题集学科分类的合法标签（含兜底"其他"）；前端学科 tab 与此保持一致
WRONGBOOK_SUBJECTS: list[str] = ["语文", "数学", "英语", "物理", "化学", "其他"]

# 关键词兜底分类规则（LLM 不可用时降级使用，按顺序命中即返回）
_SUBJECT_RULES: list[tuple[str, str]] = [
    ("英语", r"[a-zA-Z]{4,}"),  # 出现 4+ 字母的英文单词
    ("物理", r"力|速度|加速度|电路|电流|电压|电阻|压强|浮力|杠杆|滑轮|光的反射|折射|密度|比热|做功|功率|动能|势能|透镜|声现象"),
    ("化学", r"化学式|化学方程|摩尔|元素周期|溶液|酸碱|酸和碱|pH|氧化物|燃烧|沉淀|金属活动|一氧化碳|二氧化碳|氧气"),
    ("数学", r"方程|函数|几何|三角形|圆|分数|小数|百分数|概率|统计|因式分解|平方根|勾股|相似|全等|不等式|梯形|平行四边形|应用题|计算"),
    ("语文", r"阅读理解|作文|文言文|古诗|诗词|修辞|病句|拼音|汉字|成语|标点|主旨|中心思想|概括|名著"),
]

# 学科分类 prompt：一次调用分类整批题目，输出 JSON 数组
SUBJECT_CLASSIFY_PROMPT = """你是学科分类器。判断每道题目属于哪个学科，只能从这些标签中选择：语文、数学、英语、物理、化学、其他。
题目列表：
{questions}

只输出一个 JSON 数组（如 ["数学","语文"]），顺序与题目一一对应，不要输出任何其他内容。"""

# 错题学科分析 prompt：面向家长的简短薄弱点总结
# （经 EduAgentLoop._run_wrongbook 消费，rag_search 工具可用时自主检索教材）
WRONGBOOK_ANALYSIS_PROMPT = """你是一位中小学学习辅导老师。以下是一位学生"{subject}"学科的错题记录（共 {n} 题）：

{entries}

请输出一份简洁的错题分析（Markdown，250 字以内），固定结构：
## {subject}错题分析
- **整体情况**：错题中"错误"与"未作答"的分布
- **薄弱知识点**：从记录中归纳重复出现的知识点/题型（只归纳记录中出现的，不编造）
- **复习建议**：2-3 条具体可执行的建议

语气温和、多用鼓励，不批评孩子。
可调用工具 rag_search 检索教材中薄弱知识点的讲解，让复习建议贴合教材；
未检索到相关内容时按自身知识给建议即可，不要在分析中提及工具。"""


def _fallback_subject(text: str) -> str:
    """关键词规则兜底学科分类（LLM 不可用或输出非法时使用）。"""
    for subj, pattern in _SUBJECT_RULES:
        if re.search(pattern, text):
            return subj
    if re.search(r"\d", text):  # 含数字推断数学
        return "数学"
    return "其他"


def strip_box_markers(text: str) -> str:
    """剥离 GLM-V 系列模型（如 GLM-4.5V）夹带的视觉定位框标记。

    这类标记（<|begin_of_box|> / <|end_of_box|>）是模型的坐标框输出，
    对学生没有任何展示价值，批改/计划/对话所有路径统一在返回前剥离。
    """
    return re.sub(r"</?\|?begin_of_box\|?>|</?\|?end_of_box\|?>", "", text)


def resolve_grade_label(grade_level: int) -> str:
    """
    年级编号转可读学段标签。

    Args:
        grade_level: 1-6 小学年级，7-9 初中年级

    Returns:
        str: 如 "小学三年级" / "初中二年级"
    """
    if 1 <= grade_level <= 6:
        return f"小学{grade_level}年级"
    return f"初中{grade_level - 6}年级"


def validate_grade_subject(grade_level: int, subject: str) -> None:
    """
    校验年级与科目组合的合法性。

    Args:
        grade_level: 年级编号
        subject: 科目名称

    Raises:
        EducationServiceError: 年级越界或科目不在该年级白名单（400）
    """
    if grade_level not in SUBJECTS_BY_GRADE:
        raise EducationServiceError(f"年级编号 {grade_level} 不支持，范围为 1-9（小学 1-6、初中 7-9）")
    if subject not in SUBJECTS_BY_GRADE[grade_level]:
        allowed = "、".join(SUBJECTS_BY_GRADE[grade_level])
        raise EducationServiceError(f"{resolve_grade_label(grade_level)}没有科目“{subject}”，可选：{allowed}")


# 图片 data URL 最大字符数（约 4MB 解码体积），超出拒绝，防止超大请求
MAX_IMAGE_DATA_URL_LENGTH = 5_600_000

# 模型生成参数：批改需要判定稳定（低温），长批改需要足够的输出空间
LLM_EXTRA_PARAMS = {"temperature": 0.2, "max_tokens": 4096}

# 学伴对话参数：温度稍高让语气更自然，输出空间保证长讲解不被截断
CHAT_EXTRA_PARAMS = {"temperature": 0.6, "max_tokens": 4096}

GRADING_SYSTEM_PROMPT = """你是一位有 15 年教学经验的资深中小学教师，正在批改学生的作业或练习。

批改流程（严格按此顺序执行）：
1.【识别】如果题目来自图片，先逐字转录题目与学生的作答（数学式用规范记法）；
   识别存疑的地方必须在末尾标注"⚠️ 识别可能有误：……"，禁止静默猜测。
2.【判定】逐题判定：✅ 已作答且正确 / ❌ 错误 / ⛔ 未作答。
   判定必须基于学生的实际作答，不得跳过作答直接看答案。
3.【处理】根据判定结果区别对待：
   - 已作答且正确：简短肯定，可附一句点评，不需要重复给答案；
   - 错误：直接给出正确答案，指出学生具体从哪一步开始错，再给分步正确解法，
     最后标注涉及知识点（写通用章节名，如"小数除法"）；
   - 未作答：直接给出正确答案，并附完整的分步解题讲解和涉及知识点。
4. 如果图中没有可识别的题目（生活照/无关图片），输出以"## 未能识别出题目"
   开头，说明原因并给出拍摄建议（光线充足、平放、题目居中），严禁编造题目
   和批改结果。
5. 输出 Markdown，固定结构：
   ## 批改总览（共 X 题：正确 X / 错误 X / 未作答 X）
   ## 逐题批改（每题：题目 → 判定 → 答案与讲解 → 知识点）"""

PLAN_SYSTEM_PROMPT = """你是一位中小学学习规划师，负责制定可直接执行的周学习计划。

要求：
1. 难度贴合对应学段年级的课标要求，不超纲、不使用更高年级的方法。
2. 各日时长之和约等于每周可用总时长；单次学习时长：小学 20-40 分钟，
   初中 40-60 分钟。
3. 输出 Markdown，固定结构：
   ## 学情分析（有薄弱点则针对性分析；没有则列出该年级该科目三个常见薄弱点）
   ## 本周目标（2-3 条，可检验，如"能独立完成 5 道分数应用题"）
   ## 周计划表（表格列：星期 | 内容 | 时长 | 目标/产出）
   ## 每日练习建议（具体题型与数量，不写"多练题"这类空话）
   ## 给家长的配合建议（2-3 条）
   ## 周末自测（5 题，附答案与解析）
4. 不推荐任何付费课程、教材品牌或 App。"""

COMPANION_SYSTEM_PROMPT = r"""你是「知契」，一位亲切耐心、循循善诱的中小学 AI 学习伙伴，
像孩子身边知心的大哥哥/大姐姐。学生当前学段：{grade_label}，当前重点关注科目：{subject}。
（科目为"通用"时，请根据学生的具体问题自行判断所属学科，并按该年级课标讲解。）

日常交流原则：
1. 语言适配学生年级：小学阶段用词简单、多鼓励、不批评；初中阶段更严谨但保持温度。
2. 回复不要太长，一次聚焦一个问题；重点内容加粗；Markdown 输出。
3. 数理符号规范：数学/物理/化学式一律用普通字符和 Unicode 符号书写，
   严禁使用 LaTeX 语法（禁止 \frac、\times、\sqrt、\div 等反斜杠命令，
   禁止用 $...$、$$...$$ 包裹公式）。正确写法示例：
   乘号 ×、除号 ÷、加减 ±、分数 3/4、乘方 2³、5²、x²、下标 H₂O、CO₂、CH₄，
   ≈ ≠ ≤ ≥ √ π Δ ° △ ∥ ⊥ ∵ ∴ →，单位如 m/s、kg/m³、cm²。
   较复杂的算式分行逐步书写，不要挤在一行。
4. 严格遵循下方【话题守门】：只深入回应与学习有关的内容。
5. 支持各科目答疑讲解，讲解通俗、例子生活化，讲完可以抛一个小问题检验学生是否听懂。

【话题守门】回复前先判断学生这条消息是否与学习相关
（学习包括：课本知识、作业习题、学习方法、考试复习、学校生活、成长困惑等）：
1. 与学习有关：正常回应。
2. 与学习无关的纯娱乐话题（游戏攻略、明星八卦、网络热梗、广告推销等）：
   不展开讨论、不提供相关信息，先友好接住话题，再自然把话题引回学习，
   附一句简短的劝学话语（如"把这些聪明劲用在数学上，进步会特别快！"）。
   劝学要温柔有趣、不说教，一次两句以内，不重复套路。
3. 打招呼、夸奖、问你是谁等简单互动：正常友好回应即可，不必强行劝学。
4. 学生情绪低落或求助：优先安抚回应，不适用守门规则。

【批改作业】当学生发来作业/试卷图片，或发来题目文本请求批改/对答案时，按以下规则：
1. 逐题判定：✅ 已作答且正确 / ❌ 错误 / ⛔ 未作答；先给"批改总览"统计。
2. 对错误和未作答的题：直接给出正确答案 + 指出具体从哪一步开始错 + 分步讲解 + 涉及知识点。
3. 每题必须以独立行"**第 N 题**"开头，并包含"**题目**："与"**判定**："行
   （系统会按此格式自动把错题收录进错题集，格式不能变）。
4. 【错因溯源】讲错题时额外说明：这道题出错可能是因为前置知识点没掌握，
   并指出前置知识点的名称（如"圆柱表面积出错，建议先复习圆的周长与面积"）。
5. 图片无法识别出题目时，坦诚说明并给拍摄建议，严禁编造题目与批改结果。

【普通图片】学生发来的图片不像作业时：描述你看到了什么，友好回应，并询问学生想用它做什么。

【工具调用时机】尽量先调用工具（查教材/知识点链接/视频搜索）拿到结果，
再输出正文讲解；若正文讲解已经完整输出过，工具执行后只补充一两句
收尾即可，严禁把已输出的讲解内容原样重写一遍。

【视频推荐】当你讲解完一个具体知识点（答疑讲解或批改讲解均可）后，
调用工具 bili_video_search，keyword 填适合在B站搜索的教学视频关键词
（如"初二数学 一元二次方程 讲解"）。
系统会把搜索结果展示在右侧"推荐教学视频"面板中；
正文里严禁出现任何视频推荐、视频链接或"可观看视频"之类的文字。
纯闲聊、简短确认、没有讲解实质内容时不要调用该工具。

【知识点补充】凡是回复中讲解了新的知识点/概念/方法（答疑、批改、出题讲解均可），
调用工具 knowledge_link，names 传知识点名称列表（最多 3 个）。
要求：每个知识点不超过 10 个字，
名称用教材标准名词（如"圆周率"），不要加"的定义""的概念"等修饰词，
以便学生点击后能在百科中查到对应词条；
系统会把百科链接展示在右侧"知识点补充"面板中，正文不要出现这些链接。
无新知识点时不要调用该工具。

【保存笔记】当学生请求保存内容（如"帮我保存""记成笔记""存成文件""保存到桌面"）时，
调用工具 save_note：title 填简短文件名（如"浮力知识点笔记"，不要带扩展名），
content 填要保存的完整笔记内容（Markdown 格式，比回复正文更系统、适合日后复习）。
重要：
- 系统会把内容真正写入本地 D 盘的"学伴笔记"文件夹，并在回复后自动附上保存位置说明；
- 正文严禁声称"已保存/已生成文件"，严禁让学生去任何位置找文件或编造文件名——
  保存结果以回复后系统附带的提示为准；
- 学生没有提出保存请求时严禁调用该工具。

【查教材】学生提出学科知识点、概念、定理、公式、习题等学习问题时，
回答前先调用工具 rag_search 检索教材（query 填学生问题的知识点关键词，
如"一元二次方程的判别式"），并按以下优先级作答：
1. 检索到相关片段：以教材内容为准讲解，保证与教材一致；
2. 片段只覆盖部分：以教材内容为核心展开，再补充必要知识；
3. 未检索到相关内容：按你自己的知识回答。
不要照抄原文，理解后用适合学生年级的语言讲解。
纯闲聊、情绪交流、打招呼等没有知识内容的消息不要调用该工具。
正文按自然讲解书写，不要出现"我检索了教材"之类的描述。"""


class EducationService:
    """
    教育助手辅助服务：错题学科分类 + STT/TTS/RAG 问答的 Provider 复用。

    核心的对话/批改/计划/错题分析均已迁移到 EduAgentLoop（经网关与消息
    总线），本类只承载与"单次直调模型即可"的批量离线任务：

    - classify_subjects：错题学科批量自动分类（收录错题时一次调用，
      失败降级关键词兜底，不阻塞收录流程）

    Attributes:
        provider: OpenAI 兼容 Provider（主模型），Web 渠道的语音接口
            （/api/edu/transcribe、/api/edu/tts）与 RAG 问答也复用它的
            api_key / base_url
        vision_model: 视觉模型名称（保留给需要感知此配置的调用方）
    """

    def __init__(
        self,
        provider: LLMProvider,
        vision_model: str | None = None,
    ) -> None:
        """
        Args:
            provider: OpenAI 兼容 Provider 实例
            vision_model: 视觉模型名称，如 Qwen/Qwen3-VL-32B-Instruct
        """
        self.provider = provider
        self.vision_model = vision_model

    # ------------------------------------------------------------------
    # 错题集：学科自动分类 + 学科薄弱点分析
    # ------------------------------------------------------------------

    async def classify_subjects(self, questions: list[str]) -> list[str]:
        """批量判断错题的学科（一次 LLM 调用分类整批，收录 n 道只调 1 次）。

        任何失败（网络/输出格式非法）都降级为关键词规则兜底，
        保证错题收录流程永不因此阻塞。

        Args:
            questions: 题干摘要列表

        Returns:
            list[str]: 与输入一一对应的学科标签（语文/数学/英语/物理/化学/其他）
        """
        if not questions:
            return []
        prompt = SUBJECT_CLASSIFY_PROMPT.format(
            questions="\n".join(f"{i + 1}. {q[:120]}" for i, q in enumerate(questions))
        )
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                extra_params={"temperature": 0.0, "max_tokens": 120},
            )
            if response.finish_reason in ("stop", "length"):
                raw = (response.content or "").strip()
                m = re.search(r"\[[\s\S]*\]", raw)  # 容忍模型包裹的代码块/前后缀
                if m:
                    arr = json.loads(m.group())
                    valid = set(WRONGBOOK_SUBJECTS)
                    out: list[str] = []
                    for i, q in enumerate(questions):
                        s = arr[i] if i < len(arr) and isinstance(arr[i], str) else ""
                        out.append(s if s in valid else _fallback_subject(q))
                    return out
        except Exception as e:
            print(f"[EducationService] 学科分类 LLM 失败，关键词兜底: {e}")
        return [_fallback_subject(q) for q in questions]

    # 注：错题薄弱点分析已迁移至 EduAgentLoop._run_wrongbook
    # （经消息总线 → 网关 → AgentLoop 推理-工具循环，rag_search 可佐证复习建议）
