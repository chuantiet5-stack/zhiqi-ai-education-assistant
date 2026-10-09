"""
YOLO 视觉工具：坐姿检测 + 分神检测（教育场景）

定位：作为 Tool 接入 nanoclaw 的 ToolRegistry，供 AgentLoop 在对话中调用。
LLM 可通过该工具实时读取摄像头画面，得到「坐姿结论」与「注意力状态」
的结构化文本结论，用于【监督学习 / 自习陪学】场景（提醒坐姿、判定分神）。

场景约定（重要）：
- 摄像头置于学生前方/侧上方，学生伏案低头看学习平板/书本为正常状态；
- 注意力判定原则：**抬头或乱看（转头/侧脸/身体侧转）才判定分神**，
  而不是“看学习平板=正常、不看屏幕=分神”那种平板的自我监督逻辑；
- 坐姿判定：端正=坐直且轻微自然低头；严重低头伏案或歪斜才给提醒，
  避免把正常低头学习误报为“趴桌/盯手机”。

实现要点：
- 视觉模型：Ultralytics YOLO-Pose（yolov8n-pose.pt，COCO 17 关键点），
  兼容 COCO 关键点序号：0 鼻,1 左眼,2 右眼,3 左耳,4 右耳,5 左肩,6 右肩,
  7 左肘,8 右肘,9 左腕,10 右腕,11 左髋,12 右髋,13 左膝,14 右膝,15 左踝,16 右踝
- 采样策略：默认从摄像头连续采集数秒多帧，按帧计算指标后取中位数统计，
  避免单帧抖动造成误报（尤其 CPU 推理下帧率低、结果波动大）。
- 事件循环安全：摄像头采集与推理均为 CPU/IO 阻塞操作，统一丢进
  asyncio.to_thread 线程池执行，不阻塞 Agent 的事件循环（与 web_search 同约定）。
- 容错约定：依赖缺失 / 无摄像头 / 推理异常一律返回错误字符串而非抛异常，
  保证 Agent 对话循环不被中断（与 registry 的容错约定一致）。

依赖：ultralytics + opencv-python。
权重位于项目 models/yolov8n-pose.pt（本地化，无需外网下载），可替换为
更高精度版本（yolov8s-pose/m-pose）或自定义权重。
"""

import asyncio
import math
import os
import pathlib
import statistics
from typing import Any, Optional

from agent.tools.base import Tool

# ---- 依赖探测：缺失时降级，在 execute 中给出明确错误提示 ----
try:
    import cv2  # noqa: F401
    from ultralytics import YOLO

    _VISION_READY = True
    _VISION_IMPORT_ERR = ""
except Exception as _e:  # pragma: no cover
    cv2 = None
    YOLO = None
    _VISION_READY = False
    _VISION_IMPORT_ERR = str(_e)


# =====================================================================
# 对外连接接口（默认关闭）
# ---------------------------------------------------------------------
# 用途：供学习平板 / 上位机 / 伴学服务订阅 yolo_vision 的实时检测结果，
#       用于远端展示、告警或联动提醒。
# 后续对接步骤（已注释，届时取消注释即可启用）：
#   1. CONNECTION_ENABLED 置为 True；
#   2. 填入真实 CONNECTION_ENDPOINT（如伴学服务地址）；
#   3. 在 YoloVisionTool._external_hook 中取消注释实际请求代码并按协议适配载荷。
# =====================================================================
CONNECTION_ENABLED = False          # 对外推送总开关
CONNECTION_ENDPOINT = ""            # 目标地址，例: "http://127.0.0.1:9000/vision"
CONNECTION_TARGET = "study-client"  # 目标标识，供远端识别调用方
CONNECTION_TIMEOUT = 3.0            # 对外请求超时(秒)
# =====================================================================


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # agent/tools/ -> 项目根
_DEFAULT_MODEL = str(_PROJECT_ROOT / "models" / "yolov8n-pose.pt")


def _resolve_model_path(model_path: str) -> str:
    """相对路径优先按项目根解析，便于从任意工作目录启动"""
    if os.path.isabs(model_path):
        return model_path
    cand = str(_PROJECT_ROOT / model_path)
    return cand if os.path.exists(cand) else model_path


# COCO 17 关键点索引命名，便于阅读
_KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}


def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _mid(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _angle_deg(vx: float, vy: float) -> float:
    """向量与「竖直向上」方向的夹角(0~180 度)，用于衡量低头/抬头。

    图像坐标 y 轴向下：坐直平视时鼻尖在肩中点上方，vy 为负 → 夹角小；
    低头时鼻尖下移/前移，vy 增大 → 夹角大；只有后仰过顶才接近 180。
    """
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return 0.0
    cos = -vy / norm  # 与竖直向上单位向量 (0,-1) 的夹角
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def analyze_pose_single(keypoints) -> dict:
    """
    对单帧关键点计算坐姿与注意力指标（规则引擎，纯函数便于测试）。

    Args:
        keypoints: 单人的 17x3 关键点数组 [[x, y, conf], ...]

    Returns:
        dict: 各项指标的原始度量 + 布尔标志 + 文字结论
    """
    def kp(i):
        x, y, c = keypoints[i]
        return (x, y) if c > 0.3 else None

    nose = kp(_KP["nose"])
    l_sh = kp(_KP["left_shoulder"])
    r_sh = kp(_KP["right_shoulder"])
    l_hip = kp(_KP["left_hip"])
    r_hip = kp(_KP["right_hip"])
    l_ear = kp(_KP["left_ear"])
    r_ear = kp(_KP["right_ear"])

    metrics = {"shoulder_tilt": None, "head_down": None, "lean": None,
               "side_ratio": None, "head_turn": None, "shoulder_span": None}

    if l_sh and r_sh:
        # 肩膀连线相对水平的倾斜角：越大说明一侧肩膀吊越高（歪斜）
        metrics["shoulder_tilt"] = math.degrees(math.atan2(abs(l_sh[1] - r_sh[1]),
                                                           abs(l_sh[0] - r_sh[0]) + 1e-6))
        metrics["shoulder_span"] = abs(l_sh[0] - r_sh[0])
        # 侧身程度：正面时双肩 x 间距达到最大；侧身时间距收缩。
        # 以肩宽与肩髋竖直距离比作为近似归一化尺度。
        # 仅当髋部可见时计算，否则置 None，避免分母退化导致数值失真（如爆成 1e8）。
        if l_hip and r_hip:
            sh_m = _mid(l_sh, r_sh)
            hip_m = _mid(l_hip, r_hip)
            spine_len = abs(sh_m[1] - hip_m[1]) + 1e-6
            metrics["side_ratio"] = metrics["shoulder_span"] / spine_len
        else:
            metrics["side_ratio"] = None

        # 左右倾：肩中点相对髋中点的水平偏移，正=偏左 负=偏右
        # 归一化尺度取「肩髋竖直距离」，避免局部高度差极小时放大误报
        if l_hip and r_hip:
            sh_mid = _mid(l_sh, r_sh)
            hip_mid = _mid(l_hip, r_hip)
            d = sh_mid[0] - hip_mid[0]
            spine = abs(sh_mid[1] - hip_mid[1]) + 1e-6  # 肩线与髋线的竖直距离
            metrics["lean"] = d / spine

    if nose and l_sh and r_sh:
        sh_mid = _mid(l_sh, r_sh)
        # 低头/抬头：鼻尖相对肩中点的向量与「竖直向上」方向的夹角。
        # 夹角小=坐直平视, 增大=低头(看平板/写字), 严重时>65 视为伏案过深。
        metrics["head_down"] = _angle_deg(nose[0] - sh_mid[0], nose[1] - sh_mid[1])

    if l_ear and r_ear:
        # 头部左右偏转：两耳可见间距占肩宽比例，越小越可能侧脸
        ear_span = abs(l_ear[0] - r_ear[0])
        if metrics["shoulder_span"] and metrics["shoulder_span"] > 10:
            metrics["head_turn"] = ear_span / metrics["shoulder_span"]

    return metrics


def judge_posture(m: dict) -> str:
    """根据单帧指标给出坐姿文字判断（监督学习/自习场景，规则阈值可调）。

    原则：端正=坐直且轻微自然低头；只有明显歪斜/躯干侧倾/伏案过深才提醒，
    避免把正常低头看书/看平板误报为不良坐姿。
    """
    issues = []
    if m["shoulder_tilt"] is not None:
        if m["shoulder_tilt"] > 18:
            issues.append("肩膀明显歪斜(单侧过高)")
        elif m["shoulder_tilt"] > 8:
            issues.append("身体轻微左右歪斜")
    if m["head_down"] is not None:
        if m["head_down"] > 140:
            issues.append("伏案过深,疑似趴桌")
        elif m["head_down"] > 60:
            issues.append("低头较深,可适当挺直")
    if m["lean"] is not None:
        if abs(m["lean"]) > 0.35:
            issues.append("躯干向一侧倾斜")
    if not issues:
        return "端正"
    return "、".join(issues)


def judge_focus(m: dict) -> str:
    """根据单帧指标给出注意力判断（监督学习/自习陪学场景）。

    判定原则（与场景约定一致）：
    - 正常：低头注视学习内容（head_down 处于伏案区间、面部朝前/朝下）；
    - 分神：① 抬头（head_down 过小，甚至后仰）或 ② 乱看（转头侧脸/身体侧转）。
    """
    # ① 抬头：坐直或后仰时鼻尖与肩基本同高/在上方 → head_down 很小
    if m["head_down"] is not None:
        if m["head_down"] < 8:
            return "明显抬头,未注视学习内容,疑似分神"
        if m["head_down"] < 22:
            return "头部抬起偏高,可能走神"
    # ② 乱看：头部偏转(侧脸)或身体侧转
    if m["head_turn"] is not None and m["head_turn"] < 0.5:
        return "头部偏向一侧,疑似乱看(分神)"
    if m["side_ratio"] is not None and m["side_ratio"] < 0.5:
        return "身体侧转明显,可能分神"
    return "低头注视学习内容,注意力集中"


class YoloVisionTool(Tool):
    """
    坐姿与注意力检测工具（YOLO-Pose）。

    实时读取摄像头画面，返回坐姿端正度与注意力状态的文字结论，
    用于【监督学习/自习陪学】场景：学生低头伏案=正常，抬头或乱看=分神，
    并对严重歪斜/伏案过深的坐姿给出提醒。
    """

    # diffirent 采样帧数上限，防止 duration 过长导致长时间阻塞
    MAX_FRAMES = 120

    @property
    def name(self) -> str:
        return "yolo_vision"

    @property
    def description(self) -> str:
        return (
            "检测学生/用户的实时坐姿与注意力状态（监督学习/自习陪学场景）。"
            "需要了解当前坐姿是否端正、是否分神、是否离开座位时使用。"
            "注意力判定以「抬头或乱看」为分神信号（正常=低头注视学习内容）。"
            "从摄像头采集画面分析，返回："
            "1) 画面中人数；2) 坐姿结论（端正/歪斜/低头过深/倾斜）；"
            "3) 注意力结论（注视学习/抬头/乱看/分神/离开）。"
            "适合监督孩子学习、自习室管理等场景的坐姿提醒与分神提醒。"
            "参数 mode 可选 posture(仅坐姿) / focus(仅注意力) / full(两者,默认)。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["posture", "focus", "full"],
                    "description": "检测模式：posture=仅坐姿, focus=仅注意力, full=两者全检测",
                    "default": "full",
                },
                "duration": {
                    "type": "number",
                    "description": "采样时长(秒)，默认 3，越长结果越稳但等待越久",
                    "default": 3,
                },
                "camera_index": {
                    "type": "integer",
                    "description": "摄像头索引，默认 0（通常为内置摄像头）",
                    "default": 0,
                },
                "model_path": {
                    "type": "string",
                    "description": "YOLO-Pose 权重路径，默认项目 models/yolov8n-pose.pt",
                    "default": "models/yolov8n-pose.pt",
                },
            },
            "required": ["mode"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if not _VISION_READY:
            return (
                "错误: 视觉依赖未就绪（ultralytics 或 opencv 未安装），"
                f"导入异常: {_VISION_IMPORT_ERR}。请先安装: pip install ultralytics opencv-python"
            )

        mode = kwargs.get("mode", "full")
        duration = float(kwargs.get("duration", 3))
        camera_index = int(kwargs.get("camera_index", 0))
        model_path = _resolve_model_path(kwargs.get("model_path", "models/yolov8n-pose.pt"))

        if mode not in ("posture", "focus", "full"):
            return f"错误: 不支持的 mode={mode}，可选 posture/focus/full"

        try:
            # 阻塞的采集+推理整体丢进线程池
            result = await asyncio.to_thread(
                self._run_detection, mode, duration, camera_index, model_path
            )
        except Exception as e:
            return f"错误: yolo_vision 执行失败 - {e}"

        # 仅在拿到有效检测结果时尝试对外推送（连接接口默认关闭，见模块顶部）
        if result.startswith("检测结果:"):
            await self._external_hook(result)
        return result

    # ------------------------------------------------------------------
    def _run_detection(self, mode: str, duration: float,
                       camera_index: int, model_path: str) -> str:
        import cv2
        from ultralytics import YOLO

        model = YOLO(model_path)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            return "错误: 无法打开摄像头，请检查摄像头是否被占用或索引是否正确"

        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        frames_to_read = max(1, min(int(duration * fps), self.MAX_FRAMES))

        posture_notes: list[str] = []
        focus_notes: list[str] = []
        person_counts: list[int] = []
        seen = False

        try:
            for _ in range(frames_to_read):
                ok, frame = cap.read()
                if not ok:
                    break
                results = model(frame, verbose=False)

                person_cnt = 0
                for r in results:
                    if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
                        continue
                    person_cnt = person_cnt + r.keypoints.xy.shape[0]
                    # 取画面中最大的一个人体做姿态分析（置信度最高的框）
                    kps = r.keypoints.xy[0].tolist()
                    confs = r.keypoints.conf[0].tolist()
                    keypoints = [[kps[i][0], kps[i][1], confs[i]] for i in range(min(len(kps), 17))]
                    m = analyze_pose_single(keypoints)
                    if mode in ("posture", "full"):
                        posture_notes.append(judge_posture(m))
                    if mode in ("focus", "full"):
                        focus_notes.append(judge_focus(m))
                    seen = True
                    break  # 只对单条框分析，多人场景由人数统计体现
                person_counts.append(person_cnt)
        finally:
            cap.release()

        if not seen or not person_counts:
            return "检测结果: 画面中未检测到人体（可能无人、背对摄像头或距离过远）"

        lines = [
            f"检测人数: {statistics.median(person_counts):.0f} (采样{len(person_counts)}帧)"
        ]
        if mode in ("posture", "full"):
            # 取出现最多的结论作为稳定判定
            top = statistics.mode(posture_notes) if posture_notes else "无有效姿态"
            lines.append(f"坐姿结论: {top}")
        if mode in ("focus", "full"):
            top_f = statistics.mode(focus_notes) if focus_notes else "无有效注意力"
            lines.append(f"注意力结论: {top_f}")
        return "检测结果:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # 对外连接接口（默认关闭，见模块顶部 CONNECTION_* 配置）
    # 后续对接：CONNECTION_ENABLED=True 并填入 CONNECTION_ENDPOINT 后，
    #           取消下方被注释的请求代码并按实际协议适配载荷即可启用。
    # ------------------------------------------------------------------
    async def _external_hook(self, report: str) -> None:
        """将本次检测报告推送给外部对接方（学习平板/上位机/伴学服务）。

        设计约束：
        - 默认关闭：CONNECTION_ENABLED=False 时直接短路返回，零开销；
        - 失败隔离：任何异常都被吞掉，绝不因推送失败影响主检测结果；
        - 纯占位：当前仅预留协议结构与开关，未产生任何网络请求。
        """
        if not CONNECTION_ENABLED or not CONNECTION_ENDPOINT:
            return
        try:
            # TODO(对接): 取消注释并按实际协议适配（请求体/鉴权/重试策略）
            # import aiohttp
            # async with aiohttp.ClientSession() as session:
            #     await session.post(
            #         CONNECTION_ENDPOINT,
            #         json={"target": CONNECTION_TARGET,
            #               "type": "yolo_vision_report",
            #               "report": report},
            #         timeout=aiohttp.ClientTimeout(total=CONNECTION_TIMEOUT),
            #     )
            return
        except Exception:  # 对接失败不影响本地检测结果
            return
