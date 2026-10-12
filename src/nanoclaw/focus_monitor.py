"""专注模式监督服务：连接 edu.html「专注模式」按钮与 YOLO-Pose 实时检测。

职责：
- 被 Web 渠道的 /api/edu/focus/* 接口驱动（start / stop / status）；
- 监督期间独占摄像头，持续抓帧 + YOLO-Pose 推理（独立线程，不阻塞事件循环），
  纯后端运行，摄像头画面不对外展示（不做标注/编码/推流，节省 CPU）；
- 每 SUMMARY_INTERVAL 秒汇总一次窗口内的逐帧结论（取众数），按提醒规则
  （分神/严重坐姿/离座）生成事件；事件只存内存，不写 SQLite；
- 前端轮询 status 发现新事件后，把提醒文本送 /api/edu/tts 合成语音直接
  播报给学生（语音失败时聊天区的文字提醒兜底）。

设计约束：
- 单摄像头独占：同一时刻只允许一个监督会话，重复 start 返回冲突；
- 不改动 yolo_vision.py：纯复用其公开的 analyze_pose_single /
  judge_posture / judge_focus 纯函数与阈值约定；
- 失败隔离：摄像头打开失败/推理异常记录 last_error 并停止监督，
  前端状态卡展示错误信息。
"""

import asyncio
import statistics
import threading
import time
from pathlib import Path

from agent.tools.yolo_vision import (
    _VISION_READY,
    _VISION_IMPORT_ERR,
    _resolve_model_path,
    analyze_pose_single,
    judge_posture,
    judge_focus,
)

# 逐帧推理节流间隔（秒）：CPU 推理约 5 FPS，兼顾流畅与占用
INFER_INTERVAL = 0.2
# 结论汇总周期（秒）：窗口内逐帧结论取众数作为一个"轮次"
SUMMARY_INTERVAL = 30.0
# 提醒事件保留上限（内存）
MAX_EVENTS = 100

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL = str(_PROJECT_ROOT / "models" / "yolov8n-pose.pt")


class FocusMonitor:
    """专注监督器：持续检测 + 事件生成（单摄像头，无画面输出）。"""

    def __init__(self) -> None:
        self.user_key: str | None = None       # 被监督学生的登录用户名
        self.started_at: float | None = None
        self.rounds = 0                        # 已完成的汇总轮数
        self.events: list[dict] = []           # 提醒事件（最新在后）
        self.latest: dict | None = None        # 最近一轮汇总结论
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None
        self._stop_event = threading.Event()
        # 上一次告警文本（去重用）：必须初始化，否则首个汇总窗口出现
        # 分神/坐姿告警时读取未定义属性，监督线程会静默崩溃
        self._last_alert_text: str | None = None
        self._model_path = _resolve_model_path(_DEFAULT_MODEL)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, user_key: str) -> dict:
        """开启监督（启动摄像头采集+推理线程）。

        Returns:
            {"ok": True, "summary_interval": ..} 或
            {"ok": False, "error": ...}（冲突/依赖缺失/已在运行）
        """
        if self.running:
            return {"ok": False, "error": "专注监督已在进行中（同一摄像头只能服务一个会话）"}
        if not _VISION_READY:
            return {"ok": False, "error": f"视觉依赖未就绪（需安装 ultralytics 与 opencv-python）：{_VISION_IMPORT_ERR}"}

        self.user_key = user_key or "guest"
        self.started_at = time.time()
        self.rounds = 0
        self.events = []
        self.latest = None
        self.last_error = None
        self._stop_event.clear()
        self._task = asyncio.create_task(asyncio.to_thread(self._stream_loop))
        return {"ok": True, "summary_interval": SUMMARY_INTERVAL}

    async def stop(self) -> dict:
        """停止监督（置停机事件，等采集线程退出并释放摄像头）。"""
        if self._task is not None and not self._task.done():
            self._stop_event.set()
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        self._task = None
        return {"ok": True, "rounds": self.rounds, "alert_count": len(self.events)}

    def status(self) -> dict:
        """当前状态快照（供前端轮询）。"""
        return {
            "running": self.running,
            "user_key": self.user_key,
            "rounds": self.rounds,
            "alert_count": len(self.events),
            "latest": self.latest,
            "events": self.events[-10:],
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    def _stream_loop(self) -> None:
        """采集+推理主循环（独立线程，纯后端无画面输出）：抓帧 → 推理 → 存结论。

        每帧把结论存入窗口，每 SUMMARY_INTERVAL 秒汇总一次并应用提醒规则。
        摄像头打开失败或连续推理异常时置 last_error 并结束监督。
        """
        import cv2
        from ultralytics import YOLO

        try:
            model = YOLO(self._model_path)
        except Exception as e:
            self.last_error = f"模型加载失败: {e}"
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.last_error = "无法打开摄像头，请检查摄像头是否被其他程序占用"
            return

        window: list[tuple[str, str, int]] = []  # (posture, focus, persons)
        window_started = time.time()
        last_infer = 0.0
        infer_fail = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    # 偶发读取失败跳过；持续失败报错退出
                    if time.time() - (last_infer or time.time()) > 5.0:
                        self.last_error = "摄像头画面读取中断"
                        break
                    time.sleep(0.05)
                    continue

                now = time.time()
                if now - last_infer < INFER_INTERVAL:
                    continue  # 节流：未到推理间隔的帧直接丢弃
                last_infer = now

                try:
                    results = model(frame, verbose=False)
                except Exception:
                    infer_fail += 1
                    if infer_fail >= 30:
                        self.last_error = "推理连续异常，监督已停止"
                        break
                    continue
                infer_fail = 0

                # 取最大人体做姿态分析（与 yolo_vision 同约定）
                posture, focus, persons = None, None, 0
                for r in results:
                    if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
                        continue
                    persons = int(r.keypoints.xy.shape[0])
                    kps = r.keypoints.xy[0].tolist()
                    if r.keypoints.conf is not None:
                        confs = r.keypoints.conf[0].tolist()
                    else:
                        confs = [1.0] * len(kps)
                    keypoints = [[kps[i][0], kps[i][1], confs[i]] for i in range(min(len(kps), 17))]
                    m = analyze_pose_single(keypoints)
                    posture = judge_posture(m)
                    focus = judge_focus(m)
                    break  # 只分析最大的一人，人数整体体现

                window.append((posture, focus, persons))

                # 到达汇总周期：窗口结论取众数作为一个轮次
                if now - window_started >= SUMMARY_INTERVAL:
                    window_started = now
                    self._summarize(window)
                    window = []
        finally:
            cap.release()
            self._stop_event.set()

    def _summarize(self, window: list[tuple[str | None, str | None, int]]) -> None:
        """把一个窗口的逐帧结论汇总为一个轮次，并应用提醒规则。"""
        now = time.time()
        self.rounds += 1

        if not window:
            self.last_error = "本轮无有效画面数据"
            return

        postures = [p for p, _, _ in window if p]
        focuses = [f for _, f, _ in window if f]
        persons_list = [n for _, _, n in window]

        posture = statistics.mode(postures) if postures else None
        focus = statistics.mode(focuses) if focuses else None
        persons = statistics.median(persons_list) if persons_list else 0
        self.latest = {
            "ts": now,
            "persons": int(persons),
            "posture": posture,
            "focus": focus,
        }

        # 离座：整个窗口几乎没人
        if persons_list and statistics.mode(persons_list) == 0:
            self._push_event(now, "画面中检测不到人，学生可能离开了座位", "leave")
            self._last_alert_text = None
            return

        # 注意力提醒：结论不是「注意力集中」即为分神信号
        focus_ok = focus is None or "注意力集中" in focus
        if not focus_ok and focus != self._last_alert_text:
            self._push_event(now, f"注意力提醒：{focus}", "focus")

        # 坐姿提醒：只报严重问题（轻微歪斜/低头较深属于常态，不打扰）
        posture_issue = None
        if posture and posture != "端正":
            if "伏案过深" in posture or "趴桌" in posture:
                posture_issue = posture
            elif "明显歪斜" in posture or "倾斜" in posture:
                posture_issue = posture
        if posture_issue and posture_issue != self._last_alert_text:
            self._push_event(now, f"坐姿提醒：{posture_issue}", "posture")

        self._last_alert_text = focus if not focus_ok else (posture_issue or focus)

    def _push_event(self, ts: float, message: str, kind: str) -> None:
        self.events.append({
            "ts": ts,
            "time": time.strftime("%H:%M:%S", time.localtime(ts)),
            "kind": kind,       # focus=分神 / posture=坐姿 / leave=离座
            "message": message,
        })
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]
