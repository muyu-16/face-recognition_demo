# -*- coding: utf-8 -*-
"""
live_enrollment.py
==================
**浏览器 WebRTC 实时录入人脸**：浏览器把摄像头画面推给 Python，
边抓边做"机位 + 质量"分析，**每个机位自动留下最好的一张**；
屏幕上实时显示"已覆盖 X/7"和"再向左转 3°（目标 15°~30°）"这类角度引导，
凑齐 7 机位黄金标准自动收工。

与另外两种录入方式的区别
------------------------
| 方式 | 谁在读摄像头 | 需要浏览器授权 | 需要先录视频 | 帧率 |
| --- | --- | --- | --- | --- |
| 📁 上传图片 | — | 否 | 否 | — |
| 📷 浏览器摄像头（连拍） | 浏览器 | **是** | 否 | 手动逐张 |
| **🎥 浏览器实时录入（WebRTC，本模块）** | 浏览器 | **是** | 否 | 约 20 FPS |

画面流畅（约 20 FPS），并且能**一边看画面一边看到实时提示**：
人脸框、当前机位与质量分、已覆盖机位进度、以及"下一步请把头向左转"。

工程要点（都是踩过的坑）
    1. **多后端依次尝试**：DSHOW → MSMF → 默认，并把每个后端的结果显示出来；
    2. **失败绝不缓存**：句柄只放 st.session_state，而且只放"确实 isOpened() 成功"的
       （早期版本用 @st.cache_resource 缓存了坏句柄，摄像头恢复了也打不开）；
    3. **预热**：打开后先丢弃几帧，很多摄像头前 3~5 帧是空的；
    4. **探测前先释放自己**：否则是自己挡自己，永远探测不到设备。
"""

from __future__ import annotations

import functools
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from enrollment_guide import (
    PLAN,
    PLAN_BY_KEY,
    REQUIRED_KEYS,
    YAW_TOO_MUCH,
    appearance_of,
    assess_quality,
    coverage,
)

import enrollment_guide as guide   # 角度阈值/引导语都从这里取，避免重复定义


def _window_text(key: str) -> str:
    """把机位的角度窗口写成 HUD 用的 ASCII 文本，例如 "(yaw 15-30 deg)"。"""
    window = guide.slot_window(key)
    if window is None:
        return ""
    (y_min, y_max), (p_min, p_max), _tol = window
    if key in ("half_left", "half_right", "big_left", "big_right"):
        return f"(yaw {y_min:.0f}-{y_max:.0f} deg)"
    if key in ("up", "down"):
        return f"(pitch {p_min:.0f}-{p_max:.0f} deg)"
    return f"(yaw<={y_max:.0f} pitch<={p_max:.0f} deg)"


def _ascii_advice(advice: str, target_key: Optional[str]) -> str:
    """把中文引导语映射成一句英文 HUD 提示（cv2.putText 画不了中文）。"""
    if not advice:
        return "keep going"
    if "抬" in advice and "低" not in advice:
        return "tilt UP a bit more"
    if "低" in advice:
        return "tilt DOWN a bit more"
    if "左" in advice and "右" not in advice:
        return "turn LEFT a bit more"
    if "右" in advice:
        return "turn RIGHT a bit more"
    if "回正" in advice or "放平" in advice:
        return "face the camera / level your gaze"
    if "反了" in advice:
        return "wrong direction - turn the other way"
    if "少一点" in advice:
        return "too much - ease off a bit"
    if "✅" in advice:
        return "angle OK"
    return f"next: {target_key}" if target_key else "keep going"

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# ---- 浏览器 WebRTC 实时录入所需依赖（可选）----
try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        webrtc_streamer,
    )

    try:                       # 0.4x 起 mode 要传 WebRtcMode 枚举
        from streamlit_webrtc import WebRtcMode

        _MODE_SENDRECV: object = WebRtcMode.SENDRECV
    except Exception:          # pragma: no cover
        _MODE_SENDRECV = "sendrecv"

    WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    webrtc_streamer = None
    _MODE_SENDRECV = "sendrecv"
    WEBRTC_AVAILABLE = False

_RTC_CONFIG = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if WEBRTC_AVAILABLE else None
)

DEFAULT_REFRESH = 0.4          # 每 0.4 秒抓一帧
DEFAULT_MIN_SCORE = 55         # 质量分下限
DEFAULT_PER_SLOT = 1           # 每个机位保留几张


# ===========================================================================
# 摄像头打开 / 释放 / 探测
# ===========================================================================
def annotate_preview(bgr: np.ndarray, face, record: Optional[dict],
                     accepted: bool, hud_lines: Sequence[str]) -> np.ndarray:
    """在预览画面上画人脸框 + 机位/质量标签 + HUD 文字，返回新的 BGR 图。"""
    from realtime import annotate_bgr          # 复用统一的"中文绘制"实现

    canvas = bgr
    if face is not None:
        label = ""
        unknown = True
        if record is not None:
            mark = "✅ 已采用" if accepted else "…"
            label = f"{record['slot_title']} {record['score']:.0f}分 {mark}"
            unknown = not accepted
        canvas = annotate_bgr(canvas, [face.box], [label], [unknown])

    if cv2 is not None:
        y = 26
        for line in hud_lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2, cv2.LINE_AA)
            y += 26
    return canvas


# ===========================================================================
# 实时录入会话：边抓边筛选，每个机位留最好的一张
# ===========================================================================
class LiveEnrollmentSession:
    """实时录入状态机。

    每次 step() 抓一帧、分析、决定是否收进某个机位；界面每 0.4 秒调一次。
    只修改自身属性（不写 st.session_state 的键），因此不会触发整页刷新。
    """

    def __init__(self, detector, per_slot: int = DEFAULT_PER_SLOT,
                 min_score: float = DEFAULT_MIN_SCORE) -> None:
        self.detector = detector
        self.per_slot = max(1, int(per_slot))
        self.min_score = float(min_score)
        self.best: Dict[str, List[dict]] = {}      # 机位 -> 已采用的帧（最多 per_slot 张）
        self.attempts = 0                          # 抓帧次数
        self.accepted = 0                          # 采用次数
        self.last_canvas: Optional[np.ndarray] = None
        self.last_note = "正在启动……"
        self.last_slot = ""
        self.last_face = None                      # 最近一次检测到的人脸（画框用）
        self.last_record: Optional[dict] = None    # 最近一次的机位/质量分析
        self.last_accepted = False                 # 最近一帧是否被收录
        self.nose_baseline: Optional[float] = None  # 本人"正视鼻-眼比"基线（自动标定）
        self.reference: Optional[dict] = None      # 参考正脸照的信息（提供本人俯仰基线）
        self.last_pose_line = ""                   # HUD 里那一行姿态/引导提示（ASCII）

    # ---------------- 主循环 ----------------
    def process_frame(self, bgr: np.ndarray) -> dict:
        """处理一帧 BGR 图像：检测 -> 机位/质量评估 -> 决定是否收录 -> 画框与 HUD。

        WebRTC 帧处理器每 N 帧调一次这个方法。
        """
        self.attempts += 1
        pil = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        faces = self.detector.detect(pil)

        if not faces:
            self.last_note = "没有检测到人脸：靠近镜头、正对光线、别遮住脸"
            self.last_face = None
            self.last_record = None
            self.last_accepted = False
            self.last_canvas = self.draw(bgr)
            return self.snapshot(accepted=False)

        face = faces[0]
        # 带上"本人正视基线"和"参考正面照外观"（如果已经标定过），
        # 抬头/低头的角度判定都会更准
        record = assess_quality(pil, face, nose_baseline=self.nose_baseline,
                                reference=self.reference)
        pose = record.get("pose", {}) or {}
        slot = record["slot"]
        accepted = False
        yaw_deg = float(pose.get("yaw_deg", 0.0) or 0.0)
        pitch_deg = float(pose.get("pitch_deg", 0.0) or 0.0)
        self.last_pose_line = ""

        # 用本人第一张"平视正面照"标定俯仰基线（人群经验值对个别人可能偏大偏小）
        if (self.nose_baseline is None and slot == "front"
                and abs(yaw_deg) <= guide.YAW_FRONT_DEG
                and abs(pitch_deg) <= guide.PITCH_TOL_DEG):
            baseline = float(pose.get("nose_ratio_corrected", float("nan")))
            if baseline == baseline and 0.20 <= baseline <= 0.95:      # 非 NaN 且在合理范围
                self.nose_baseline = baseline
        if self.reference is None and slot == "front":
            self.reference = appearance_of(record)

        # 当前应该去补哪个机位 —— 引导语就针对它说"还差几度"
        target = self.next_target()
        target_key = str(target["key"]) if target else None
        advice = guide.angle_advice(pose, target_key) or str(pose.get("guidance") or "")

        if abs(yaw_deg) > guide.YAW_TOO_MUCH_DEG:
            self.last_note = f"侧转过大（{yaw_deg:+.0f}°），稍微回一点（标准最大 60°）"
        elif record["score"] < self.min_score:
            self.last_note = (f"质量 {record['score']:.0f} 分偏低"
                              + (f"：{record['flags'][0]}" if record["flags"] else ""))
        else:
            accepted = self._offer(slot, record, pil)
            if accepted:
                self.last_note = (f"✅ 已收录「{record['slot_title']}」"
                                  f"（实测 左/右 {yaw_deg:+.0f}°、抬/低 {pitch_deg:+.0f}°，"
                                  f"质量 {record['score']:.0f} 分）")
                if self.nose_baseline is not None and slot == "front":
                    self.last_note += f"，并用这张标定了你的俯仰基线（{self.nose_baseline:.2f}）"
            elif advice:
                # 方向对了但幅度还不够 —— 直接说"再往哪边几度"，
                # 这比旧版的"「正面平视」已经有一张质量更高的了"有用得多
                self.last_note = (f"当前 左/右 {yaw_deg:+.0f}°、抬/低 {pitch_deg:+.0f}°"
                                  f"（本张会被算成「{record['slot_title']}」）→ **{advice}**")
                self.last_pose_line = (f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f} | "
                                       f"{_ascii_advice(advice, target_key)}")
            else:
                self.last_note = (f"「{record['slot_title']}」已经有一张质量更高的了"
                                  f"（本张 {record['score']:.0f} 分）")

        self.last_slot = slot
        self.last_face = face
        self.last_record = record
        self.last_accepted = accepted
        self.last_canvas = self.draw(bgr)
        return self.snapshot(accepted=accepted)

    def draw(self, bgr: np.ndarray, face=None, record: Optional[dict] = None,
             accepted: Optional[bool] = None) -> np.ndarray:
        """在画面上画人脸框 + 机位/质量标签 + HUD（WebRTC 每帧都会调它）。"""
        if face is None:
            face = self.last_face
        if record is None:
            record = self.last_record
        if accepted is None:
            accepted = self.last_accepted
        return annotate_preview(bgr, face, record, bool(accepted), self.hud_lines())

    def _offer(self, slot: str, record: dict, pil: Image.Image) -> bool:
        """把这一帧提交给某个机位：名额没满就收，满了就"比质量"替换。"""
        bucket = self.best.setdefault(slot, [])
        item = {"score": float(record["score"]), "image": pil, "analysis": record}
        if len(bucket) < self.per_slot:
            bucket.append(item)
            self.accepted += 1
            return True
        worst = min(bucket, key=lambda it: it["score"])
        if item["score"] > worst["score"]:
            bucket.remove(worst)
            bucket.append(item)
            self.accepted += 1
            return True
        return False

    # ---------------- 结果 ----------------
    def frames(self) -> List[Image.Image]:
        return [item["image"] for item in self.items()]

    def analyses(self) -> List[dict]:
        return [item["analysis"] for item in self.items()]

    def items(self) -> List[dict]:
        """按采集计划的顺序（正面→左→右→抬头→低头→…）返回已采用的帧。"""
        ordered: List[dict] = []
        for plan in PLAN:
            bucket = self.best.get(plan["key"], [])
            ordered.extend(sorted(bucket, key=lambda it: -it["score"]))
        return ordered

    def coverage(self) -> dict:
        return coverage(self.analyses())

    def is_complete(self) -> bool:
        return bool(self.coverage()["complete"])

    def next_target(self) -> Optional[dict]:
        key = self.coverage().get("next_key")
        return PLAN_BY_KEY.get(key) if key else None

    def hud_lines(self) -> List[str]:
        """给预览画面用的一行行提示（用 ASCII：cv2.putText 画不了中文）。"""
        cov = self.coverage()
        lines = [f"captured={self.attempts}  accepted={len(self.items())}  "
                 f"core={len(cov['filled_core'])}/{len(REQUIRED_KEYS)}"]
        if self.last_pose_line:              # 实测角度 + "再往哪边几度"的英文提示
            lines.append(self.last_pose_line)
        target = self.next_target()
        if target:
            window_text = _window_text(str(target["key"]))
            prefix = "optional" if cov["complete"] else "NEXT"
            lines.append(f"{prefix}: {target['key']} {window_text}")
        if cov["complete"]:
            lines.append("ALL 7 CORE SLOTS COVERED - you can stop now")
        return lines

    def snapshot(self, accepted: bool) -> dict:
        cov = self.coverage()
        return {
            "canvas": self.last_canvas,
            "note": self.last_note,
            "accepted": accepted,
            "attempts": self.attempts,
            "items": self.items(),
            "coverage": cov,
            "complete": bool(cov["complete"]),
            "next_target": self.next_target(),
        }

    def clear(self) -> None:
        self.best.clear()
        self.attempts = 0
        self.accepted = 0
        self.last_canvas = None
        self.last_note = "已清空，重新开始"
        self.last_face = None
        self.last_record = None
        self.last_accepted = False
        self.nose_baseline = None
        self.reference = None
        self.last_pose_line = ""


# ===========================================================================
# 浏览器 WebRTC 实时录入
# ===========================================================================
# 全进程共用一个 MTCNN 实例：既避免"每次开流都重新加载模型"（那是开流卡顿的主因之一），
# 也让 app.py 里已经加载/预热过的模型直接被复用。
_SHARED_DETECTOR = None
_SHARED_LOCK = threading.Lock()


def set_shared_detector(detector) -> None:
    """让实时录入复用调用方（app.py）已经加载好的检测器。"""
    global _SHARED_DETECTOR
    _SHARED_DETECTOR = detector


def get_shared_detector(device: str = "cpu"):
    """取全进程共用的检测器；没有就建一个（只建一次）。"""
    global _SHARED_DETECTOR
    if _SHARED_DETECTOR is None:
        with _SHARED_LOCK:
            if _SHARED_DETECTOR is None:
                from face_detector import FaceDetector

                _SHARED_DETECTOR = FaceDetector(device=device, align=True, keep_all=True)
    return _SHARED_DETECTOR


def warmup_detector(detector, size: Tuple[int, int] = (320, 240)) -> float:
    """预热：第一次前向最慢（torch 惰性初始化 + 算子选择），提前付掉这段时间。

    返回耗时秒数。失败不影响功能（只是没有加速效果）。
    """
    try:
        dummy = Image.new("RGB", size, (128, 128, 128))
        t0 = time.time()
        detector.detect(dummy)
        return time.time() - t0
    except Exception:
        return 0.0


if WEBRTC_AVAILABLE:

    class EnrollmentVideoProcessor(VideoProcessorBase):
        """WebRTC 逐帧录入处理器。

        结构（这是"开摄像头一开始很卡"的修复关键）：

            recv()  ← 浏览器每一帧都会调它，**必须立刻返回**
              │  1. 把当前帧交给后台线程（只保留最新一帧，旧的直接丢）
              │  2. 用"上一次的分析结果"把框和 HUD 画在当前帧上
              ▼
            后台线程 _analysis_loop()  ← 慢慢跑 MTCNN（CPU 上每帧 0.3~0.6 秒也没关系）

        为什么以前会卡：旧版直接在 recv() 里跑 MTCNN，一帧要 0.3~0.6 秒，
        而浏览器每秒推 20~30 帧 → 处理不过来就**排队积压**，表现就是"刚开流那几秒
        又卡又延迟"。等浏览器的拥塞控制自己把码率/帧率降下来之后才"看起来不卡了"。
        现在把检测挪到后台线程，视频通路永远不等模型，所以**一开就是流畅的**；
        后台线程处理不完就自动丢掉旧帧，永远只分析最新的那一帧。

        注意：本模式必须让 `async_processing=False`，
        这样处理器与 Streamlit 主脚本在**同一个进程**里，
        主脚本才能通过 `ctx.video_processor` 把它采集到的帧取出来入库。
        """

        def __init__(
            self,
            per_slot: int = DEFAULT_PER_SLOT,
            min_score: float = DEFAULT_MIN_SCORE,
            detect_interval: int = 2,
            backbone: str = "inception_resnet_v1",
            device: str = "auto",
            warmup: bool = True,
            idle_timeout: float = 45.0,
        ) -> None:
            # 复用全进程共用的模型（app.py 已加载过的那个），避免开流时重新加载
            self.detector = get_shared_detector(device=device)
            self.detect_interval = max(1, int(detect_interval))
            self.session = LiveEnrollmentSession(
                self.detector, per_slot=per_slot, min_score=min_score
            )
            self._frame_id = 0
            self.fps = 0.0
            self.analyze_ms = 0.0            # 最近一次分析的耗时（毫秒），显示用
            self.dropped = 0                 # 因为处理不过来而被丢掉的帧数
            self.busy = False                # 后台是否正在分析（界面显示"分析中"，测试也用它同步）
            self._t_last = time.time()
            self._idle_timeout = float(idle_timeout)

            # ---- 后台分析线程 ----
            self._lock = threading.Lock()
            self._pending: Optional[np.ndarray] = None
            self._stop = False
            self._last_frame_at = time.time()
            self.warmup_seconds = warmup_detector(self.detector) if warmup else 0.0
            self._worker = threading.Thread(target=self._analysis_loop,
                                            name="enroll-analysis", daemon=True)
            self._worker.start()

        # ---------------- 后台线程：真正跑模型的地方 ----------------
        def _analysis_loop(self) -> None:
            while not self._stop:
                frame = None
                with self._lock:
                    if self._pending is not None:
                        frame, self._pending = self._pending, None
                if frame is None:
                    # 长时间没有新帧（用户按了 STOP）就自己退出，避免线程泄漏
                    if time.time() - self._last_frame_at > self._idle_timeout:
                        break
                    time.sleep(0.005)
                    continue
                t0 = time.time()
                self.busy = True
                try:
                    self.session.process_frame(frame)
                except Exception:
                    # 视频流里绝不能抛异常，否则整条流会断
                    pass
                finally:
                    self.analyze_ms = (time.time() - t0) * 1000.0
                    self.busy = False

        # ---------------- 视频通路：绝不阻塞 ----------------
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self._frame_id += 1

            now = time.time()
            dt = now - self._t_last
            self._t_last = now
            self._last_frame_at = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            # 每 detect_interval 帧提交一次；后台忙不过来时直接覆盖旧帧（不排队）
            if (self._frame_id - 1) % self.detect_interval == 0:
                with self._lock:
                    if self._pending is not None:
                        self.dropped += 1          # 上一帧还没被分析就被替换了
                    self._pending = img.copy()

            canvas = self.session.draw(img)          # 用缓存的框 + 最新 HUD 画在当前帧上
            if cv2 is not None:
                cv2.putText(canvas, f"{self.fps:4.1f} FPS", (10, canvas.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(canvas, format="bgr24")


def create_enrollment_processor(**kwargs):
    """模块级工厂（WebRTC 在某些配置下会要求可 pickle，所以不用局部闭包）。"""
    return EnrollmentVideoProcessor(**kwargs)


def render_webrtc_enrollment(detector=None, per_slot: int = DEFAULT_PER_SLOT,
                             min_score: float = DEFAULT_MIN_SCORE,
                             detect_interval: int = 2,
                             guide_renderer=None,
                             name: str = "") -> List[Tuple[str, Image.Image]]:
    """渲染"浏览器 WebRTC 实时录入"面板，返回采集到的 [(标签, 图), ...]。

    流程：点 START → 浏览器授权 → 画面里实时显示"已收录 X/5 + 下一步转头提示"
    → **达标后自动保存**（也可以手动点按钮）→ 下方出现帧与覆盖表 → 点注册。

    两条硬规则（用户要求）：
      1. **必须先填姓名**：没填姓名时不会保存（自动/手动都不行），页面会一直提示；
      2. **达标自动保存**：条件可选（集齐核心机位 / 集齐且平均质量分达标 /
         集齐且每张都达标），达标即把帧冻结到 session_state，之后按 STOP 也不会丢。
    """
    # 复用 app.py 已经加载（并预热）过的检测器，避免开流时重新加载模型造成卡顿
    if detector is not None:
        set_shared_detector(detector)
    if not WEBRTC_AVAILABLE:
        st.warning(
            "浏览器实时录入需要额外依赖 **streamlit-webrtc** 与 **av**（可选安装）。\n\n"
            "没装也可以先用「📷 摄像头拍照（连拍）」或「📁 上传图片」，效果一样。"
        )
        st.code("uv pip install --python .venv\\Scripts\\python.exe streamlit-webrtc av",
                language="bash")
        return []

    has_name = bool(str(name).strip())

    st.markdown(
        "1. **先在①处填好姓名**（没填姓名不会保存）；\n"
        "2. 点下面的 **START**，浏览器弹「允许使用摄像头？」→ 点**允许**；\n"
        "3. 站到镜头前，**跟着画面上的提示慢慢转头**——画面上会实时叠加：\n"
        "   人脸框 + 实测角度(yaw/pitch) + `core=x/7` + `NEXT: half_left (yaw 15-30 deg)`；\n"
        "4. **达标后会自动保存**（也可以在达标后手动点按钮），下方出现帧与覆盖表，"
        "再点最下面的「🚀 提取特征并注册」即可。"
    )
    if not has_name:
        st.warning("⚠️ **请先在「① 姓名」里输入姓名**——采集达标后才会自动保存。")
    st.info(
        "⚠️ 摄像头授权对**页面地址**敏感：必须用 `http://localhost:8501` 打开。\n"
        "用局域网 IP 访问时浏览器会**直接禁用**摄像头（连授权弹窗都不会出现）；"
        "误点过「阻止」后弹窗也不会再出现，需要在地址栏左侧的网站设置里把摄像头改回「允许」并刷新。"
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        detect_interval = st.slider("每 N 帧检测一次", 1, 15, int(detect_interval), 1,
                                    help="检测在后台线程跑，不会拖慢画面；"
                                         "数值小 = 框和提示更新更勤，但更吃 CPU")
    with col_p2:
        min_score = st.slider("单帧最低质量分", 0, 90, int(min_score), 5,
                              help="低于这个分数的帧一开始就不会被收录")
    with col_p3:
        per_slot = st.slider("每个机位保留几张", 1, 3, int(per_slot), 1)
    st.caption("💡 刚点 START 时通常要 1~2 秒建立连接（画面可能先黑一下），"
               "之后应当是流畅的；模型已经在后台线程里跑，不会卡住画面。")

    with st.expander("⚙️ 自动保存设置", expanded=True):
        auto_save = st.toggle("达标后自动保存（不用再点按钮）", value=True)
        rule_options = ["集齐 7 个核心机位（黄金标准）",
                        "集齐且**平均**质量分达标", "集齐且**每张**都达标"]
        col_a1, col_a2 = st.columns([2, 1])
        with col_a1:
            rule = st.radio("达标条件", rule_options, index=0, horizontal=False,
                            help="越往后越严格；"
                                 "「质量分」是清晰度/亮度/对比度/姿态综合出来的 0~100 分")
            rule_index = rule_options.index(rule)      # st.radio 返回的是文本，这里换成下标
        with col_a2:
            target_score = st.slider("质量分门槛", 0, 100, 80, 5,
                                     disabled=(rule_index == 0))

    def _meets_rule(best_items: List[dict]) -> Tuple[bool, str]:
        """判断是否达到"自动保存"条件，返回 (是否达标, 说明)。

        注意：这里按**选项下标**判断，不要按文案字符串判断
        （踩过坑：文案从"集齐 5 个"改成"集齐 7 个"后 startswith 就匹配不上了，
        默认条件静默失效，变成"每张都要 80 分"）。
        """
        cov_now = coverage([it["analysis"] for it in best_items])
        if not cov_now["complete"]:
            return False, ""
        scores = [float(it["score"]) for it in best_items]
        if not scores:
            return False, ""
        if rule_index == 0:
            return True, "已集齐 7 个核心机位（黄金标准）"
        if rule_index == 1:
            avg = sum(scores) / len(scores)
            return avg >= target_score, f"平均质量分 {avg:.0f}（门槛 {target_score}）"
        lowest = min(scores)
        return lowest >= target_score, f"最低质量分 {lowest:.0f}（门槛 {target_score}）"

    def _freeze(best_items: List[dict], reason: str) -> None:
        """把当前帧冻结到 session_state（之后按 STOP 也不会丢）。"""
        st.session_state["webrtc_frames"] = [it["image"] for it in best_items]
        st.session_state["webrtc_analysis"] = [
            {k: v for k, v in it["analysis"].items() if k != "face"} for it in best_items
        ]
        st.session_state["webrtc_saved_reason"] = reason
        st.session_state["webrtc_saved_count"] = len(best_items)

    # async_processing=False 是关键：处理器与主脚本同进程，主脚本才能取到它采集的帧
    ctx = webrtc_streamer(
        key="live-enrollment-webrtc",
        mode=_MODE_SENDRECV,
        video_processor_factory=functools.partial(
            create_enrollment_processor,
            per_slot=int(per_slot),
            min_score=float(min_score),
            detect_interval=int(detect_interval),
        ),
        rtc_configuration=_RTC_CONFIG,
        # 640×480 是精度与速度的平衡点：MTCNN 的耗时大致与画面面积成正比，
        # 降到 480×360 还能再快一倍，但关键点会略微变糙。
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        },
        async_processing=False,
    )

    processor = ctx.video_processor if ctx is not None else None
    if processor is None:
        st.info("还没有开始采集：点上面的 **START** 并允许摄像头，采集结果会自动出现在这里。")
        return []

    session = processor.session

    def _status_and_autosave() -> None:
        """定时刷新的状态区：显示进度、判断是否达标、达标即自动保存。

        放在 st.fragment 里（每 0.5 秒重跑这一小块），所以"达标"能被立刻发现，
        而不需要用户去点任何按钮。这里只读处理器的状态、只在达标时写一次 session_state。
        """
        best_items = session.items()
        cov_now = session.coverage()
        ok, why = _meets_rule(best_items)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("已收录机位（核心）", f"{len(cov_now['filled_core'])}/{len(REQUIRED_KEYS)}")
        col_b.metric("已收录帧", len(best_items))
        col_c.metric("分析帧数", session.attempts)
        col_d.metric("实时帧率", f"{processor.fps:.1f}")
        st.caption(f"分析耗时 {processor.analyze_ms:.0f} ms/帧"
                   f"（在后台跑，不影响画面）"
                   + (f"｜因忙不过来丢弃 {processor.dropped} 帧"
                      if processor.dropped else ""))
        st.progress(cov_now["core_ratio"])
        st.caption(f"本次状态：{session.last_note}")

        saved_count = st.session_state.get("webrtc_saved_count", 0)
        if ok and has_name:
            if saved_count != len(best_items):
                _freeze(best_items, why)
            st.success(f"✅ 已自动保存 {len(best_items)} 帧（{why}）"
                       f"——下方可以直接注册，按 **STOP** 关闭摄像头即可")
        elif ok and not has_name:
            st.error("⚠️ 已经达标，但**还没有填姓名**，所以不会保存。\n\n"
                     "请在「① 姓名」里输入姓名（填完这里会自动保存，不用再拍）。")
        else:
            next_target = session.next_target()
            if next_target:
                st.info(f"👉 下一步：**{next_target['title']}**\n\n{next_target['hint']}")
            if auto_save:
                st.caption("达标后会自动保存" + ("（当前条件：%s）" % rule))
            else:
                st.caption("已关闭自动保存：达标后请手动点下面的「✅ 采集完成，取用这些帧」")

        # 冻结过的帧直接显示在这一块里（所以 STOP 之后依然看得到）
        frozen_frames = st.session_state.get("webrtc_frames") or []
        frozen_analysis = st.session_state.get("webrtc_analysis") or []
        if frozen_frames:
            st.markdown(f"#### 📸 已保存的帧（{len(frozen_frames)} 张，每张都是该机位里质量最高的）")
            cols = st.columns(min(5, len(frozen_frames)))
            for idx, img in enumerate(frozen_frames):
                rec = frozen_analysis[idx] if idx < len(frozen_analysis) else {}
                with cols[idx % 5]:
                    try:
                        st.image(img, caption=f"{rec.get('slot_title', '')}"
                                              f"｜{rec.get('score', 0):.0f} 分", width="stretch")
                    except TypeError:                       # pragma: no cover
                        st.image(img, caption=f"{rec.get('slot_title', '')}"
                                              f"｜{rec.get('score', 0):.0f} 分",
                                 use_container_width=True)
            if guide_renderer is not None:
                guide_renderer(frozen_analysis)

    # 流式期间每 0.5 秒刷新一次状态区（只重跑这一小块，不整页刷新）。
    # 保存完成之后就停下定时刷新：既省 CPU，也不再每 0.5 秒重传一遍已保存的图片。
    already_saved = bool(st.session_state.get("webrtc_frames"))
    fragment = st.fragment(run_every=(None if already_saved else 0.5))(_status_and_autosave)
    fragment()

    # ---------------- 手动保存 / 标记 / 清空 ----------------
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        manual = st.button("✅ 采集完成，取用这些帧", type="primary",
                           disabled=not has_name,
                           help="没填姓名时不可用" if not has_name else "把当前采集到的帧冻结保存")
    with col_m2:
        if st.button("🗑️ 清空重录"):
            session.clear()
            st.session_state["webrtc_frames"] = []
            st.session_state["webrtc_analysis"] = []
            st.session_state["webrtc_saved_count"] = 0
            st.session_state.pop("webrtc_saved_reason", None)
            st.rerun()

    if manual:
        if not has_name:
            st.error("请先填写姓名再保存。")
        else:
            items_now = session.items()
            _freeze(items_now, "手动保存")
            st.success(f"✅ 已保存 {len(items_now)} 帧")
            st.rerun()

    # ---------------- 返回给注册流程 ----------------
    frozen_frames = st.session_state.get("webrtc_frames") or []
    if frozen_frames:
        return [(f"WebRTC_{i + 1}", img) for i, img in enumerate(frozen_frames)]

    # 还没保存（例如：还没达标、或没填姓名）时，也先把当前帧交给注册流程，
    # 这样用户如果就是想手动注册也不被挡住。
    return [(f"WebRTC_{i + 1}", it["image"]) for i, it in enumerate(session.items())]
