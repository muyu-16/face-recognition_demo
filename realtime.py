# -*- coding: utf-8 -*-
"""
realtime.py
===========
加分项：摄像头**实时**人脸识别（浏览器 WebRTC 视频流，逐帧检测识别）。

本模块只负责"浏览器 WebRTC 视频流"这一种实时方案：
    浏览器通过 WebRTC 把摄像头画面推给 Python，Python 端每一帧都会进
    FaceRecognitionProcessor.recv()，识别完再把画好框的帧送回浏览器。
    画面最流畅（约 20 FPS），绿框=已注册人员、红框=陌生人，左上角显示 FPS。

另一种"手动逐张"的方案（连拍识别）实现在 app.py 的 render_burst_mode() 里，
它只用 Streamlit 内置的 camera_input，不需要 streamlit-webrtc / av。

依赖：
    uv pip install --python .venv\\Scripts\\python.exe streamlit-webrtc av
    # 没装时 app.py 的"实时识别"页只会显示安装提示，不影响其它功能

关键工程细节（都是踩过的坑）
    1. **跳帧检测**：不是每帧都跑 MTCNN，而是每 N 帧检测一次（detect_interval），
       中间的帧直接复用上一次的检测框，否则 CPU 上画面会卡成 PPT；
    2. **人脸库缓存**：库的（姓名, 特征矩阵）缓存若干秒，避免每帧都读 pickle；
    3. **工厂函数必须可 pickle**：streamlit-webrtc 的 async_processing=True 会把
       video_processor_factory 送进子进程执行，局部闭包在 Windows 上会直接报
       "Can't pickle local object"，所以这里用模块级函数 + functools.partial；
    4. **模型在子进程内部构建**，不跨进程序列化 PyTorch 模型（Windows 上尤其容易出问题）；
    5. **浏览器授权对地址敏感**：必须用 localhost 或 HTTPS 打开，
       用局域网 IP 访问时浏览器会直接禁用摄像头（连弹窗都不会有）。
"""

from __future__ import annotations

import functools
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ---- 可选依赖（cv2 / streamlit-webrtc / av）统一由 webrtc_common 探测，缺失时降级 ----
from webrtc_common import (
    av,
    cv2,
    VideoProcessorBase,
    webrtc_streamer,
    WEBRTC_AVAILABLE,
    _MODE_SENDRECV,
    _RTC_CONFIG,
)


# ===========================================================================
# 公共工具：画框、人脸库缓存、识别
# ===========================================================================
def annotate_bgr(
    frame_bgr: np.ndarray,
    boxes: Sequence[np.ndarray],
    labels: Sequence[str],
    unknown: Sequence[bool],
    line_width: int = 3,
) -> np.ndarray:
    """在 BGR 画面上画人脸框 + 中文标签，返回新的 BGR 图像。

    为什么不用 cv2.putText：它不支持中文，姓名会变成一堆问号。
    所以这里统一把 BGR 转成 PIL 画完再转回 BGR。
    """
    if not boxes:
        return frame_bgr

    from face_detector import load_font

    img_rgb = frame_bgr[:, :, ::-1].copy()
    pil = Image.fromarray(np.ascontiguousarray(img_rgb))
    draw = ImageDraw.Draw(pil)
    font = load_font(20)

    for box, text, is_unknown in zip(boxes, labels, unknown):
        color = (255, 64, 64) if is_unknown else (0, 255, 0)
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if text:
            ty = max(0.0, y1 - 26)
            try:
                l, t, r, b = draw.textbbox((x1, ty), text, font=font)
            except Exception:  # pragma: no cover
                l, t, r, b = x1, ty, x1 + 130, ty + 22
            draw.rectangle([l - 3, t - 2, r + 3, b + 2], fill=color)
            draw.text((x1, ty), text, fill=(0, 0, 0), font=font)

    rgb_out = np.asarray(pil, dtype=np.uint8)
    return np.ascontiguousarray(rgb_out[:, :, ::-1])


def load_gallery(db_path: str, face_dir: str, ttl: float = 5.0,
                 cache: Optional[dict] = None) -> Tuple[List[str], np.ndarray]:
    """读取人脸库的 (姓名列表, 特征矩阵)，带 TTL 缓存，避免每帧都读 pickle。"""
    holder = cache if cache is not None else {}
    now = time.time()
    if holder.get("data") is not None and (now - holder.get("at", 0.0)) < ttl:
        return holder["data"]

    try:
        from face_database import FaceDatabase

        db = FaceDatabase(db_path=db_path, face_dir=face_dir, auto_load=True)
        data = db.get_gallery()
    except Exception:
        # 读库失败时退回上一份缓存（首次则退回空库），保证视频流不会中断
        data = holder.get("data") or ([], np.zeros((0, 512), dtype=np.float32))
    holder["data"] = data
    holder["at"] = now
    return data


def _recognize_faces(
    detector,
    extractor,
    names: List[str],
    gallery: np.ndarray,
    pil_image: Image.Image,
    threshold: float,
    max_faces: int,
    aggregate: str = "max",
    mean_topk: int = 3,
):
    """对一帧做 检测 -> 提特征 -> 比对，返回 (boxes, labels, unknown)。"""
    from similarity import identify

    faces = detector.detect(pil_image)
    if not faces:
        return [], [], []

    faces = faces[: max(1, int(max_faces))]
    boxes = [face.box for face in faces]

    # 人脸库为空时不必提特征，直接把每张脸标成"人脸库为空"
    if not names or gallery.shape[0] == 0:
        return boxes, ["人脸库为空"] * len(faces), [True] * len(faces)

    crops = [detector.crop(pil_image, f) for f in faces]
    embeddings = extractor.extract_batch(crops, batch_size=8)

    labels: List[str] = []
    unknown: List[bool] = []
    for emb in embeddings:
        result = identify(emb, names, gallery, threshold=float(threshold), top_k=3,
                          aggregate=aggregate, mean_topk=int(mean_topk))
        if result.is_known:
            labels.append(f"{result.name} {result.similarity:.2f}")
            unknown.append(False)
        else:
            labels.append(f"陌生人 {result.similarity:.2f}")
            unknown.append(True)
    return boxes, labels, unknown


# ===========================================================================
# 浏览器 WebRTC 视频流
# ===========================================================================
if WEBRTC_AVAILABLE:

    class FaceRecognitionProcessor(VideoProcessorBase):
        """WebRTC 视频帧处理器：检测 -> 识别 -> 把画好框的帧送回浏览器。

        只接收"路径 + 参数"这类简单配置，模型在 __init__ 里自己加载：
        因为 async_processing=True 时这个对象活在**独立子进程**里，
        传 PyTorch 模型进去会跨进程序列化，Windows 上非常容易出问题。
        """

        def __init__(
            self,
            db_path: str = "face_db.pkl",
            face_dir: str = "registered_faces",
            backbone: str = "inception_resnet_v1",
            device: str = "auto",
            threshold: float = 0.6,
            detect_interval: int = 8,
            min_face_size: int = 40,
            margin: int = 20,
            align: bool = True,
            aggregate: str = "max",
            mean_topk: int = 3,
            max_faces: int = 5,
            gallery_ttl: float = 5.0,
        ) -> None:
            from face_detector import FaceDetector
            from feature_extractor import FeatureExtractor

            self.detector = FaceDetector(
                image_size=160, margin=margin, min_face_size=min_face_size,
                device=device, align=align, keep_all=True,
            )
            pretrained = "vggface2" if backbone == "inception_resnet_v1" else True
            self.extractor = FeatureExtractor(backbone=backbone, pretrained=pretrained,
                                              device=device)
            self.db_path = db_path
            self.face_dir = face_dir
            self.threshold = float(threshold)
            self.detect_interval = max(1, int(detect_interval))
            self.aggregate = aggregate
            self.mean_topk = int(mean_topk)
            self.max_faces = int(max_faces)
            self.gallery_ttl = float(gallery_ttl)

            self._lock = threading.Lock()
            self._frame_id = 0
            self._boxes: List[np.ndarray] = []
            self._labels: List[str] = []
            self._unknown: List[bool] = []
            self._gallery_cache: Dict[str, object] = {}
            self.fps = 0.0
            self._t_last = time.time()

        def recv(self, frame):
            """收到一帧 -> 处理 -> 返回画好结果的一帧。"""
            img_bgr = frame.to_ndarray(format="bgr24")
            self._frame_id += 1

            now = time.time()
            dt = now - self._t_last
            self._t_last = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            # 每 detect_interval 帧才真正跑一次"检测 + 识别"，其余帧复用上次的框。
            # 用 (frame_id - 1) % N：这样 N=1（每帧都检测）也能成立，
            # 写成 frame_id % N == 1 的话 N=1 会永远不检测（踩过的坑）。
            if (self._frame_id - 1) % self.detect_interval == 0:
                try:
                    self._detect_and_recognize(img_bgr)
                except Exception:
                    # 视频流里绝不能抛异常，否则整条流会断掉
                    pass

            with self._lock:
                boxes = list(self._boxes)
                labels = list(self._labels)
                unknown = list(self._unknown)

            canvas = annotate_bgr(img_bgr, boxes, labels, unknown)
            if cv2 is not None:
                cv2.putText(canvas, f"{self.fps:4.1f} FPS", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(canvas, format="bgr24")

        def _detect_and_recognize(self, img_bgr: np.ndarray) -> None:
            pil = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
            names, gallery = load_gallery(self.db_path, self.face_dir,
                                          ttl=self.gallery_ttl, cache=self._gallery_cache)
            boxes, labels, unknown = _recognize_faces(
                self.detector, self.extractor, names, gallery, pil,
                threshold=self.threshold, max_faces=self.max_faces,
                aggregate=self.aggregate, mean_topk=self.mean_topk,
            )
            with self._lock:
                self._boxes, self._labels, self._unknown = boxes, labels, unknown


def create_webrtc_processor(**kwargs):
    """WebRTC 处理器的**模块级**工厂函数（必须能被 pickle，见文件头注释第 3 条）。

    streamlit-webrtc 在 async_processing=True 时会把工厂函数送进子进程，
    局部闭包在 Windows 上会报 "Can't pickle local object: _factory"。
    """
    return FaceRecognitionProcessor(**kwargs)


def render_realtime_tab(
    db_path: Path | str,
    face_dir: Path | str,
    backbone: str,
    device: str,
    threshold: float,
    detect_interval: int,
    min_face_size: int,
    margin: int,
    align: bool,
    aggregate: str,
    mean_topk: int,
    max_faces: int = 5,
    async_processing: bool = True,
) -> None:
    """渲染浏览器 WebRTC 视频流界面。"""
    if not WEBRTC_AVAILABLE:
        st.warning(
            "浏览器视频流需要额外依赖 **streamlit-webrtc** 与 **av**（可选安装）。\n\n"
            "没装也可以先用「📷 连拍识别」模式演示——它只用 Streamlit 内置摄像头。"
        )
        st.code("uv pip install --python .venv\\Scripts\\python.exe streamlit-webrtc av",
                language="bash")
        return

    st.markdown(
        "1. 点下面的 **START**，浏览器会弹出「允许使用摄像头？」→ 点**允许**；\n"
        "2. 画面里：**绿框** = 已注册人员（框上显示姓名与相似度），**红框** = 陌生人；\n"
        "3. 左上角显示 FPS。侧边栏「每 N 帧检测一次」控制识别频率：数值越大越流畅。"
    )
    st.info(
        "⚠️ 摄像头授权对**页面地址**敏感：必须用 `http://localhost:8501` 打开。\n"
        "如果用局域网 IP（如 `http://192.168.x.x:8501`）访问，浏览器会**直接禁用摄像头**，"
        "连授权弹窗都不会出现。授权弹窗被误点「阻止」后也不会再出现，"
        "需要在地址栏左侧的网站设置里把摄像头改回「允许」，然后刷新页面。"
    )

    # 用 functools.partial 绑定参数：partial 可被 pickle，闭包不行
    factory = functools.partial(
        create_webrtc_processor,
        db_path=str(db_path),
        face_dir=str(face_dir),
        backbone=backbone,
        device=device,
        threshold=float(threshold),
        detect_interval=int(detect_interval),
        min_face_size=int(min_face_size),
        margin=int(margin),
        align=bool(align),
        aggregate=aggregate,
        mean_topk=int(mean_topk),
        max_faces=int(max_faces),
    )

    try:
        webrtc_streamer(
            key="face-realtime",
            mode=_MODE_SENDRECV,
            video_processor_factory=factory,
            rtc_configuration=_RTC_CONFIG,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=bool(async_processing),
        )
    except Exception as err:
        # 不往外抛：抛出去会让整个页面变成报错页，这里给出可操作的处理建议
        st.error(
            f"视频流启动失败：{err}\n\n"
            "可以按下面顺序处理：\n"
            "1. 把上面的「在独立进程中处理视频」开关**关掉**再试"
            "（pickle / multiprocessing 相关报错基本都是这个原因）；\n"
            "2. 确认页面是用 `http://localhost:8501` 打开的"
            "（用局域网 IP 时浏览器会禁用摄像头）；\n"
            "3. 检查浏览器摄像头权限：地址栏左侧图标 → 网站设置 → 摄像头 → 允许 → 刷新；\n"
            "4. 关掉可能占用摄像头的程序（微信 / 腾讯会议 / OBS 等）。"
        )
        with st.expander("查看完整错误堆栈"):
            st.code(traceback.format_exc(), language="text")
        return

    st.caption(
        "首次启动要在视频处理进程里加载 MTCNN 与特征模型，可能要等 10~30 秒；"
        "画面卡顿就把侧边栏「每 N 帧检测一次」调大（例如 15），或改用 GPU。"
    )
