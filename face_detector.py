# -*- coding: utf-8 -*-
"""
face_detector.py
================
人脸检测 + 对齐 + 裁剪模块（基于 facenet-pytorch 的 MTCNN）。

MTCNN（Multi-task Cascaded Convolutional Networks，多任务级联卷积网络）
由 P-Net / R-Net / O-Net 三级网络级联而成：
    1. P-Net 在图像金字塔上快速生成候选框；
    2. R-Net 过滤掉大量误检框；
    3. O-Net 精修边界框，并回归出 5 个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）。
它的输出正好是"人脸框 + 5 点关键点"，非常适合做人脸识别的预处理。

本模块在此基础上额外提供两个能力：
    * 对齐（alignment）：用双眼关键点把倾斜人脸旋转到"双眼连线水平"的标准姿态，
      使同一个人的不同姿态图片特征更接近；
    * 裁剪（crop）：按人脸框 + 边距裁成正方形并缩放到 160x160（FaceNet 的输入尺寸）。

对外主要接口（app.py 只依赖这几个）：
    detector = FaceDetector(...)
    faces    = detector.detect(image)                 # 只检测
    faces, crops = detector.detect_and_crop(image)    # 检测 + 裁剪（喂给特征提取器）
    canvas   = detector.draw(image, faces, labels)    # 画框，用于界面展示

输入输出约定：
    - 输入图片可以是 文件路径 / PIL.Image / numpy.ndarray；
    - numpy 数组一律按 **RGB** 通道顺序解释（OpenCV 读入的 BGR 请先自己转成 RGB）；
    - 输出的人脸图统一为 numpy.ndarray，dtype=uint8，形状 (H, W, 3)，RGB 顺序。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from facenet_pytorch import MTCNN

# ---------------------------------------------------------------------------
# 类型别名与默认参数
# ---------------------------------------------------------------------------
ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]

DEFAULT_IMAGE_SIZE = 160   # FaceNet 系列模型的输入边长
DEFAULT_MARGIN = 20        # 裁剪时四周额外扩出的边距（原图像素）
DEFAULT_MIN_FACE_SIZE = 40  # 小于该尺寸的人脸会被 MTCNN 忽略

# 常见中文字体，用于在结果图上绘制中文姓名（找不到就退回默认字体）
_CJK_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """决定使用 CPU 还是 GPU（CUDA）。

    preferred 取值：
        None / "auto"  -> 有可用的 CUDA 显卡就用显卡，否则用 CPU
        "cpu"          -> 强制 CPU
        "cuda" / "cuda:0" -> 强制显卡（不可用时抛错）
    说明：本函数是项目公共工具，feature_extractor / app.py 也会复用。
    """
    if preferred is None or str(preferred).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = str(preferred).lower()
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "你选择了 GPU（CUDA），但当前环境检测不到可用的 CUDA 设备。\n"
            "请改选 CPU，或安装与本机显卡匹配的 CUDA 版 PyTorch。"
        )
    return torch.device(preferred)


@dataclass
class Face:
    """一张检测到的人脸。"""

    box: np.ndarray                              # [x1, y1, x2, y2]，原图像素坐标
    prob: float                                  # 置信度，0~1
    landmarks: Optional[np.ndarray] = None       # (5, 2) 关键点：左右眼、鼻尖、左右嘴角

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """人脸框面积，用来挑选"最大的人脸"。"""
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (float((self.box[0] + self.box[2]) / 2.0),
                float((self.box[1] + self.box[3]) / 2.0))

    def as_dict(self) -> dict:
        return {
            "box": [round(float(v), 2) for v in self.box],
            "prob": round(float(self.prob), 4),
            "size": [round(self.width, 1), round(self.height, 1)],
        }


class FaceDetector:
    """MTCNN 人脸检测器封装：检测 -> 对齐 -> 裁剪。"""

    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        margin: int = DEFAULT_MARGIN,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
        thresholds: Sequence[float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        device: Optional[str] = None,
        align: bool = True,
        keep_all: bool = True,
    ) -> None:
        """
        参数说明：
            image_size    : 裁剪输出的人脸图边长（FaceNet 用 160）
            margin        : 裁剪时四周额外扩出的像素边距，留一点背景更稳
            min_face_size : 最小可检测人脸边长，太小的人脸会被忽略
            thresholds    : P/R/O 三级网络的置信度阈值
            factor        : 图像金字塔的缩放因子，越小越慢但越不容易漏检
            device        : 'auto' / 'cpu' / 'cuda'
            align         : 是否用关键点做双眼水平对齐
            keep_all      : True=保留所有人脸，False=只保留最大的一张
        """
        self.image_size = int(image_size)
        self.margin = int(margin)
        self.min_face_size = int(min_face_size)
        self.align = bool(align)
        self.keep_all = bool(keep_all)
        self.device = resolve_device(device)

        # post_process=False：不做标准化，像素标准化的差异留给 feature_extractor 处理，
        # 这样检测模块与特征模块解耦，便于现场向老师解释每一步。
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=self.margin,
            min_face_size=int(min_face_size),
            thresholds=[float(t) for t in thresholds],
            factor=float(factor),
            keep_all=self.keep_all,
            select_largest=not self.keep_all,
            post_process=False,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    @staticmethod
    def to_pil(image: ImageLike) -> Image.Image:
        """把 路径 / PIL / numpy 统一转换成 RGB 的 PIL.Image。"""
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:                       # 灰度图 -> RGB
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:               # 浮点图（0~1 或 0~255）转成 uint8
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
        else:
            img = Image.open(image)                  # 文件路径
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
        """把人脸框限制在图像范围内。"""
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(x1, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        x2 = max(0.0, min(x2, width - 1.0))
        y2 = max(0.0, min(y2, height - 1.0))
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _map_points(points: np.ndarray, angle_deg: float,
                    center: Tuple[float, float]) -> np.ndarray:
        """把点集按 PIL 旋转相同的规则做坐标映射。

        PIL 的 Image.rotate(a) 表示把图像内容逆时针旋转 a 度；
        在"x 向右、y 向下"的图像坐标系里，等价于对以 center 为原点的向量做：
            x' =  cos(a) * x + sin(a) * y
            y' = -sin(a) * x + cos(a) * y
        关键点必须跟着一起变换，否则旋转后框会跑偏。
        """
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        pts = np.asarray(points, dtype=np.float32)
        cx, cy = float(center[0]), float(center[1])
        vx, vy = pts[:, 0] - cx, pts[:, 1] - cy
        out = np.stack([cos_t * vx + sin_t * vy,
                        -sin_t * vx + cos_t * vy], axis=1)
        out[:, 0] += cx
        out[:, 1] += cy
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def detect(self, image: ImageLike) -> List[Face]:
        """检测图中所有人脸，按人脸面积从大到小排序。"""
        pil = self.to_pil(image)
        # MTCNN.detect 返回 (boxes, probs, landmarks)；
        # 图片里没有人脸时 boxes 为 None，所以必须判空。
        boxes, probs, landmarks = self.mtcnn.detect(pil, landmarks=True)
        faces: List[Face] = []
        if boxes is None:
            return faces

        for i, box in enumerate(boxes):
            if box is None:
                continue
            prob = float(probs[i]) if probs is not None and probs[i] is not None else 0.0
            lm = landmarks[i] if landmarks is not None else None
            faces.append(
                Face(
                    box=self._clip_box(np.asarray(box, dtype=np.float32),
                                       pil.width, pil.height),
                    prob=prob,
                    landmarks=None if lm is None else np.asarray(lm, dtype=np.float32),
                )
            )

        faces.sort(key=lambda f: f.area, reverse=True)  # 大脸优先，通常是画面主角
        return faces

    def detect_largest(self, image: ImageLike) -> Optional[Face]:
        """只取最大的那张脸（单人识别场景最常用）。"""
        faces = self.detect(image)
        return faces[0] if faces else None

    # ------------------------------------------------------------------
    # 对齐 + 裁剪
    # ------------------------------------------------------------------
    def crop(
        self,
        image: ImageLike,
        face: Face,
        image_size: Optional[int] = None,
        margin: Optional[int] = None,
    ) -> np.ndarray:
        """把一张人脸裁成 (image_size, image_size, 3) 的 uint8 RGB 图。

        流程：可选对齐旋转 -> 按框加边距取正方形区域 -> 缩放。
        正方形裁剪很关键：如果直接按长方形框缩放，人脸会被拉扁，特征会失真。
        """
        size = int(image_size or self.image_size)
        pad = int(self.margin if margin is None else margin)
        pil = self.to_pil(image)
        box = np.asarray(face.box, dtype=np.float32)

        # ---- 步骤 1：对齐。用双眼连线角度把脸转正，并把框一起旋转 ----
        if self.align and face.landmarks is not None and len(face.landmarks) >= 2:
            left_eye, right_eye = face.landmarks[0], face.landmarks[1]
            dx = float(right_eye[0] - left_eye[0])
            dy = float(right_eye[1] - left_eye[1])
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                angle = math.degrees(math.atan2(dy, dx))
                if abs(angle) >= 0.5:  # 小于 0.5 度就没必要旋转，省时间
                    center = (pil.width / 2.0, pil.height / 2.0)
                    pil = pil.rotate(angle, resample=Image.BILINEAR,
                                     center=center, fillcolor=(0, 0, 0))
                    corners = np.array([[box[0], box[1]], [box[2], box[1]],
                                        [box[2], box[3]], [box[0], box[3]]],
                                       dtype=np.float32)
                    corners = self._map_points(corners, angle, center)
                    box = np.array([corners[:, 0].min(), corners[:, 1].min(),
                                    corners[:, 0].max(), corners[:, 1].max()],
                                   dtype=np.float32)
                    box = self._clip_box(box, pil.width, pil.height)

        # ---- 步骤 2：扩边距 + 取正方形 ----
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        side = max(float(box[2] - box[0]), float(box[3] - box[1])) + 2.0 * pad
        side = max(side, 4.0)  # 防止异常小框导致裁剪失败
        x1, y1 = cx - side / 2.0, cy - side / 2.0
        x2, y2 = cx + side / 2.0, cy + side / 2.0

        # 先按图像边界裁剪（可能不再是正方形），再用黑边补齐成正方形，避免形变
        crop = pil.crop((int(round(x1)), int(round(y1)),
                         int(round(x2)), int(round(y2))))
        crop = self._pad_to_square(crop)

        # ---- 步骤 3：缩放到模型输入尺寸 ----
        if crop.size != (size, size):
            crop = crop.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(crop, dtype=np.uint8)

    @staticmethod
    def _pad_to_square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
        """用纯色补齐成正方形（居中放置原图）。"""
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    def detect_and_crop(
        self,
        image: ImageLike,
        max_faces: Optional[int] = None,
    ) -> Tuple[List[Face], List[np.ndarray]]:
        """一步完成"检测 + 裁剪"，返回 (人脸列表, 人脸图列表)，两者一一对应。

        max_faces: 最多处理几张脸（按面积从大到小取），None 表示不限。
        """
        faces = self.detect(image)
        if max_faces is not None:
            faces = faces[: int(max_faces)]
        crops = [self.crop(image, f) for f in faces]
        return faces, crops

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def draw(
        self,
        image: ImageLike,
        faces: Sequence[Face],
        labels: Optional[Sequence[str]] = None,
        color=(0, 255, 0),
        unknown_color=(255, 64, 64),
        unknown_flags: Optional[Sequence[bool]] = None,
        line_width: int = 3,
        draw_landmarks: bool = True,
    ) -> Image.Image:
        """在图上画人脸框、关键点和标签，返回新的 PIL.Image（不修改原图）。

        unknown_flags 为 True 的框用红色显示（用于标记"陌生人"）。
        """
        canvas = self.to_pil(image).copy()
        draw = ImageDraw.Draw(canvas)
        font = _load_font(max(14, int(min(canvas.size) / 35)))

        for idx, face in enumerate(faces):
            is_unknown = bool(unknown_flags[idx]) if unknown_flags is not None and idx < len(unknown_flags) else False
            box_color = unknown_color if is_unknown else color
            x1, y1, x2, y2 = [float(v) for v in face.box]
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

            # 关键点：小圆点 + 连左右眼，直观展示对齐依据
            if draw_landmarks and face.landmarks is not None:
                for (px, py) in face.landmarks:
                    r = 2
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 215, 0))

            text = ""
            if labels is not None and idx < len(labels) and labels[idx]:
                text = str(labels[idx])
            if text:
                tx, ty = x1, max(0.0, y1 - font.size - 6)
                try:
                    l, t, r, b = draw.textbbox((tx, ty), text, font=font)
                except Exception:            # 极老版本 Pillow 兜底
                    l, t, r, b = tx, ty, tx + 8 * len(text), ty + font.size
                draw.rectangle([l - 3, t - 2, r + 3, b + 2], fill=box_color)
                draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
        return canvas

    def __repr__(self) -> str:  # 方便在 Streamlit 里直接打印对象
        return (f"FaceDetector(image_size={self.image_size}, margin={self.margin}, "
                f"align={self.align}, device='{self.device}')")


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """加载一个能显示中文的字体；找不到就退回 Pillow 默认字体（中文会显示为方块）。"""
    for path in _CJK_FONT_CANDIDATES:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# 公开别名：realtime.py 等模块需要用它来绘制中文姓名
load_font = _load_font


# ---------------------------------------------------------------------------
# 模块自测：python face_detector.py [图片路径]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    detector = FaceDetector(device="auto")
    print("[face_detector] 初始化完成 ->", detector)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        faces, crops = detector.detect_and_crop(path)
        print(f"[face_detector] 在 {path} 中检测到 {len(faces)} 张人脸")
        for i, (f, c) in enumerate(zip(faces, crops), 1):
            print(f"  人脸 {i}: box={f.as_dict()['box']} prob={f.prob:.3f} "
                  f"crop={c.shape} {c.dtype}")
        if faces:
            out = detector.draw(path, faces, [f"face{i}" for i in range(len(faces))])
            out.save("detect_preview.png")
            print("[face_detector] 结果图已保存为 detect_preview.png")
    else:
        # 没有传图片时，用一张纯色图验证"无人脸"分支不会崩溃
        blank = Image.new("RGB", (320, 240), (128, 128, 128))
        print("[face_detector] 纯色图检测结果：", len(detector.detect(blank)), "张人脸（预期 0）")
        print("[face_detector] 用法：python face_detector.py <图片路径>")
