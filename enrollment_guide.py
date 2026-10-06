# -*- coding: utf-8 -*-
"""
enrollment_guide.py
===================
人脸"采集指引"：把人脸注册从"随手拍 5 张"变成"有覆盖的 5 张"。

为什么需要它（也是本模块要解决的问题）
--------------------------------------
人脸识别的相似度是一个**相对打分**：只跟人脸库里已有的样本比。
如果 5 张注册照都是同一个角度、同一种光照，会发生两件坏事：

1. **覆盖不足**：实时识别时头稍微一转、换个光，特征就"不认识"了，
   相似度掉到阈值以下 → 本人被拒；
2. **打分被压缩**：库内样本彼此太像时，同一个人的不同姿态差异被放大、
   不同人之间的差异被压缩，于是分数集中在高位（例如 0.85~0.95），
   **绝对值很高但区分度变差**——这就是"到 90% 以上反而不准"的根源。
   这时更该看的是 **top1 与 top2 的分差（margin）**，而不是绝对相似度。

本模块提供 4 个能力（都是纯函数，不依赖 Streamlit，方便单测）：
    * estimate_pose(landmarks)        —— 用 MTCNN 的 5 个关键点粗估头部姿态（左右/俯仰）
    * assess_quality(image, face)     —— 清晰度 / 亮度 / 对比度 / 人脸占比 / 姿态幅度
    * classify_slot(pose, size_ratio) —— 判断这张属于采集计划里的哪个"机位"
    * coverage(records)               —— 统计已覆盖 / 还缺哪些机位，并给出下一步拍法

注意：姿态估计用的是"正面假设 + 关键点几何"的**近似**算法，
用来判断"这张是正面还是侧脸"足够可靠，但不是精确的头部姿态测量。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 采集计划：core 为必做机位（正好 5 张，满足作业"至少 5 张"的要求且有覆盖），
#           extra 为加分机位（想更稳就补上）
# ---------------------------------------------------------------------------
PLAN: List[Dict[str, object]] = [
    {"key": "front", "group": "core", "title": "正脸（0°，平视）",
     "hint": "正对镜头、双眼平视、表情自然、肩膀放松。这张最关键，先拍它。",
     "yaw": (0.0, 10.0), "pitch": (0.0, 8.0), "tol": 3.0},
    {"key": "half_left", "group": "core", "title": "左半侧（左偏 15°~30°）",
     "hint": "头慢慢向左转，**眼睛仍看着镜头**；转到 20° 上下（画面里鼻尖偏向右侧）。",
     "yaw": (15.0, 30.0), "pitch": (0.0, 12.0), "tol": 4.0},
    {"key": "half_right", "group": "core", "title": "右半侧（右偏 15°~30°）",
     "hint": "同上，朝另一边转。左右各一张，别只拍一边。",
     "yaw": (15.0, 30.0), "pitch": (0.0, 12.0), "tol": 4.0},
    {"key": "up", "group": "core", "title": "轻微抬头（10°~15°）",
     "hint": "下巴微抬、**眼睛仍然看着镜头**（能看到一点鼻孔就差不多了）。"
             "别只抬眉毛/翻眼睛，也不要后仰太多。",
     "yaw": (0.0, 12.0), "pitch": (10.0, 15.0), "tol": 5.0},
    {"key": "down", "group": "core", "title": "轻微低头（10°~15°）",
     "hint": "下巴微收、**眼睛仍然看着镜头**（想象镜头比你的眼睛高一点）。"
             "不要真的低头看键盘或手机，那样五官会被挡住。",
     "yaw": (0.0, 12.0), "pitch": (10.0, 15.0), "tol": 5.0},
    {"key": "big_left", "group": "core", "title": "左大侧（左偏 45°~60°）",
     "hint": "继续向左转到接近侧脸（能明显看到一侧脸颊）。"
             "这一张最能救「侧脸实时识别翻车」。",
     "yaw": (45.0, 60.0), "pitch": (0.0, 15.0), "tol": 6.0},
    {"key": "big_right", "group": "core", "title": "右大侧（右偏 45°~60°）",
     "hint": "同上，朝另一边转到接近侧脸。",
     "yaw": (45.0, 60.0), "pitch": (0.0, 15.0), "tol": 6.0},
    {"key": "near", "group": "extra", "title": "靠近镜头一点（加分）",
     "hint": "让脸占画面更大，模拟近距离刷脸（人脸占比会明显变大）。"},
]

PLAN_BY_KEY: Dict[str, Dict[str, str]] = {item["key"]: item for item in PLAN}
REQUIRED_KEYS: List[str] = [item["key"] for item in PLAN if item["group"] == "core"]
OPTIONAL_KEYS: List[str] = [item["key"] for item in PLAN if item["group"] == "extra"]

# 人脸占比（人脸框面积 / 整图面积）的经验阈值
FACE_RATIO_TOO_FAR = 0.04      # 小于它 = 脸太小（离太远 / 图太大）
FACE_RATIO_NEAR = 0.30         # 大于它才算"靠近镜头"（笔记本自拍常在 0.15~0.30）
FACE_RATIO_OK = 0.06           # 达到它就算"够大、可用于注册"

# ---------------------------------------------------------------------------
# 姿态：比值 ↔ 角度 的换算（用真实注册照标定出来的，见 README）
#   yaw_ratio   ≈ YAW_TAN_K   × tan(θ)    → 半侧 22°≈0.22、大侧 52°≈0.70
#   pitch_offset≈ PITCH_SIN_K × sin(φ)    → 抬头/低头 12°≈0.19
# 标定依据：旧注册照里 ~30° 侧脸比值 0.360、~50° 侧脸 0.618（反解 K≈0.52~0.62）；
#           现注册照里低头那张比值变化 0.227（反解 φ≈14.6°，与目测 15° 一致）。
# 有了它，判定就能直接写在"15°~30°""45°~60°"这些**角度窗口**上，
# 而不是抽象的比值——用户也终于能看到"当前转了多少度、还差几度"。
# ---------------------------------------------------------------------------
YAW_TAN_K = 0.55               # yaw_ratio ≈ 0.55·tan(θ)
PITCH_SIN_K = 0.90             # pitch_offset ≈ 0.90·sin(φ)

# 姿态幅度阈值（角度制）
YAW_FRONT_DEG = 10.0           # |yaw| ≤ 10° 算正脸
YAW_HALF_DEG = 22.5            # 半侧机位中心
YAW_BIG_DEG = 52.5             # 大侧机位中心
YAW_TOO_MUCH_DEG = 70.0        # 超过 70° 关键点会明显漂，提示"转太多"
PITCH_TOL_DEG = 8.0            # |pitch| ≤ 8° 算平视
PITCH_LIGHT_DEG = 12.5         # 轻微抬头/低头机位中心

# 以下比值阈值由角度阈值换算而来（保持向后兼容：老代码/自检仍在用这些名字）
YAW_FRONT = YAW_TAN_K * float(np.tan(np.radians(YAW_FRONT_DEG)))
YAW_TURNED = YAW_TAN_K * float(np.tan(np.radians(YAW_HALF_DEG)))
YAW_TOO_MUCH = YAW_TAN_K * float(np.tan(np.radians(YAW_TOO_MUCH_DEG)))
PITCH_TOL = PITCH_SIN_K * float(np.sin(np.radians(PITCH_TOL_DEG)))
# ---- 俯仰判定用的是"鼻-眼垂直距离"，不是"嘴-眼距离" ----
# 实测（真实注册照，见 README）：正视 0.52~0.65；低头照 0.780（变化 23%）；
# 而"嘴-眼距离"在同一张照上只变化 14% —— 鼻-眼距离信噪比高得多，
# 这就是原先"抬头/低头很难录入"的主因。
FRONTAL_NOSE_RATIO = 0.58      # 正视时"眼→鼻尖的垂直距离 / 眼距"的人群基线（本人基线会自动标定）
PITCH_REL = PITCH_SIN_K * float(np.sin(np.radians(PITCH_LIGHT_DEG)))   # ≈0.195，pitch_score 达到 1.0
YAW_NOSE_COMPENSATION = 0.30   # 侧转对"鼻-眼比"的保守补偿（见下）
# 为什么系数只有 0.3、而不是拟合出来的更大值：
#   真实照片上这个耦合的**符号都不稳定**（小幅侧转让鼻-眼比略降、大幅侧转让它升高），
#   所以只保留一个保守补偿避免大误差；真正的精度靠"用本人正脸照当俯仰基线" +
#   "侧转窗口优先判定"来保证（俯仰只在小角度侧转时才参与机位判定）。
# 人脸占比超过它才算"靠近镜头"（加分机位）。取 0.45 是为了不抢正脸机位：
# 笔记本自拍常在 0.15~0.30，0.45 已经是"脸占半屏"的贴脸距离。
FACE_RATIO_VERY_NEAR = 0.45
# 兼容旧名字：外部（README / 自检）可能还在引用
FRONTAL_VM_RATIO = 1.12        # 正视时"眼-嘴距离 / 眼距"的经验基线（现在只作为辅助信息）

GUIDE_NOTES = """
**为什么"5 张都差不多"会让识别变不准？**

相似度是**相对打分**——只跟人脸库里的样本比。库内样本越单一：

- 实时识别时一旦姿态/光照偏离库内那几张，本人相似度会掉到阈值以下（本人被拒）；
- 反过来，库内样本彼此太像时，同一个人不同姿态的差异被放大、不同人之间的差异被压缩，
  分数会集中在高位（例如都在 0.85~0.95），**绝对值很高却没什么区分度**。

所以：

1. **注册要覆盖机位**（本指引的 5 个核心机位：正面 / 左转 / 右转 / 抬头 / 低头），
   让库内样本"张张不同"，同一人的实时画面总能找到一张角度接近的注册图；
2. **识别时别只看绝对值**，同时看 **top1 与 top2 的分差（margin）**：
   分差很小说明"两个人（或两张图）几乎一样像"，此时判定不可靠；
3. 侧边栏把「同一人多张注册图的汇总方式」设成 **mean_topk**，
   用同一个人最像的几张取平均，比只看单张更稳。
"""


# ---------------------------------------------------------------------------
# 姿态估计
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def estimate_pose(landmarks: Optional[Sequence],
                  nose_baseline: Optional[float] = None) -> Dict[str, float]:
    """用 MTCNN 的 5 个关键点粗估头部姿态。

    关键点顺序（facenet-pytorch / MTCNN 约定）：
        [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]   —— 这里的"左右"是**画面上的左右**

    三个尺度无关的几何量（都除以双眼间距，所以跟远近无关）：

        * 左右转 yaw_ratio = (鼻尖x − 双眼中点x) / 眼距
          正视≈0；头转向画面右侧时鼻尖右移，比值为正。
        * 俯仰 nose_ratio  = (鼻尖y − 双眼中点y) / 眼距        ← 俯仰主特征
          抬头时鼻尖在画面里靠近眼睛 → 变小；低头 → 变大。
        * 辅助 vm_ratio    = |嘴角中点 − 双眼中点| / 眼距       ← 只作辅助/展示

    为什么俯仰用"鼻-眼"而不是"嘴-眼"：
        在真实注册照上实测，同一个人抬头时
            鼻-眼比  0.526 → 0.237（变化 55%）
            嘴-眼比  1.116 → 0.961（只变化 14%）
        后者信噪比太差，配上阈值就容易"抬了头却说不够"，这正是原先
        "抬头/低头很难录入"的主因。

    侧转补偿：
        侧转会让鼻-眼比虚高（实测头转 30° 时从 0.53 虚高到 0.77 以上），
        所以按 `nose_ratio − 0.6×|yaw_ratio|` 线性补偿掉，
        "抬头 + 轻微侧转"就不会被误判成侧转。
        （旧版是"侧转较大就把俯仰清零"，而人抬头必带一点侧转 → 永远录不上。）

    参数 nose_baseline：这个人自己的正视基线（一般由实时录入会话里第一张
        正面照标定）。不传则用人群经验值 FRONTAL_NOSE_RATIO。

    返回 dict：yaw_ratio / yaw_label / nose_ratio / nose_ratio_corrected /
              vm_ratio / pitch_offset / pitch_score / yaw_score /
              pitch_label / inter_eye_px / guidance
    """
    empty = {"yaw_ratio": 0.0, "yaw_label": "未知", "nose_ratio": float("nan"),
             "nose_ratio_corrected": float("nan"), "vm_ratio": float("nan"),
             "pitch_offset": 0.0, "pitch_score": 0.0, "yaw_score": 0.0,
             "pitch_label": "未知", "inter_eye_px": 0.0, "guidance": ""}
    if landmarks is None:
        return empty
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 5:
        return empty

    left_eye, right_eye, nose, mouth_l, mouth_r = pts[0], pts[1], pts[2], pts[3], pts[4]
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    if inter_eye < 1e-3:
        return empty

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_l + mouth_r) / 2.0

    yaw_ratio = float((nose[0] - eye_mid[0]) / inter_eye)
    vm_ratio = float(np.linalg.norm(mouth_mid - eye_mid) / inter_eye)
    nose_ratio = float((nose[1] - eye_mid[1]) / inter_eye)
    mouth_w = float(np.linalg.norm(mouth_r - mouth_l) / inter_eye)   # 嘴宽比（表情指标）

    # 去掉侧转带来的虚高，再用"这个人自己的基线"换算成相对俯仰
    nose_corrected = nose_ratio - YAW_NOSE_COMPENSATION * abs(yaw_ratio)
    base = FRONTAL_NOSE_RATIO
    if nose_baseline is not None and 0.20 <= float(nose_baseline) <= 0.95:
        base = float(nose_baseline)          # 本人标定的基线更准
    pitch_offset = (nose_corrected - base) / base

    # 换算成角度：yaw_ratio ≈ K·tan(θ)、pitch_offset ≈ K·sin(φ)
    yaw_deg = float(np.degrees(np.arctan(yaw_ratio / YAW_TAN_K)))
    pitch_deg = float(np.degrees(np.arcsin(
        float(np.clip(pitch_offset / PITCH_SIN_K, -1.0, 1.0)))))
    # 统一成直觉符号：yaw_deg > 0 = 头向左转；pitch_deg > 0 = 抬头、< 0 = 低头
    # （pitch_offset 的符号是"鼻眼比变大=低头"，所以这里要取反）
    pitch_deg = -pitch_deg

    # 两个"达到 1.0 就算明显"的分数：谁大听谁的（不再"侧转一票否决俯仰"）
    yaw_score = abs(yaw_ratio) / YAW_TURNED if YAW_TURNED else 0.0
    pitch_score = abs(pitch_offset) / PITCH_REL if PITCH_REL else 0.0

    if abs(yaw_ratio) < YAW_FRONT:
        yaw_label = f"正面（{yaw_deg:+.0f}°）"
    elif yaw_ratio > 0:
        yaw_label = f"头向左转 {abs(yaw_deg):.0f}°"
    else:
        yaw_label = f"头向右转 {abs(yaw_deg):.0f}°"

    if pitch_score >= 1.0 and pitch_score >= yaw_score:
        pitch_label = f"抬头 {pitch_deg:.0f}°" if pitch_deg > 0 else f"低头 {abs(pitch_deg):.0f}°"
    elif abs(pitch_offset) > PITCH_TOL and yaw_score < 1.0:
        pitch_label = f"抬头 {pitch_deg:.0f}°" if pitch_deg > 0 else f"低头 {abs(pitch_deg):.0f}°"
    elif yaw_score >= 1.0 and pitch_score < 1.0:
        pitch_label = "未判定（侧转较大）"
    else:
        pitch_label = f"平视（{pitch_deg:+.0f}°）"

    # 实时引导语：方向对了但幅度还不够时，明确告诉用户"再往哪边一点、还差几度"
    guidance = ""
    if max(yaw_score, pitch_score) < 1.0:
        if pitch_score >= 0.45 and pitch_score >= yaw_score:
            guidance = (f"再抬头约 {max(1.0, PITCH_TOL_DEG - abs(pitch_deg)):.0f}°"
                        if pitch_deg > 0 else
                        f"再低头约 {max(1.0, PITCH_TOL_DEG - abs(pitch_deg)):.0f}°")
        elif yaw_score >= 0.45:
            direction = "左" if yaw_ratio > 0 else "右"
            need = YAW_FRONT_DEG - abs(yaw_deg)
            guidance = (f"头再向{direction}转约 {max(1.0, need):.0f}°"
                        if abs(yaw_deg) < YAW_FRONT_DEG else
                        f"头再向{direction}转一点")

    return {"yaw_ratio": yaw_ratio, "yaw_label": yaw_label, "yaw_deg": yaw_deg,
            "nose_ratio": nose_ratio, "nose_ratio_corrected": nose_corrected,
            "vm_ratio": vm_ratio, "mouth_w": mouth_w, "pitch_offset": pitch_offset,
            "pitch_deg": pitch_deg, "pitch_score": pitch_score, "yaw_score": yaw_score,
            "pitch_label": pitch_label, "inter_eye_px": inter_eye,
            "guidance": guidance}


# ---------------------------------------------------------------------------
# 参考正脸照：只用来提供"这个人自己的俯仰基线"
# ---------------------------------------------------------------------------
def appearance_of(record: Dict[str, object]) -> Dict[str, float]:
    """从一次 assess_quality 结果里取出"基准信息"（目前只有俯仰基线）。

    带的是 `nose_ratio_corrected` —— 作为**这个人自己的俯仰基线**，
    比用人群常数准得多（不同人脸型、不同摄像头高度都会让鼻-眼比整体偏移）。
    """
    pose = record.get("pose") or {}
    return {
        "nose_ratio_corrected": float(pose.get("nose_ratio_corrected", float("nan"))
                                      or float("nan")),
    }


def _sharpness(gray: np.ndarray) -> float:
    """清晰度：优先用 OpenCV 的 Laplacian 方差，没装 cv2 就用 numpy 梯度方差兜底。"""
    try:
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gy, gx = np.gradient(gray.astype(np.float32))
        return float((gx ** 2 + gy ** 2).var())


def _to_pil(image) -> Image.Image:
    """把 PIL / ndarray / 路径统一转成 RGB 的 PIL.Image。"""
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")
    return Image.open(image).convert("RGB")


def _resolve_nose_baseline(nose_baseline: Optional[float],
                           reference: Optional[Dict[str, float]]) -> Optional[float]:
    """决定这次评估用哪个"正视鼻-眼比"基线。

    优先级：显式传入的本人基线 > 参考正面照里的基线 > 人群常数（由 estimate_pose 兜底）。

    为什么要把参考正面照里的基线也用上：鼻-眼比受脸型、摄像头高度影响很大
    （实测同一批照片里正脸是 0.59~0.65，而侧脸只有 0.49~0.57），
    用"本人正脸照"当基线，抬头/低头的角度才算得准。
    """
    if nose_baseline is not None:
        try:
            return float(nose_baseline)
        except (TypeError, ValueError):
            pass
    if reference:
        value = reference.get("nose_ratio_corrected")
        try:
            base = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if base == base and 0.20 <= base <= 0.95:      # 非 NaN 且在合理范围
            return base
    return None


def assess_quality(image, face, nose_baseline: Optional[float] = None,
                   reference: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """评估一张照片的质量与姿态，返回可直接展示的字典。

    输入：
        image : 原图（PIL / ndarray / 路径）
        face  : face_detector.Face（需要有 .box 和 .landmarks）
        nose_baseline : 可选，这个人自己的"正视鼻-眼比"基线
                        （实时录入时由会话自动标定，能进一步提高抬头/低头判定准确率）
        reference     : 可选，本次采集的"参考正脸照"信息（见 appearance_of）。
                        它的作用是提供**这个人自己的俯仰基线**，让抬头/低头的角度更准。
    返回：
        {
          "pose": {...}, "yaw_label", "pitch_label",
          "size_ratio", "brightness", "sharpness", "flags": [...],
          "score": 0~100, "slot": 机位 key, "slot_title": 机位名
        }
    """
    pil = _to_pil(image)
    width, height = pil.size
    box = np.asarray(face.box, dtype=np.float32)
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    size_ratio = box_area / float(width * height) if width and height else 0.0

    # 只取人脸区域做质量统计，避免背景干扰
    x1 = int(max(0, min(box[0], width - 1)))
    y1 = int(max(0, min(box[1], height - 1)))
    x2 = int(max(x1 + 1, min(box[2], width)))
    y2 = int(max(y1 + 1, min(box[3], height)))
    crop = np.asarray(pil.crop((x1, y1, x2, y2)).convert("L"), dtype=np.float32)
    brightness = float(crop.mean()) if crop.size else 0.0
    contrast = float(crop.std()) if crop.size else 0.0
    sharpness = _sharpness(crop) if crop.size else 0.0

    pose = estimate_pose(getattr(face, "landmarks", None),
                         nose_baseline=_resolve_nose_baseline(nose_baseline, reference))

    flags: List[str] = []
    score = 100.0

    if size_ratio < FACE_RATIO_TOO_FAR:
        flags.append("人脸偏小：离镜头近一点，或在侧边栏把「最小人脸尺寸」调小")
        score -= 20
    if brightness < 55:
        flags.append("画面偏暗：补光或换个亮一点的位置（暗光下特征会明显变差）")
        score -= 12
    elif brightness > 205:
        flags.append("画面过亮（可能过曝）：避开直射光")
        score -= 12
    if contrast < 22:
        flags.append("对比度偏低：脸和背景太接近，换个背景更好")
        score -= 8
    if sharpness < 30:
        flags.append("不够清晰（可能拍糊了）：端稳手机/坐稳后再拍")
        score -= 12
    if abs(pose["yaw_ratio"]) > YAW_TOO_MUCH:
        flags.append(f"侧转过大（{pose['yaw_label']}）：超过约 45° 时关键点会漂，建议小一点")
        score -= 12

    slot = classify_slot(pose, size_ratio)

    return {
        "pose": pose,
        "yaw_label": pose["yaw_label"],
        "pitch_label": pose["pitch_label"],
        "size_ratio": size_ratio,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "flags": flags,
        "score": float(max(0.0, min(100.0, score))),
        "slot": slot,
        "slot_title": PLAN_BY_KEY.get(slot, {}).get("title", "未分类"),
    }


def slot_window(key: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """取某个机位的角度窗口：(yaw 区间(绝对值), pitch 区间(绝对值), 容忍度)。"""
    item = PLAN_BY_KEY.get(key)
    if not item or "yaw" not in item:
        return None
    return (tuple(item["yaw"]), tuple(item["pitch"]), float(item.get("tol", 4.0)))  # type: ignore[arg-type]


def slot_match(key: str, yaw_deg: float, pitch_deg: float) -> Tuple[bool, float]:
    """判断当前角度是否落在某个机位的窗口（含容忍度）内。

    返回 (是否命中, 距离窗口的"还差多少度")。距离 0 表示正好在窗口里。
    """
    window = slot_window(key)
    if window is None:
        return False, 0.0
    (y_min, y_max), (p_min, p_max), tol = window
    ay, ap = abs(yaw_deg), abs(pitch_deg)

    # 侧转方向的符号要求
    if key.endswith("_left") and yaw_deg <= 0:
        return False, abs(yaw_deg) + y_min
    if key.endswith("_right") and yaw_deg >= 0:
        return False, abs(yaw_deg) + y_min
    # 抬头/低头的符号要求（pitch_deg: 正=抬头）
    if key == "up" and pitch_deg <= 0:
        return False, abs(pitch_deg) + p_min
    if key == "down" and pitch_deg >= 0:
        return False, abs(pitch_deg) + p_min

    d_yaw = max(0.0, y_min - ay, ay - y_max)
    d_pitch = max(0.0, p_min - ap, ap - p_max)
    ok = d_yaw <= tol and d_pitch <= tol
    return ok, max(d_yaw, d_pitch)


def classify_slot(pose: Dict[str, float], size_ratio: float) -> str:
    """把一张照片归到「7 机位黄金标准」里的某个机位（按**主导角度**判定）。

    黄金标准里每个机位就是一个主导角度，所以判定也照这个来，极其直白：

        |yaw| ≥ 14.5°  → 左/右半侧（<37.5°）或左/右大侧（≥37.5°）
        |pitch| ≥ 11°  → 抬头 / 低头（此时侧转已经很小，两者不会互相干扰）
        否则            → 正脸（贴脸时归"靠近镜头"加分机位）

    为什么侧转不要求俯仰也对：人转头时下巴很难完全保持水平，
    如果侧转机位还要求"俯仰在 ±12° 内"，用户会永远凑不齐——
    偏离多少由 angle_advice() 提示（例如"顺手把视线放平"）。
    """
    yaw_deg = float(pose.get("yaw_deg", 0.0) or 0.0)
    pitch_deg = float(pose.get("pitch_deg", 0.0) or 0.0)
    ay, ap = abs(yaw_deg), abs(pitch_deg)

    # ---- 1) 侧转机位（主导角度 = 左右转）----
    if ay >= YAW_HALF_DEG - 8.0:                      # ≈14.5° 起算半侧
        size = "half" if ay < (YAW_HALF_DEG + YAW_BIG_DEG) / 2.0 else "big"
        return f"{size}_{'left' if yaw_deg > 0 else 'right'}"

    # ---- 2) 俯仰机位（主导角度 = 抬头/低头；此时侧转 <14.5°，耦合很小）----
    if ap >= PITCH_TOL_DEG + 3.0:                     # ≈11° 起算轻微抬头/低头
        return "up" if pitch_deg > 0 else "down"

    # ---- 3) 平视：贴脸才算"靠近镜头"（加分机位），否则就是正脸 ----
    if size_ratio >= FACE_RATIO_VERY_NEAR:
        return "near"
    return "front"


def angle_advice(pose: Dict[str, float], target_key: Optional[str]) -> str:
    """根据当前角度和目标机位，给出"再转/再抬几度"的中文提示。

    这是"黄金标准"能不能落地的关键：用户看不到角度，程序得告诉他往哪边、还差多少。
    """
    yaw_deg = float(pose.get("yaw_deg", 0.0) or 0.0)
    pitch_deg = float(pose.get("pitch_deg", 0.0) or 0.0)
    window = slot_window(target_key) if target_key else None

    if window is None:
        return ""
    (y_min, y_max), (p_min, p_max), tol = window

    # 正脸：只要别乱转就行
    if target_key == "front":
        if abs(yaw_deg) <= y_max + tol and abs(pitch_deg) <= p_max + tol:
            return "保持住，这就是正脸 ✅"
        if abs(yaw_deg) > abs(pitch_deg):
            return (f"头回正一点（现在偏了 {abs(yaw_deg):.0f}°，"
                    f"{'左' if yaw_deg > 0 else '右'}偏）")
        return (f"视线回平（现在{'抬头' if pitch_deg > 0 else '低头'} "
                f"{abs(pitch_deg):.0f}°）")

    # 侧转类机位
    if target_key in ("half_left", "half_right", "big_left", "big_right"):
        want_left = target_key.endswith("_left")
        a = abs(yaw_deg)
        if (yaw_deg > 0) != want_left and a > 3:
            return (f"方向反了：这是{'左' if yaw_deg > 0 else '右'}偏 {a:.0f}°，"
                    f"要的是向{'左' if want_left else '右'}转")
        if a < y_min:
            return (f"再向{'左' if want_left else '右'}转约 {y_min - a:.0f}°"
                    f"（现在 {a:.0f}°，目标 {y_min:.0f}°~{y_max:.0f}°）")
        if a > y_max + tol:
            return (f"转少一点：现在 {a:.0f}°，目标 {y_min:.0f}°~{y_max:.0f}°")
        if abs(pitch_deg) > p_max + tol:
            return (f"顺手把视线放平（现在{'抬头' if pitch_deg > 0 else '低头'} "
                    f"{abs(pitch_deg):.0f}°）")
        return f"角度刚好 ✅（{a:.0f}°）"

    # 抬头 / 低头机位
    if target_key in ("up", "down"):
        want_up = target_key == "up"
        a = abs(pitch_deg)
        if (pitch_deg > 0) != want_up and a > 3:
            return (f"方向反了：这是{'抬头' if pitch_deg > 0 else '低头'} {a:.0f}°，"
                    f"要的是{'抬头' if want_up else '低头'}")
        if a < p_min:
            return (f"再{'抬' if want_up else '低'}头约 {p_min - a:.0f}°"
                    f"（现在 {a:.0f}°，目标 {p_min:.0f}°~{p_max:.0f}°）")
        if a > p_max + tol:
            return (f"{'抬' if want_up else '低'}得少一点：现在 {a:.0f}°，"
                    f"目标 {p_min:.0f}°~{p_max:.0f}°")
        if abs(yaw_deg) > y_max + tol:
            return f"顺手把头回正（现在偏了 {abs(yaw_deg):.0f}°）"
        return f"角度刚好 ✅（{a:.0f}°）"

    return ""


# ---------------------------------------------------------------------------
# 覆盖度统计
# ---------------------------------------------------------------------------
def coverage(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """根据已采集照片的分析结果，统计机位覆盖情况并给出下一步建议。

    records: assess_quality() 返回的字典列表（按采集顺序）
    """
    used = [str(r.get("slot", "")) for r in records]
    filled_core = [k for k in REQUIRED_KEYS if k in used]
    missing_core = [k for k in REQUIRED_KEYS if k not in filled_core]
    filled_extra = [k for k in OPTIONAL_KEYS if k in used]

    # 下一步建议：先补核心机位，再考虑加分机位
    if missing_core:
        next_key = missing_core[0]
        advice = f"请拍：**{PLAN_BY_KEY[next_key]['title']}** —— {PLAN_BY_KEY[next_key]['hint']}"
    else:
        next_extra = [k for k in OPTIONAL_KEYS if k not in filled_extra]
        if next_extra:
            next_key = next_extra[0]
            advice = (f"核心机位已齐 ✅ 想更稳可以再补：**{PLAN_BY_KEY[next_key]['title']}**"
                      f" —— {PLAN_BY_KEY[next_key]['hint']}"
                      f"\n\n（这是**加分项**，不补也能直接注册。）")
        else:
            next_key = None
            advice = "所有机位都已覆盖 ✅ 可以开始注册了"

    return {
        "used": used,
        "filled_core": filled_core,
        "missing_core": missing_core,
        "filled_extra": filled_extra,
        "core_ratio": len(filled_core) / float(len(REQUIRED_KEYS)) if REQUIRED_KEYS else 0.0,
        "next_key": next_key,
        "advice": advice,
        "complete": not missing_core,
    }


def duplicate_pose(pose: Dict[str, float], records: Sequence[Dict[str, object]],
                   yaw_tol_deg: float = 8.0, pitch_tol_deg: float = 8.0) -> bool:
    """判断新照片的姿态是否与已采集的某张"几乎一样"（提醒用户换机位）。

    容差直接用**角度**（8°），这样和 7 机位标准（15°~30°、45°~60° 这些窗口）同一套语言：
    两个机位之间至少差 15°，所以 8° 之内就算"同一张"是合理的。
    """
    yaw_deg = float(pose.get("yaw_deg", 0.0) or 0.0)
    pitch_deg = float(pose.get("pitch_deg", 0.0) or 0.0)
    for rec in records:
        old = rec.get("pose") or {}
        old_yaw = float(old.get("yaw_deg", 0.0) or 0.0)
        old_pitch = float(old.get("pitch_deg", 0.0) or 0.0)
        if (abs(yaw_deg - old_yaw) <= yaw_tol_deg
                and abs(pitch_deg - old_pitch) <= pitch_tol_deg):
            return True
    return False


def first_front_reference(items: Sequence[Tuple[object, object]],
                          ) -> Optional[Dict[str, float]]:
    """从一批 (原图, face) 里找出第一张"平视正面照"，返回它的外观基线。

    用于注册一批图片时先定出"参考正脸照"——它的鼻-眼比会作为**这个人的俯仰基线**，
    这样"轻微抬头/低头"的角度才算得准（人群常数对个别人会偏大偏小）。
    优先用真正的"正面"机位；没有的话退而用"靠近镜头"那张（同样是平视姿态）。
    返回 None 表示这批图里没有可用的平视照（那就退化成用人群常数）。
    """
    fallback: Optional[Dict[str, float]] = None
    for image, face in items:
        if face is None:
            continue
        try:
            analysis = assess_quality(image, face)
        except Exception:
            continue
        if analysis.get("slot") == "front":
            return appearance_of(analysis)
        if fallback is None and analysis.get("slot") == "near":
            fallback = appearance_of(analysis)
    return fallback


def plan_table(records: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    """生成"采集计划进度表"，供界面直接渲染。"""
    used = [str(r.get("slot", "")) for r in records]
    table: List[Dict[str, str]] = []
    for item in PLAN:
        if item["key"] in used:
            status = "✅ 已覆盖"
        elif item["group"] == "core":
            status = "⬜ 待采集（核心）"
        else:
            status = "⬜ 待采集（加分）"
        table.append({"机位": f"{item['title']}", "状态": status, "拍法": item["hint"]})
    return table


def summary(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """汇总若干张照片的整体情况（界面上给一句总评 + 常见问题计数）。"""
    if not records:
        return {"count": 0, "avg_score": 0.0, "problem_counts": {}, "text": "还没有照片"}

    problem_counts: Dict[str, int] = {}
    for rec in records:
        for flag in rec.get("flags", []) or []:
            key = str(flag).split("：")[0]
            problem_counts[key] = problem_counts.get(key, 0) + 1

    avg = float(np.mean([float(r.get("score", 0.0)) for r in records]))
    cov = coverage(records)
    if cov["complete"]:
        text = (f"共 {len(records)} 张，平均质量 {avg:.0f} 分；"
                f"5 个核心机位已覆盖 ✅")
    else:
        text = (f"共 {len(records)} 张，平均质量 {avg:.0f} 分；"
                f"还缺 {len(cov['missing_core'])} 个核心机位："
                + "、".join(PLAN_BY_KEY[k]["title"] for k in cov["missing_core"]))
    return {"count": len(records), "avg_score": avg,
            "problem_counts": problem_counts, "text": text}


# ---------------------------------------------------------------------------
# 自检/演示用的合成关键点（与真实注册照实测的几何一致，供 __main__ 与 self_test 共用）
# ---------------------------------------------------------------------------
DEMO_EYE_Y = 190.0
DEMO_INTER_EYE = 140.0


def demo_landmarks(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """按实测几何生成一组 MTCNN 风格关键点（眼距 140，尺度与真实照片同量级）。

    参数直接就是**角度**（与 7 机位黄金标准的窗口一致）：
        yaw_deg   : 左右转角，正数 = 头向左转（画面里鼻尖偏向右侧），负数 = 向右转
        pitch_deg : 俯仰角，正数 = 抬头、负数 = 低头

    换算关系（与 estimate_pose 反着来，保证"给 22° 就判出 22°"）：
        nose_dx = YAW_TAN_K · tan(yaw_deg)
        鼻眼比相对基线的变化 = -PITCH_SIN_K · sin(pitch_deg)
    真实照片实测（见 README）：正视鼻-眼比 0.52~0.65、低头照 0.780；
    侧转会让鼻-眼比虚高约 0.6×|yaw|（estimate_pose 里做了同样的补偿）。
    """
    nose_dx = YAW_TAN_K * float(np.tan(np.radians(yaw_deg)))
    # 注意：pitch_offset 是"相对基线的变化"，所以这里要乘上基线才是鼻-眼比的绝对变化量
    nose_delta = (-FRONTAL_NOSE_RATIO * PITCH_SIN_K
                  * float(np.sin(np.radians(pitch_deg))))   # 抬头 → 鼻眼比变小
    eye_l = (320.0 - DEMO_INTER_EYE / 2, DEMO_EYE_Y)
    eye_r = (320.0 + DEMO_INTER_EYE / 2, DEMO_EYE_Y)
    # 先加上侧转补偿量，这样 estimate_pose 扣掉补偿后正好得到想要的俯仰
    nose_y = DEMO_EYE_Y + (FRONTAL_NOSE_RATIO + nose_delta
                           + YAW_NOSE_COMPENSATION * abs(nose_dx)) * DEMO_INTER_EYE
    mouth_y = DEMO_EYE_Y + (FRONTAL_VM_RATIO + 0.44 * nose_delta) * DEMO_INTER_EYE
    nose = (320.0 + nose_dx * DEMO_INTER_EYE, nose_y)
    mouth_l = (270.0 + 0.43 * nose_dx * DEMO_INTER_EYE, mouth_y)
    mouth_r = (370.0 + 0.43 * nose_dx * DEMO_INTER_EYE, mouth_y)
    return np.asarray([eye_l, eye_r, nose, mouth_l, mouth_r], dtype=np.float32)


# 机位自检用例：(参数, 人脸占比, 期望机位) —— 就是用户给的 7 机位黄金标准。
# "抬头+轻微侧转"专门覆盖旧版会误判的情况（以前侧转会把俯仰清零）。
DEMO_CASES: Dict[str, tuple] = {
    "正脸 0°": (dict(yaw_deg=0.0, pitch_deg=0.0), 0.10, "front"),
    "左半侧 22°": (dict(yaw_deg=+22.0), 0.10, "half_left"),
    "右半侧 22°": (dict(yaw_deg=-22.0), 0.10, "half_right"),
    "轻微抬头 12°": (dict(pitch_deg=+12.0), 0.10, "up"),
    "轻微低头 12°": (dict(pitch_deg=-12.0), 0.10, "down"),
    "左大侧 52°": (dict(yaw_deg=+52.0), 0.10, "big_left"),
    "右大侧 52°": (dict(yaw_deg=-52.0), 0.10, "big_right"),
    "抬头12°+侧转8°": (dict(yaw_deg=+8.0, pitch_deg=+12.0), 0.10, "up"),
    "贴脸（脸占半屏）": (dict(), 0.55, "near"),
}


# ---------------------------------------------------------------------------
# 模块自测：python enrollment_guide.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ok = True
    for name, (kwargs, size_ratio, expect) in DEMO_CASES.items():
        pose = estimate_pose(demo_landmarks(**kwargs))
        slot = classify_slot(pose, size_ratio=size_ratio)
        flag = "✓" if slot == expect else "✗"
        if slot != expect:
            ok = False
        print(f"  {flag} {name:16s} 实测 yaw={pose['yaw_deg']:+6.1f}° "
              f"pitch={pose['pitch_deg']:+5.1f}° -> 机位 {slot}（期望 {expect}）")

    # 真实注册照复核（如果本地有的话）
    import glob

    files = sorted(glob.glob("registered_faces/*/*.png"))
    if files:
        print(f"\n  本地有 {len(files)} 张真实注册小图，按 app 的「两遍法」复核机位判定：")
        try:
            from PIL import Image as _Image
            from face_detector import FaceDetector as _FD

            det = _FD(device="cpu", align=False, keep_all=True)
            pairs = []
            for path in files:
                img = _Image.open(path).convert("RGB")
                faces = det.detect(img)
                pairs.append((path, img, faces[0] if faces else None))
            # 第一遍找参考正脸（和 app.py 的做法一致），第二遍带参考做判定
            ref = first_front_reference([(img, face) for _p, img, face in pairs])
            for path, img, face in pairs:
                if face is None:
                    continue
                rec = assess_quality(img, face, reference=ref)
                pose = rec["pose"]
                print(f"    {path.split(chr(92))[-1]}: 实测 yaw={pose['yaw_deg']:+6.1f}° "
                      f"pitch={pose['pitch_deg']:+5.1f}° -> {rec['slot_title']}"
                      f"　建议：{angle_advice(pose, rec['slot'])}")
        except Exception as err:
            print(f"    复核失败：{err}")

    # 覆盖度统计
    fake_records = [{"slot": "front"}, {"slot": "front"}, {"slot": "half_left"}]
    cov = coverage(fake_records)
    print(f"\n  覆盖：已覆盖={cov['filled_core']} 缺失={cov['missing_core']} "
          f"比例={cov['core_ratio']:.0%}")
    print(f"  建议：{cov['advice'][:70]}…")
    print(f"  重复姿态检测（与已有正面照重复）："
          f"{duplicate_pose({'yaw_ratio': 0.0, 'pitch_offset': 0.0}, fake_records)}")
    print(f"  汇总：{summary(fake_records)['text']}")
    print("  模块自测" + ("通过 ✅" if ok else "有不符合预期的项 ❌"))
