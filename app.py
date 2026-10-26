# -*- coding: utf-8 -*-
"""
app.py
======
基于预训练模型的人脸识别 Demo —— Streamlit 主界面（项目入口）。

整体流程（现场答辩可以照着这张图讲）：

    输入图片（上传 / 摄像头拍照）
        │
        ▼
    [face_detector.py]  MTCNN 检测人脸框 + 5 点关键点
        │                └─► 双眼对齐 + 裁剪缩放到 160x160
        ▼
    [feature_extractor.py]  预训练骨干网络输出 512 维特征向量（并做 L2 归一化）
        │
        ▼
    [similarity.py]  与 [face_database.py] 中所有人脸特征算余弦相似度
        │                └─► 取最高分 s
        ▼
    s >= 阈值(默认 0.6)  -> 输出姓名；否则 -> 输出"陌生人"

运行方式：
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 启动前检查（放在所有第三方 import 之前，这样缺库时能给出中文提示而不是一屏 traceback）
#
# 常见错误：直接执行 `python app.py`。
# 本文件是 Streamlit 应用，必须由 Streamlit 的 Web 服务器来跑：
#     streamlit run app.py
# 直接 python 执行时没有 Streamlit 运行时，界面无从渲染，因此这里主动拦截并给出正确命令。
# ---------------------------------------------------------------------------
def _check_run_environment() -> None:
    """检查依赖是否齐全、启动方式是否正确；不满足时打印中文提示并退出。"""
    import sys

    # ---- 1. 依赖检查 ----
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "Pillow"),
        ("streamlit", "streamlit"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("facenet_pytorch", "facenet-pytorch"),
    ]
    missing = []
    for module_name, package_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)

    if missing:
        print("=" * 72)
        print(f"❌ 当前 Python（{sys.executable}）缺少依赖：" + "、".join(missing))
        print()
        print("   请先安装依赖，建议用虚拟环境（在项目根目录执行）：")
        print("       python -m venv .venv")
        print(r"       .venv\Scripts\activate          （Windows；macOS/Linux 用 source .venv/bin/activate）")
        print("       pip install -r requirements.txt")
        print()
        print("   国内网络慢可以加镜像：")
        print("       pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print()
        print("   或者直接双击项目里的 run.bat（自动建环境 + 装依赖 + 自检 + 启动）。")
        print("=" * 72)
        raise SystemExit(1)

    # ---- 2. 启动方式检查：必须由 streamlit run 启动 ----
    running_under_streamlit = True  # 判断不出来时不拦截
    try:
        from streamlit.runtime import exists as _streamlit_exists
        running_under_streamlit = bool(_streamlit_exists())
    except Exception:
        try:
            import streamlit as _st
            running_under_streamlit = bool(_st.runtime.exists())
        except Exception:
            running_under_streamlit = True

    if not running_under_streamlit:
        print("=" * 72)
        print("⚠️  启动方式不对：这是 Streamlit 应用，不能用 python app.py 直接运行。")
        print()
        print("   正确命令（在项目根目录执行）：")
        print("       streamlit run app.py")
        print()
        print("   或者双击 run.bat。")
        print("   说明：python app.py 只是执行了一遍脚本，没有启动 Web 服务器，")
        print("         因此既不会有界面，也不会有任何输出。")
        print("=" * 72)
        raise SystemExit(1)


_check_run_environment()

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from image_utils import to_rgb_pil
from ui_utils import show_image

# ---------------------------------------------------------------------------
# 页面配置必须是最先执行的 Streamlit 命令
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="基于预训练模型的人脸识别 Demo",
    page_icon="🧑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 依赖检查：缺库时给同学一个能看懂的中文提示，而不是一屏 traceback
# ---------------------------------------------------------------------------
try:
    from face_detector import FaceDetector
    from feature_extractor import FeatureExtractor
    from face_database import FaceDatabase, sanitize_name
    from similarity import (
        DEFAULT_THRESHOLD,
        find_best_threshold,
        identify,
        similarity_stats,
    )

    import enrollment_guide as guide   # 人脸采集指引（机位/质量分析）
    import live_enrollment as live_enroll     # 浏览器实时录入（WebRTC 视频流）
    import realtime  # 实时识别（加分项）；未装 streamlit-webrtc 时内部会自行降级
except ImportError as err:  # pragma: no cover
    st.error(f"❌ 缺少依赖：{err}")
    st.markdown("请在项目根目录执行下面的命令安装依赖后重新运行：")
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
APP_TITLE = "基于预训练模型的人脸识别 Demo"
DB_PATH = Path("face_db.pkl")                      # 人脸库特征文件（pickle）
FACE_DIR = Path("registered_faces")                # 注册时保存的人脸小图目录
MIN_REGISTER_IMAGES = 5                            # 作业要求：每人至少 5 张人脸

BACKBONE_LABELS = {
    "inception_resnet_v1": "InceptionResnetV1（face 预训练，推荐）",
    "resnet18": "ResNet18（ImageNet 预训练，按作业字面要求）",
}
AGGREGATE_LABELS = {
    "max": "取最大相似度（推荐）",
    "mean_topk": "取 Top-N 平均（更稳）",
}


# ---------------------------------------------------------------------------
# 模型与人脸库的加载（用 st.cache_resource 缓存，避免每次交互都重新加载模型）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_detector(device: str, min_face_size: int, margin: int,
                  align: bool, keep_all: bool) -> FaceDetector:
    """构建 MTCNN 检测器（按参数缓存），并做一次"预热"。

    预热的意义：PyTorch 的第一次前向要付"惰性初始化 + 算子选择"的钱（往往 0.5~2 秒）。
    如果第一次前向发生在实时视频流开始之后，那几秒画面就会明显卡顿。
    所以在应用启动阶段先用一张空白图跑一遍，把这笔开销提前付掉。
    预热时顺便把它登记为"实时录入共用检测器"，避免开流时再加载一次模型。
    """
    detector = FaceDetector(
        image_size=160,
        margin=int(margin),
        min_face_size=int(min_face_size),
        device=device,
        align=bool(align),
        keep_all=bool(keep_all),
    )
    try:
        import live_enrollment as live_enroll

        live_enroll.warmup_detector(detector)       # 预热 + 让实时录入复用这个实例
        live_enroll.set_shared_detector(detector)
    except Exception:
        pass                                        # 预热失败不影响正常功能
    return detector


@st.cache_resource(show_spinner=False)
def load_extractor(backbone: str, device: str) -> FeatureExtractor:
    """构建特征提取器（按骨干网络缓存）。"""
    pretrained = "vggface2" if backbone == "inception_resnet_v1" else True
    return FeatureExtractor(backbone=backbone, pretrained=pretrained, device=device)


@st.cache_resource(show_spinner=False)
def load_database(db_path: str, face_dir: str) -> FaceDatabase:
    """加载人脸库（常驻内存，增删后会自动写盘）。"""
    return FaceDatabase(db_path=Path(db_path), face_dir=Path(face_dir), auto_load=True)


# ---------------------------------------------------------------------------
# 小工具函数
# ---------------------------------------------------------------------------
def detect_and_extract(
    image,
    detector: FaceDetector,
    extractor: FeatureExtractor,
    max_faces: Optional[int] = None,
) -> Tuple[List, List[np.ndarray], np.ndarray]:
    """检测 + 裁剪 + 提特征，返回 (人脸列表, 人脸图列表, 特征矩阵(N,D))。"""
    faces, crops = detector.detect_and_crop(image, max_faces=max_faces)
    if not faces:
        return [], [], np.zeros((0, extractor.embedding_dim), dtype=np.float32)
    embeddings = extractor.extract_batch(crops)
    return faces, crops, embeddings


# ---------------------------------------------------------------------------
# 摄像头相关的公共工具
# ---------------------------------------------------------------------------
SHOT_ICONS = {"ok": "✅", "warning": "⚠️", "error": "❌"}


def image_fingerprint(image) -> str:
    """给一张图算指纹（先缩到 64x64 再取哈希）。

    用途：摄像头连拍时判断"这张和已采集的是不是完全同一张照片"，
    避免反复按快门把同一张图注册 5 遍——那样 5 条特征几乎相同，
    等于只注册了 1 张，样本多样性为零。
    """
    arr = np.asarray(to_rgb_pil(image).resize((64, 64)), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _make_record(name: str, similarity: float, threshold: float,
                 is_unknown: Optional[bool], face_count: int) -> dict:
    """构造一条识别记录（识别记录表与连拍记录表共用同一套字段）。

    两个页面都在记"某时刻、识别成了谁、相似度多少"，
    字段名必须保持一致，否则后面拼 DataFrame 会错位。
    """
    return {
        "时刻": datetime.now().strftime("%H:%M:%S"),
        "识别结果": name,
        "相似度": round(float(similarity), 4),
        "阈值": round(float(threshold), 2),
        "是否陌生人": is_unknown,
        "人脸数": face_count,
    }


def flash(level: str, message: str) -> None:
    """暂存一条提示，等下一次渲染再显示。

    摄像头拍照后需要 st.rerun() 重置组件（换 key），
    如果直接 st.warning 再 rerun，用户根本看不到，所以要先暂存。
    """
    st.session_state["flash"] = (level, message)


def show_flash() -> None:
    """显示并清空上一条暂存的提示。"""
    item = st.session_state.pop("flash", None)
    if not item:
        return
    level, message = item
    {"ok": st.success, "warning": st.warning,
     "error": st.error, "info": st.info}.get(level, st.info)(message)


def evaluate_shot(image, detector: FaceDetector, faces: Optional[List] = None) -> dict:
    """对刚拍到的一张图做"即时体检"：几张脸、置信度多少、能不能直接用于注册。

    faces 可以由调用方传入（避免重复跑一次 MTCNN）。
    """
    if faces is None:
        faces = detector.detect(image)
    count = len(faces)
    if count == 0:
        return {"count": 0, "prob": 0.0, "level": "error",
                "message": "未检测到人脸：靠近镜头、正对光源、别遮住脸，然后重拍"}
    prob = float(faces[0].prob)
    if count > 1:
        return {"count": count, "prob": prob, "level": "warning",
                "message": f"检测到 {count} 张脸，注册时只取最大的一张，建议单独入镜"}
    if prob < 0.90:
        return {"count": 1, "prob": prob, "level": "warning",
                "message": f"置信度偏低（{prob:.2f}），建议换个角度或补光再拍一张"}
    return {"count": 1, "prob": prob, "level": "ok",
            "message": f"合格（置信度 {prob:.2f}）"}


def _first_front_reference(detected: List[Tuple[str, "Image.Image", list]]):
    """从"已检测好的图片列表"里找出参考正面照的外观基线（见 enrollment_guide）。

    它的鼻-眼比会作为这个人的**俯仰基线**，让抬头/低头的角度算得更准。
    """
    return guide.first_front_reference([(image, faces[0] if faces else None)
                                        for _label, image, faces in detected])


def render_capture_guide(analyses: List[dict]) -> None:
    """渲染「采集指引」面板：机位覆盖表 + 下一步拍法 + 为什么重要。

    analyses 是 enrollment_guide.assess_quality() 的结果列表（按采集顺序）。
    """
    st.markdown("#### 📋 采集指引（按机位拍，别 5 张都一样）")
    cov = guide.coverage(analyses)
    st.progress(cov["core_ratio"],
                text=f"核心机位覆盖 {len(cov['filled_core'])}/{len(guide.REQUIRED_KEYS)}"
                     f"（7 机位黄金标准：正脸 / 左半侧 / 右半侧 / 抬头 / 低头 / 左大侧 / 右大侧）")
    if cov["complete"]:
        st.success(cov["advice"])
    else:
        st.info(cov["advice"])

    st.dataframe(pd.DataFrame(guide.plan_table(analyses)), width="stretch", hide_index=True)

    if analyses:
        summary = guide.summary(analyses)
        st.caption(summary["text"])
        if summary["problem_counts"]:
            problems = "、".join(f"{k}×{v}" for k, v in summary["problem_counts"].items())
            st.caption(f"常见问题统计：{problems}")

    with st.expander("❓ 为什么「5 张都差不多」会让实时识别不准？", expanded=False):
        st.markdown(guide.GUIDE_NOTES)


def run_recognition(image, detector: FaceDetector, extractor: FeatureExtractor,
                    db: FaceDatabase, cfg: Dict[str, object]) -> Optional[dict]:
    """对一张图跑完整识别流程并返回数据（不做渲染），供识别页与连拍模式复用。"""
    faces, crops, embeddings = detect_and_extract(
        image, detector, extractor, max_faces=int(cfg["max_faces"])
    )
    if not faces:
        return None

    names, gallery = db.get_gallery()
    results = [
        identify(
            emb, names, gallery,
            threshold=float(cfg["threshold"]),
            top_k=int(cfg["top_k"]),
            aggregate=str(cfg["aggregate"]),
            mean_topk=int(cfg["mean_topk"]),
        )
        for emb in embeddings
    ]
    labels = [f"{r.name} {r.similarity:.3f}" for r in results]
    canvas = detector.draw(image, faces, labels,
                           unknown_flags=[not r.is_known for r in results])
    return {"faces": faces, "crops": crops, "results": results, "canvas": canvas}


def init_session_state() -> None:
    """初始化 Streamlit 会话状态（摄像头采集缓存、识别记录等）。"""
    defaults = {
        # ---- 人员注册页：摄像头连拍 ----
        "reg_shots": [],         # 已采集的图片（PIL.Image），与下面几个列表一一对应
        "reg_shot_hashes": [],   # 每张图的指纹，用于"完全重复"去重
        "reg_shot_info": [],     # 每张图的即时体检结果
        "reg_analysis": [],      # 每张图的采集分析（机位/质量，来自 enrollment_guide）
        "reg_shot_seq": 0,       # 摄像头组件 key 的递增序号（换 key = 重置组件，可继续拍）
        "reg_preview": None,     # 最近一次"仅检测预览"的结果
        # ---- 人脸识别页：摄像头拍照 ----
        "rec_shot_seq": 0,       # "重新拍照"时 +1，重置摄像头组件
        "rec_history": [],       # 摄像头识别记录
        "rec_last_fp": None,     # 上一次已记录的图片指纹，避免同一次拍照被重复记入
        # ---- 实时识别页：无依赖的连拍模式 ----
        "burst_shot_seq": 0,
        "burst_last_fp": None,
        "burst_records": [],     # [(时刻, 姓名, 相似度, 是否陌生人)]
        "burst_ok": 0,           # 命中已注册人员的次数
        "burst_unknown": 0,      # 判为陌生人的次数
        "burst_last_canvas": None,   # 最近一次的识别画面（已画框）
        "burst_last_label": "",      # 最近一次的判定文字
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_reg_session() -> None:
    """清空摄像头采集缓存的 4 个一一对应列表（reg_shots / hashes / info / analysis）。"""
    st.session_state.reg_shots = []
    st.session_state.reg_shot_hashes = []
    st.session_state.reg_shot_info = []
    st.session_state.reg_analysis = []


# ===========================================================================
# 侧边栏：所有可调参数
# ===========================================================================
def render_sidebar() -> Dict[str, object]:
    """渲染侧边栏参数面板，返回参数字典。"""
    with st.sidebar:
        st.title("⚙️ 参数设置")
        st.caption("所有参数实时生效，方便现场对比不同设置的效果")

        backbone = st.selectbox(
            "特征提取骨干网络",
            options=list(BACKBONE_LABELS.keys()),
            format_func=lambda k: BACKBONE_LABELS[k],
            index=0,
            help="InceptionResnetV1 是人脸专用预训练模型，识别效果好；"
                 "ResNet18 是通用 ImageNet 预训练骨干，对应作业字面要求。"
                 "切换骨干后特征空间不同，需要重新注册人脸库。",
        )

        device = st.selectbox(
            "运行设备", options=["auto", "cpu", "cuda"], index=0,
            help="auto = 有显卡自动用显卡。CPU 也能跑，只是慢一些。",
        )

        st.divider()
        threshold = st.slider(
            "相似度阈值（低于该值判为陌生人）",
            min_value=0.0, max_value=1.0, value=float(DEFAULT_THRESHOLD), step=0.01,
            help="作业要求的可调参数。调高→更严格（本人可能被拒）；调低→更宽松（别人可能被误认）。",
        )
        top_k = st.slider("展示 Top-K 候选", min_value=1, max_value=10, value=5, step=1)

        aggregate = st.radio(
            "同一人多张注册图的汇总方式",
            options=["max", "mean_topk"],
            format_func=lambda k: AGGREGATE_LABELS[k],
            index=0,
            help="一个人注册了 5 张图就有 5 条特征，max = 取最像的那张；"
                 "mean_topk = 取最像的前 3 张平均，抗单张噪声。",
        )
        mean_topk = st.slider("mean_topk 的 N", 1, 5, 3, 1,
                              disabled=(aggregate != "mean_topk"))

        with st.expander("🔧 MTCNN 检测参数", expanded=False):
            min_face_size = st.slider("最小人脸尺寸(px)", 20, 200, 40, 5,
                                      help="小于该尺寸的人脸不检测，调小可以提高对远处人脸的召回。")
            margin = st.slider("裁剪边距(px)", 0, 80, 20, 5,
                               help="人脸框四周额外扩出的像素，留一点背景更稳。")
            align = st.toggle("使用关键点做双眼对齐", value=True,
                              help="把倾斜人脸旋转到双眼水平，能明显提升侧脸/歪头时的准确率。")
            keep_all = st.toggle("保留画面中的所有人脸", value=True)
            max_faces = st.slider("单张图最多处理几张脸", 1, 10, 3, 1)

        with st.expander("🎥 实时识别设置（加分项）", expanded=False):
            detect_interval = st.slider("每 N 帧检测一次", 1, 30, 8, 1,
                                        help="数值越大越流畅但识别更新越慢；CPU 建议 8~15。")

        st.divider()
        if st.button("🔄 重新加载人脸库", width="stretch"):
            load_database.clear()
            st.rerun()

        st.caption(
            "**依赖提示**：`streamlit-webrtc` + `av` 用于实时视频识别（可选）。\n\n"
            "首次运行会自动下载预训练权重，需要联网。"
        )

    return {
        "backbone": backbone,
        "device": device,
        "threshold": float(threshold),
        "top_k": int(top_k),
        "aggregate": aggregate,
        "mean_topk": int(mean_topk),
        "min_face_size": int(min_face_size),
        "margin": int(margin),
        "align": bool(align),
        "keep_all": bool(keep_all),
        "max_faces": int(max_faces),
        "detect_interval": int(detect_interval),
    }


# ===========================================================================
# 标签页 1：人员注册
# ===========================================================================
def _collect_upload_candidates() -> List[Tuple[str, Image.Image]]:
    """来源①：上传图片（可多选）。"""
    candidates: List[Tuple[str, Image.Image]] = []
    uploaded_files = st.file_uploader(
        f"选择人脸图片（建议同一人不同角度/表情，至少 {MIN_REGISTER_IMAGES} 张）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="reg_uploader",
    )
    for file in uploaded_files or []:
        try:
            candidates.append((file.name, to_rgb_pil(file)))
        except Exception as err:
            st.warning(f"图片 {file.name} 读取失败：{err}")
    return candidates


def _collect_camera_candidates(detector: FaceDetector) -> List[Tuple[str, Image.Image]]:
    """来源②：浏览器摄像头连拍。

    拍照即自动入库，并立刻做一次「体检 + 机位识别」，告诉用户这张属于哪个机位、
    质量几分、下一张该拍什么；不合格的当场提示。
    """
    st.caption(
        "拍照即自动存入采集列表，无需再点按钮；每张照片会立刻做一次「体检 + 机位识别」："
        "告诉你这张属于哪个机位（正面/左转/右转/抬头/低头）、质量几分、"
        "以及**下一张该拍什么**。不合格（没人脸/多张脸/太糊/太暗）的会直接提示。"
    )

    # 摄像头组件用递增的 key：取到照片后 key+1，组件即被重置，可以接着拍下一张
    shot = st.camera_input("③ 拍照（每点一次快门自动入库）",
                           key=f"reg_camera_{st.session_state.reg_shot_seq}")

    if shot is not None:
        # ---- 抓到新照片：去重 -> 体检 + 机位分析 -> 入库 -> rerun 以复位摄像头 ----
        image = to_rgb_pil(shot)
        fingerprint = image_fingerprint(image)
        if fingerprint in st.session_state.reg_shot_hashes:
            flash("warning", "这张照片与已采集的**完全重复**，已忽略；"
                             "请换个角度 / 表情 / 距离再拍，重复样本对识别没有帮助。")
        else:
            faces = detector.detect(image)
            info = evaluate_shot(image, detector, faces=faces)
            # 参考"已有的正脸照"提供本人俯仰基线
            reference = None
            for rec in st.session_state.reg_analysis:
                if rec and rec.get("slot") == "front":
                    reference = guide.appearance_of(rec)
                    break
            analysis = (guide.assess_quality(image, faces[0], reference=reference)
                        if faces else None)

            st.session_state.reg_shots.append(image)
            st.session_state.reg_shot_hashes.append(fingerprint)
            st.session_state.reg_shot_info.append(info)
            st.session_state.reg_analysis.append(analysis)

            index_now = len(st.session_state.reg_shots)
            if analysis is None:
                flash("error", f"已采集第 {index_now} 张，但{info['message']}")
            else:
                prior = [a for a in st.session_state.reg_analysis[:-1] if a]
                repeated = guide.duplicate_pose(analysis["pose"], prior)
                cov = guide.coverage([a for a in st.session_state.reg_analysis if a])

                parts = [f"已采集第 {index_now} 张：机位「{analysis['slot_title']}」，"
                         f"质量 {analysis['score']:.0f} 分"]
                if analysis["flags"]:
                    parts.append("；".join(analysis["flags"][:2]))
                if repeated:
                    parts.append("⚠️ 这张和已有照片的姿态几乎一样，换机位更有价值")
                parts.append(cov["advice"])
                level = "warning" if (analysis["flags"] or repeated) else "ok"
                flash(level, "　".join(parts))
        st.session_state.reg_shot_seq += 1
        st.rerun()

    show_flash()

    collected = len(st.session_state.reg_shots)
    if collected:
        done = min(collected / float(MIN_REGISTER_IMAGES), 1.0)
        st.progress(done, text=(
            f"已采集 {collected}/{MIN_REGISTER_IMAGES} 张"
            + ("（已达标，可以注册；建议再拍不同角度让样本更丰富）"
               if collected >= MIN_REGISTER_IMAGES else "（还差 "
               f"{MIN_REGISTER_IMAGES - collected} 张）")
        ))
        cols = st.columns(5)
        for idx, img in enumerate(list(st.session_state.reg_shots)):
            info = st.session_state.reg_shot_info[idx]
            analysis = st.session_state.reg_analysis[idx] if idx < len(st.session_state.reg_analysis) else None
            with cols[idx % 5]:
                show_image(img, caption=f"第 {idx + 1} 张")
                st.caption(f"{SHOT_ICONS.get(info['level'], '')} {info['message']}")
                if analysis is not None:
                    st.caption(f"机位：{analysis['slot_title']}｜质量 {analysis['score']:.0f} 分")
                    for flag in analysis["flags"][:1]:
                        st.caption(f"⚠️ {flag}")
                if st.button("🗑️ 删除重拍", key=f"del_shot_{idx}"):
                    # 四个列表必须同步删除，否则指纹/体检/机位分析会错位
                    st.session_state.reg_shots.pop(idx)
                    st.session_state.reg_shot_hashes.pop(idx)
                    st.session_state.reg_shot_info.pop(idx)
                    if idx < len(st.session_state.reg_analysis):
                        st.session_state.reg_analysis.pop(idx)
                    st.rerun()
        if st.button("🗑️ 清空采集列表，全部重拍"):
            _clear_reg_session()
            st.rerun()

        st.divider()
        render_capture_guide([a for a in st.session_state.reg_analysis if a])
    else:
        st.info("还没有采集到照片：点击上面的圆形快门按钮拍照。")
        st.divider()
        render_capture_guide([])

    return [(f"摄像头_{i + 1}", img)
            for i, img in enumerate(st.session_state.reg_shots)]


def _collect_webrtc_candidates(detector: FaceDetector,
                                name: str) -> List[Tuple[str, Image.Image]]:
    """来源③：浏览器 WebRTC 实时录入。"""
    # 防御：Streamlit 只重跑主脚本，不会重新 import 已加载的模块。
    # 如果开发过程中给 live_enrollment 加了新函数而进程还是旧的，
    # 直接调用会抛 AttributeError；这里改成给出"请重启"的明确提示。
    if not hasattr(live_enroll, "render_webrtc_enrollment"):
        st.error(
            "检测到 `live_enrollment` 模块是**旧版本**（Streamlit 只重跑主脚本，"
            "不会重新加载已 import 的模块）。\n\n"
            "**请重启应用**：在运行 Streamlit 的窗口按 `Ctrl+C`，再执行\n"
            "`cd D:\\face_demo` 与 `.venv\\Scripts\\python.exe -m streamlit run app.py`。"
        )
        return []
    return live_enroll.render_webrtc_enrollment(
        detector,
        guide_renderer=render_capture_guide,
        name=name,          # 没填姓名时不会保存（自动/手动都不行）
    )


def _render_register_preview(candidates: List[Tuple[str, Image.Image]],
                             detector: FaceDetector) -> None:
    """「仅检测预览」：跑一遍 MTCNN，画框并给出机位/质量分（不提取特征）。"""
    with st.spinner("MTCNN 检测中……"):
        rows, images, captions = [], [], []
        preview_analyses: List[dict] = []
        # 先检测一遍（顺便定出"参考正脸照"当俯仰基线），再带参考做机位分析
        detected: List[Tuple[str, "Image.Image", list]] = []
        for label, image in candidates:
            faces = detector.detect(image)
            detected.append((label, image, faces))
            rows.append({
                "图片": label,
                "检测到人脸数": len(faces),
                "最大人脸置信度": round(max((f.prob for f in faces), default=0.0), 4),
                "状态": "✅ 可用" if faces else "❌ 未检测到人脸",
            })
        reference = _first_front_reference(detected)
        for (label, image, faces), row in zip(detected, rows):
            if faces:
                analysis = guide.assess_quality(image, faces[0], reference=reference)
                preview_analyses.append(analysis)
                row["机位"] = analysis["slot_title"]
                row["质量分"] = round(analysis["score"])
            images.append(detector.draw(image, faces, [f"face{i + 1}" for i in range(len(faces))])
                          if faces else image)
            captions.append(label)
        st.session_state.reg_preview = (rows, images, captions)
        st.session_state["reg_preview_analyses"] = preview_analyses

    if st.session_state.reg_preview is not None:
        rows, images, captions = st.session_state.reg_preview
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        cols = st.columns(min(5, len(images)))
        for idx, (img, cap) in enumerate(zip(images, captions)):
            with cols[idx % len(cols)]:
                show_image(img, caption=cap)
        if st.session_state.get("reg_preview_analyses"):
            st.divider()
            render_capture_guide(st.session_state["reg_preview_analyses"])


def _do_register(candidates: List[Tuple[str, Image.Image]],
                 name: str, overwrite: bool, cfg: Dict[str, object],
                 detector: FaceDetector, extractor: FeatureExtractor,
                 db: FaceDatabase) -> None:
    """正式注册：检测 → 裁剪 → 提特征 → 写入人脸库。"""
    progress = st.progress(0.0, text="开始处理……")
    rows: List[dict] = []
    kept_crops: List[np.ndarray] = []
    kept_embeddings: List[np.ndarray] = []
    kept_analyses: List[dict] = []
    total = len(candidates)

    # 第一遍：检测（顺便定出"参考正脸照"当本人俯仰基线）
    detected: List[Tuple[str, "Image.Image", list]] = []
    for label, image in candidates:
        detected.append((label, image, detector.detect(image)))
    reference = _first_front_reference(detected)

    # 第二遍：裁剪 + 提特征 + 机位分析（带参考正脸照的俯仰基线）
    for idx, (label, image, faces) in enumerate(detected, start=1):
        progress.progress((idx - 1) / total, text=f"正在处理 {label}（{idx}/{total}）")
        if not faces:
            rows.append({"图片": label, "状态": "❌ 未检测到人脸，已跳过",
                         "置信度": 0.0, "备注": "换一张更清晰/更近的照片"})
            continue

        note = "✅ 使用最大的人脸"
        if len(faces) > 1:
            note = f"⚠️ 检测到 {len(faces)} 张人脸，取最大的一张"
        crop = detector.crop(image, faces[0])           # 对齐 + 裁剪到 160x160
        embedding = extractor.extract(crop)             # 512 维特征（已 L2 归一化）
        analysis = guide.assess_quality(image, faces[0], reference=reference)
        kept_crops.append(crop)
        kept_embeddings.append(embedding)
        kept_analyses.append(analysis)
        rows.append({"图片": label, "状态": "✅ 特征提取成功",
                     "置信度": round(float(faces[0].prob), 4),
                     "机位": analysis["slot_title"],
                     "质量分": round(analysis["score"]),
                     "备注": note})

    progress.progress(1.0, text="处理完成")

    # 注册前先给出采集覆盖度提示（尤其针对上传的一批同角度照片）
    if kept_analyses:
        st.divider()
        render_capture_guide(kept_analyses)

    if len(kept_embeddings) < MIN_REGISTER_IMAGES:
        st.error(
            f"❌ 只有 {len(kept_embeddings)} 张图片成功提取到人脸特征，"
            f"少于要求的 {MIN_REGISTER_IMAGES} 张，本次未注册。\n\n"
            f"建议：换成正面、清晰、光照均匀的照片，或把侧边栏的"
            f"「最小人脸尺寸」调小一点后重试。"
        )
    else:
        try:
            info = db.add_person(
                name=name,
                embeddings=np.stack(kept_embeddings, axis=0),
                face_images=kept_crops,
                overwrite=overwrite,
                backbone=cfg["backbone"],
            )
            st.success(
                f"🎉 注册成功！『{info.name}』当前共 {info.num_images} 条人脸特征，"
                f"数据已保存到 {db.db_path}"
            )
            st.balloons()
            # 注册成功后清空摄像头采集缓存（几个列表必须一起清，保持一一对应）
            _clear_reg_session()
            st.session_state["reg_preview_analyses"] = []
            cov = guide.coverage(kept_analyses)
            if not cov["complete"]:
                missing = cov["missing_core"]
                st.warning(
                    "本次覆盖的核心机位不全，缺："
                    + "、".join(guide.PLAN_BY_KEY[k]["title"] for k in missing)
                    + "。\n\n建议补拍后**追加**（不要勾选「覆盖该人已有特征」，直接再注册一次即可追加样本）——"
                    "库内样本的角度越全，实时识别越稳。"
                )
        except Exception as err:
            st.error(f"注册失败：{err}")

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_register_tab(cfg: Dict[str, object], detector: FaceDetector,
                        extractor: FeatureExtractor, db: FaceDatabase) -> None:
    """人员注册：输入姓名，采集/上传至少 5 张人脸图，提特征入库。"""
    st.subheader("🧑‍🎓 人员注册")
    st.markdown(
        f"流程：**输入姓名 → 准备至少 {MIN_REGISTER_IMAGES} 张人脸图片 → "
        "MTCNN 检测裁剪 → 提取 512 维特征 → 保存到人脸库**"
    )

    col_name, col_mode = st.columns([1, 1])
    with col_name:
        name = st.text_input("① 姓名", placeholder="例如：张三", max_chars=32)
        overwrite = st.checkbox("覆盖该人已有特征（重新注册）", value=False)
        if name and db.has_person(name):
            st.info(f"人脸库中已有『{sanitize_name(name)}』，"
                    f"当前 {len(db.get_embeddings(name))} 条特征。"
                    f"不勾选覆盖 = 追加样本。")
    with col_mode:
        source = st.radio("② 图片来源",
                          ["📁 上传图片（可多选）", "📷 摄像头拍照（连拍）",
                           "🎥 浏览器实时录入（WebRTC）"],
                          index=0, horizontal=False,
                          help="① 上传图片最省事；② 浏览器连拍（手动逐张）；"
                               "③ 浏览器实时录入：WebRTC 视频流，边拍边自动筛出覆盖各机位的帧。")

    # ---------------- 收集候选图片 ----------------
    if source.startswith("📁"):
        candidates = _collect_upload_candidates()
    elif source.startswith("📷"):
        candidates = _collect_camera_candidates(detector)
    else:
        candidates = _collect_webrtc_candidates(detector, name)
    st.divider()

    # ---------------- 检测预览 ----------------
    col_btn1, col_btn2, col_info = st.columns([1, 1, 2])
    with col_btn1:
        do_preview = st.button("🔍 仅检测预览", width="stretch",
                               disabled=len(candidates) == 0)
    with col_btn2:
        can_register = bool(name) and len(candidates) >= MIN_REGISTER_IMAGES
        do_register = st.button("🚀 提取特征并注册", type="primary",
                                width="stretch", disabled=not can_register)
    with col_info:
        if not name:
            st.warning("请先填写姓名")
        elif len(candidates) < MIN_REGISTER_IMAGES:
            st.warning(f"当前 {len(candidates)} 张图片，作业要求每人至少 "
                       f"{MIN_REGISTER_IMAGES} 张（检测成功的人脸才算数）")
        else:
            st.success(f"已准备 {len(candidates)} 张图片，可以开始注册")

    if do_preview:
        _render_register_preview(candidates, detector)

    # ---------------- 正式注册 ----------------
    if do_register:
        _do_register(candidates, name, overwrite, cfg, detector, extractor, db)

    # ---------------- 当前人脸库概览 ----------------
    with st.expander("📊 当前人脸库概览", expanded=False):
        st.write(db.stats())


# ===========================================================================
# 标签页 2：单人识别
# ===========================================================================
def render_recognize_tab(cfg: Dict[str, object], detector: FaceDetector,
                         extractor: FeatureExtractor, db: FaceDatabase) -> None:
    """人脸识别：检测 → 提特征 → 与人脸库比对 → 输出姓名和相似度 / 陌生人。"""
    st.subheader("🔍 人脸识别")
    threshold = float(cfg["threshold"])
    st.markdown(
        f"当前判定阈值：**{threshold:.2f}**（侧边栏可调）。"
        f"最高相似度 ≥ {threshold:.2f} 判为已知人员，否则输出**陌生人**。"
    )

    if db.num_people == 0:
        st.warning("人脸库还是空的，请先到「人员注册」标签页注册至少一个人。")
        return

    names, gallery = db.get_gallery()
    st.caption(f"人脸库：{db.num_people} 人 / {len(names)} 条特征 / "
               f"特征维度 {gallery.shape[1]} / 骨干网络 {db.backbone or cfg['backbone']}")

    # 骨干网络与人脸库维度不一致时给出提示（换了骨干但没重新注册）
    if db.embedding_dim is not None and db.embedding_dim != extractor.embedding_dim:
        st.error(
            f"⚠️ 人脸库特征维度（{db.embedding_dim}）与当前骨干网络输出维度"
            f"（{extractor.embedding_dim}）不一致，请重新注册人脸库。"
        )
        return

    source = st.radio("图片来源", ["📁 上传图片", "📷 摄像头拍照"],
                      index=0, horizontal=True, key="rec_source")

    image: Optional[Image.Image] = None
    if source.startswith("📁"):
        uploaded = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                    key="rec_uploader")
        if uploaded is not None:
            image = to_rgb_pil(uploaded)
    else:
        col_cam, col_btn = st.columns([3, 1])
        with col_cam:
            shot = st.camera_input(
                "拍照后自动识别",
                key=f"rec_camera_{st.session_state.rec_shot_seq}",
            )
        with col_btn:
            st.write("")
            st.write("")
            # 摄像头组件会一直保留上一张照片，给一个按钮把它清掉重新拍
            if st.button("🔄 重新拍照", width="stretch",
                         help="清空当前这张照片，重新对视摄像头拍摄"):
                st.session_state.rec_shot_seq += 1
                st.rerun()
        if shot is not None:
            image = to_rgb_pil(shot)

    if image is None:
        st.info("请上传图片或使用摄像头拍照，识别会自动开始。")
        return

    with st.spinner("正在检测人脸并提取特征……"):
        outcome = run_recognition(image, detector, extractor, db, cfg)

    if outcome is None:
        show_image(image, caption="未检测到人脸")
        st.error("❌ 没有检测到人脸。请换一张更清晰、人脸更大的图片，"
                 "或把侧边栏「最小人脸尺寸」调小后重试。")
        return

    faces = outcome["faces"]
    crops = outcome["crops"]
    results = outcome["results"]
    canvas = outcome["canvas"]
    main = results[0]     # 最大的人脸作为主结果

    # ---- 摄像头来源：把这次结果记入识别记录（按图片指纹去重，避免拖滑块时刷屏）----
    if source.startswith("📷"):
        fingerprint = image_fingerprint(image)
        if st.session_state.rec_last_fp != fingerprint:
            st.session_state.rec_last_fp = fingerprint
            st.session_state.rec_history.append(
                _make_record(main.name, main.similarity, threshold,
                             not main.is_known, len(faces))
            )

    col_img, col_res = st.columns([1.1, 1])
    with col_img:
        show_image(canvas, caption="识别结果（绿框=已识别，红框=陌生人）")

    with col_res:
        if main.is_known:
            st.success(f"✅ 识别结果：**{main.name}**\n\n"
                       f"相似度 **{main.similarity:.4f}** ≥ 阈值 {threshold:.2f}")
        else:
            st.error(f"🚫 识别结果：**陌生人**\n\n"
                     f"最高相似度 **{main.similarity:.4f}** < 阈值 {threshold:.2f}")

        # ---- 判定可靠性提示：别只看绝对值，要看"和阈值/第二名的距离" ----
        margin = main.margin
        if main.is_known and not math.isnan(margin):   # margin 为 NaN 时跳过
            if margin < 0.05:
                st.warning(
                    f"⚠️ **本次判定可能不可靠**：第一名与第二名只差 {margin:.4f}。\n\n"
                    "出现这种情况通常是**注册样本覆盖不足**（比如 5 张都是正面），"
                    "库内样本太像会让分数挤在高位、区分度下降——"
                    "这时相似度到 0.9 以上也不代表可靠。\n\n"
                    "建议：① 去「人员注册」按**采集指引**补拍左转/右转/抬头/低头各一张；"
                    "② 侧边栏把汇总方式改成 **mean_topk**；③ 库里有多个人的话，"
                    "用「人脸库管理」页的阈值分析看类内/类间分布。"
                )
            elif abs(main.similarity - threshold) < 0.03:
                st.info(
                    f"ℹ️ 相似度 {main.similarity:.4f} 正好落在阈值 {threshold:.2f} 附近，"
                    "判定处于临界状态：换个角度/光线再拍一张往往就能稳定。"
                )

        st.markdown("**Top-K 相似度排行**")
        top_df = pd.DataFrame(
            {"相似度": [round(s, 4) for _, s in main.top_k]},
            index=[n for n, _ in main.top_k],
        )
        st.bar_chart(top_df)

        with st.expander("查看判定细节", expanded=False):
            st.write(main.as_dict())
            if len(main.top_k) >= 2:
                st.caption(f"第一名与第二名分差（margin）= {main.margin:.4f}，"
                           f"分差越大说明越可靠。")

    if len(faces) > 1:
        st.markdown(f"**画面中共检测到 {len(faces)} 张人脸，逐张结果如下：**")
        table = pd.DataFrame([
            {
                "序号": idx + 1,
                "人脸框": str([round(float(v), 1) for v in face.box]),
                "置信度": round(float(face.prob), 4),
                "识别结果": r.name,
                "相似度": round(r.similarity, 4),
                "是否陌生人": (not r.is_known),
            }
            for idx, (face, r) in enumerate(zip(faces, results))
        ])
        st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("🧩 查看该图裁剪出的人脸（送进模型的输入）", expanded=False):
        cols = st.columns(min(5, len(crops)))
        for idx, crop in enumerate(crops):
            with cols[idx % len(cols)]:
                show_image(crop, caption=f"人脸 {idx + 1} 160x160")

    # ---- 摄像头识别记录（做"连续拍多张看稳定性"的演示很好用）----
    if source.startswith("📷") and st.session_state.rec_history:
        history = st.session_state.rec_history
        hit = sum(1 for row in history if not row["是否陌生人"])
        st.divider()
        st.markdown("#### 📷 摄像头识别记录")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("识别次数", len(history))
        col_b.metric("命中已注册人员", hit)
        col_c.metric("判为陌生人", len(history) - hit)
        st.dataframe(pd.DataFrame(history[-10:][::-1]),
                     width="stretch", hide_index=True)
        st.caption("同一张照片只记一次；换角度/表情重拍会新增一行，可用来观察相似度的稳定性。")
        if st.button("🧹 清空识别记录"):
            st.session_state.rec_history = []
            st.session_state.rec_last_fp = None
            st.rerun()


# ===========================================================================
# 标签页 3：人脸库管理
# ===========================================================================
def render_database_tab(cfg: Dict[str, object], db: FaceDatabase) -> None:
    """人脸库管理：查看已注册人员、删除、重命名、清空、导出、阈值分析。"""
    st.subheader("🗂️ 人脸库管理")

    stats = db.stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("注册人数", stats["注册人数"])
    col2.metric("人脸特征总数", stats["人脸特征总数"])
    col3.metric("特征维度", stats["特征维度"] or "-")
    col4.metric("文件大小(KB)", stats["文件大小(KB)"])

    if db.last_error:
        st.warning(db.last_error)
    if db.get_meta("image_save_error"):
        st.warning(str(db.get_meta("image_save_error")))

    if db.num_people == 0:
        st.info("人脸库为空。请先到「人员注册」标签页注册人员。")
        return

    people = db.list_people()
    st.markdown("#### 已注册人员列表")
    st.dataframe(
        pd.DataFrame([p.as_dict() for p in people]),
        width="stretch", hide_index=True,
    )

    # ---------------- 缩略图 ----------------
    with st.expander("🖼️ 查看已注册人脸图片", expanded=False):
        for person in people:
            st.markdown(f"**{person.name}**（{person.num_images} 张）")
            cols = st.columns(min(6, max(1, len(person.image_paths))))
            for idx, path in enumerate(person.image_paths[:12]):
                with cols[idx % len(cols)]:
                    try:
                        show_image(Image.open(path))
                    except Exception:
                        st.caption("图片缺失")

    st.divider()
    col_del, col_ren = st.columns(2)

    # ---------------- 删除 ----------------
    with col_del:
        st.markdown("#### 🗑️ 删除指定人员")
        target = st.selectbox("选择要删除的人员", options=db.names(), key="del_target")
        confirm_del = st.checkbox("我确认删除该人员的全部特征与图片", key="del_confirm")
        if st.button("删除该人员", type="primary",
                     disabled=(not confirm_del or not target)):
            if db.delete_person(target):
                st.success(f"已删除『{target}』")
                st.rerun()
            else:
                st.error("删除失败：未找到该人员")

    # ---------------- 重命名 ----------------
    with col_ren:
        st.markdown("#### ✏️ 重命名")
        old_name = st.selectbox("选择人员", options=db.names(), key="ren_old")
        new_name = st.text_input("新姓名", key="ren_new")
        if st.button("重命名", disabled=not new_name):
            try:
                if db.rename_person(old_name, new_name):
                    st.success(f"『{old_name}』已改名为『{sanitize_name(new_name)}』")
                    st.rerun()
                else:
                    st.error("重命名失败：请检查姓名")
            except Exception as err:
                st.error(f"重命名失败：{err}")

    st.divider()

    # ---------------- 导出 / 清空 ----------------
    col_exp, col_clear = st.columns(2)
    with col_exp:
        st.markdown("#### 💾 导出人脸库")
        st.download_button(
            "下载 face_db.pkl",
            data=db.export_bytes(),
            file_name="face_db.pkl",
            mime="application/octet-stream",
            help="把这个文件拷到另一台机器上，就能复用人脸库（骨干网络需一致）。",
        )
    with col_clear:
        st.markdown("#### ⚠️ 清空人脸库")
        typed = st.text_input("输入「确认清空」以启用按钮", key="clear_confirm")
        if st.button("清空全部数据", disabled=(typed.strip() != "确认清空")):
            db.clear()
            st.success("人脸库已清空")
            st.rerun()

    # ---------------- 阈值分析（加分项，写报告用） ----------------
    st.divider()
    st.markdown("#### 📈 阈值分析（基于已注册样本自动统计）")
    names, gallery = db.get_gallery()
    if len(names) < 4 or len(set(names)) < 2:
        st.caption("至少注册 2 个人、合计 4 条以上特征才能做类内/类间相似度统计。")
    else:
        stats_sim = similarity_stats(gallery, names)
        best = find_best_threshold(gallery, names)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("类内相似度均值", f"{stats_sim['类内相似度']['均值']:.4f}")
        col_b.metric("类间相似度均值", f"{stats_sim['类间相似度']['均值']:.4f}")
        col_c.metric("建议阈值（类内最小/类间最大中点）", f"{stats_sim['建议阈值']:.3f}")
        st.caption(
            f"当前设置阈值 {cfg['threshold']:.2f}；"
            f"按已注册样本扫描得到的最佳阈值约 {best['最佳阈值']:.3f}"
            f"（成对判断准确率 {best['最佳准确率'] * 100:.1f}%）。\n\n"
            "类内 = 同一个人不同照片的相似度，类间 = 不同人之间的相似度。"
            "理想情况下两者差距越大，阈值越容易选。"
        )
        detail = pd.DataFrame([
            {"对比类型": "类内（同一人）", **stats_sim["类内相似度"]},
            {"对比类型": "类间（不同人）", **stats_sim["类间相似度"]},
        ])
        st.dataframe(detail, width="stretch", hide_index=True)


# ===========================================================================
# 标签页 4：实时识别（加分项）
# ===========================================================================
def render_realtime_tab(cfg: Dict[str, object], detector: FaceDetector,
                        extractor: FeatureExtractor, db: FaceDatabase) -> None:
    st.subheader("🎥 摄像头实时人脸识别（加分项）")
    if db.num_people == 0:
        st.warning("人脸库为空，实时识别只会显示「陌生人」。建议先注册人员。")

    mode = st.radio(
        "识别模式",
        [
            "🎥 浏览器 WebRTC 视频流（最流畅）",
            "📷 连拍识别（零额外依赖，手动逐张）",
        ],
        index=0, horizontal=False,
        help="① 浏览器视频流：WebRTC 逐帧识别，最流畅（约 20 FPS），"
             "但必须用 localhost 打开页面并允许摄像头权限；\n"
             "② 连拍识别：手动按快门逐张识别并统计，不需要额外依赖。",
    )

    if mode.startswith("🎥"):
        async_proc = st.toggle(
            "在独立进程中处理视频（更流畅）", value=True,
            help="关掉后视频处理跑在服务器主进程里。如果启动时报 "
                 "pickle / multiprocessing 相关错误，把这个开关关掉再试。",
        )
        realtime.render_realtime_tab(
            db_path=DB_PATH,
            face_dir=FACE_DIR,
            backbone=str(cfg["backbone"]),
            device=str(cfg["device"]),
            threshold=float(cfg["threshold"]),
            detect_interval=int(cfg["detect_interval"]),
            min_face_size=int(cfg["min_face_size"]),
            margin=int(cfg["margin"]),
            align=bool(cfg["align"]),
            aggregate=str(cfg["aggregate"]),
            mean_topk=int(cfg["mean_topk"]),
            max_faces=int(cfg["max_faces"]),
            async_processing=bool(async_proc),
        )
    else:
        render_burst_mode(cfg, detector, extractor, db)


def render_burst_mode(cfg: Dict[str, object], detector: FaceDetector,
                      extractor: FeatureExtractor, db: FaceDatabase) -> None:
    """零额外依赖的"连拍识别"：每拍一张就识别一次，并累计命中/陌生人统计。

    它等价于"手动逐帧"的实时识别，好处是完全不需要 streamlit-webrtc / av，
    在任何装了 Streamlit 的机器上都能演示，也方便截图留证据。
    """
    if db.num_people == 0:
        st.info("先到「人员注册」页注册至少一个人，再来这里连拍。")
        return

    st.markdown(
        "**用法**：点下面的快门拍一张 → 自动识别并记录 → 再拍下一张。"
        "可以连续拍同一个人的不同角度，观察相似度是否稳定；也可以拍没注册过的人，验证陌生人拒识。"
    )

    shot = st.camera_input("连拍识别（每按一次快门识别一次）",
                           key=f"burst_camera_{st.session_state.burst_shot_seq}")

    if shot is not None:
        image = to_rgb_pil(shot)
        fingerprint = image_fingerprint(image)
        # 同一张照片只记一次（否则拖滑块、点按钮都会重复计数）
        if st.session_state.burst_last_fp != fingerprint:
            st.session_state.burst_last_fp = fingerprint
            outcome = run_recognition(image, detector, extractor, db, cfg)
            if outcome is None:
                flash("error", "这一张没有检测到人脸，请靠近镜头、正对光源后重拍。")
                st.session_state.burst_last_canvas = image
                st.session_state.burst_last_label = "未检测到人脸"
                st.session_state.burst_records.append(
                    _make_record("未检测到人脸", 0.0, cfg["threshold"], None, 0)
                )
            else:
                main = outcome["results"][0]
                st.session_state.burst_last_canvas = outcome["canvas"]
                st.session_state.burst_last_label = f"{main.name}  相似度 {main.similarity:.4f}"
                st.session_state.burst_records.append(
                    _make_record(main.name, main.similarity, cfg["threshold"],
                                 not main.is_known, len(outcome["faces"]))
                )
                if main.is_known:
                    st.session_state.burst_ok += 1
                else:
                    st.session_state.burst_unknown += 1
            st.session_state.burst_shot_seq += 1     # 复位摄像头，准备拍下一张
            st.rerun()

    show_flash()

    col_cam, col_stat = st.columns([1.1, 1])
    with col_cam:
        canvas = st.session_state.get("burst_last_canvas")
        if canvas is not None:
            show_image(canvas, caption="最近一次识别画面（绿框=已注册，红框=陌生人）")
            st.caption(f"判定：{st.session_state.get('burst_last_label', '')}")
        else:
            st.info("还没有记录：点上面的快门开始第一张。")

    with col_stat:
        ok_count = st.session_state.burst_ok
        unknown_count = st.session_state.burst_unknown
        total = ok_count + unknown_count
        st.metric("累计识别次数", total)
        col_x, col_y = st.columns(2)
        col_x.metric("命中已注册人员", ok_count)
        col_y.metric("判为陌生人", unknown_count)
        if total:
            st.caption(f"命中率 {ok_count / total * 100:.0f}%"
                       f"（当前阈值 {float(cfg['threshold']):.2f}，可在左侧调整）")

    records = st.session_state.burst_records
    if records:
        st.markdown("#### 识别记录（最近 10 次）")
        st.dataframe(pd.DataFrame(records[-10:][::-1]),
                     width="stretch", hide_index=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🧹 清空记录与统计"):
                st.session_state.burst_records = []
                st.session_state.burst_ok = 0
                st.session_state.burst_unknown = 0
                st.session_state.burst_last_fp = None
                st.rerun()
        with col_btn2:
            if st.button("📸 导出记录为 CSV"):
                csv = pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载 burst_records.csv", data=csv,
                                   file_name="burst_records.csv", mime="text/csv")


# ===========================================================================
# 标签页 5：原理与说明
# ===========================================================================
def render_about_tab(cfg: Dict[str, object], detector: FaceDetector,
                     extractor: FeatureExtractor, db: FaceDatabase) -> None:
    st.subheader("📖 原理与工程说明（答辩速查）")

    st.markdown("#### 1. 处理流程")
    st.markdown(
        """
1. **人脸检测（MTCNN）**：三级级联网络 P-Net / R-Net / O-Net，
   先粗筛候选框，再逐级过滤和精修，最后回归出 5 个关键点。
2. **对齐与裁剪**：用左右眼关键点算出倾角，把脸旋转到双眼水平，
   再按框 + 边距裁成正方形并缩放到 160×160。
3. **特征提取（预训练骨干）**：输出 512 维向量并做 L2 归一化，
   使"余弦相似度 = 向量点积"，比对变成一次矩阵乘法。
4. **相似度比对**：与库内所有特征算余弦相似度，按人汇总取最高分。
5. **阈值判定**：最高分 ≥ 阈值 → 输出姓名；否则 → 输出"陌生人"。
        """
    )

    st.markdown("#### 2. 阈值怎么选？（课程重点）")
    st.markdown(
        """
- 阈值 **偏高**：误识率（把别人认成本人）低，但漏识率（本人被拒）高；
- 阈值 **偏低**：本人容易通过，但容易被别人"冒充"成功。
- 经验值：FaceNet 系列在 LFW 类数据上常用 0.6~0.75；
  本 Demo 默认 **0.6**，可在侧边栏实时拖动观察效果。
- 「人脸库管理」页的**阈值分析**会用已注册样本统计类内/类间相似度，
  给出一个数据驱动的建议阈值，可直接写进实验报告。
        """
    )

    st.markdown("#### 3. 关于「骨干网络」的说明")
    st.markdown(
        """
作业要求写的是 *facenet-pytorch 的 InceptionResnetV1，骨干网络使用 ResNet18*，
这两者在工程实现上是两条不同的路线，本项目**都实现了**：

| 选项 | 来源 | 预训练数据 | 特点 |
| --- | --- | --- | --- |
| `inception_resnet_v1`（默认） | facenet-pytorch | VGGFace2（人脸身份） | 人脸专用，识别准确率高 |
| `resnet18` | torchvision | ImageNet（通用物体） | 严格符合"ResNet18 骨干"的字面要求，但没学过人脸身份，区分能力弱 |

两者输出都是 512 维，但**特征空间不通用**：换骨干后必须清空并重新注册人脸库，
否则相似度会完全失去意义（程序里已做了维度检查与提示）。
        """
    )

    st.markdown("#### 4. 当前运行环境")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**检测器配置**")
        st.json({
            "device": str(detector.device),
            "image_size": detector.image_size,
            "margin": detector.margin,
            "min_face_size": detector.min_face_size,
            "align": detector.align,
            "keep_all": detector.keep_all,
        })
    with col2:
        st.markdown("**特征提取器配置**")
        st.json(extractor.describe())

    st.markdown("**人脸库状态**")
    st.json(db.stats())

    st.markdown("#### 5. 文件结构")
    st.code(
        """人脸识别大作业/
├── app.py                 # Streamlit 主界面（注册 / 识别 / 管理 / 实时 / 说明）
├── face_detector.py       # MTCNN 检测 + 关键点对齐 + 裁剪（封装 FaceDetector）
├── feature_extractor.py   # InceptionResnetV1 / ResNet18 特征提取（512 维）
├── face_database.py       # 人脸库增删查 + pickle 持久化（FaceDatabase）
├── similarity.py          # 余弦相似度、阈值判定、阈值评估工具
├── realtime.py            # 加分项：streamlit-webrtc 摄像头实时识别
├── self_test.py           # 一键自检脚本（环境 / 数学 / 模型 / 端到端）
├── requirements.txt       # 依赖清单
├── README.md              # 安装、运行与说明文档
├── run.bat                # Windows 一键安装并启动脚本
├── face_db.pkl            # 【自动生成】人脸库特征文件
└── registered_faces/      # 【自动生成】注册时保存的人脸小图""",
        language="text",
    )

    st.markdown("#### 6. 常见问题")
    st.markdown(
        """
- **权重下载失败**：首次运行需联网下载预训练权重。可手动下载后放到
  `~/.cache/torch/checkpoints`（facenet-pytorch 的 `*.pt`）或
  `~/.cache/torch/hub/checkpoints`（torchvision 的 `resnet18-*.pth`）。
- **CPU 很慢**：MTCNN 检测最耗时，可把「最小人脸尺寸」调大、或改用 GPU。
- **摄像头打不开**：浏览器需要授予摄像头权限；非本机访问时部分浏览器要求 HTTPS。
- **中文姓名显示成方块**：说明系统缺少中文字体，代码会自动回退到 Pillow 默认字体，
  可把字体文件路径加到 `face_detector._CJK_FONT_CANDIDATES`。
        """
    )


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    init_session_state()
    cfg = render_sidebar()

    st.title("🧑 " + APP_TITLE)
    st.caption(
        "人工智能视觉课程大作业 · 选题二：基于预训练模型的人脸识别 Demo ｜ "
        "MTCNN + InceptionResnetV1(ResNet18) + 余弦相似度 + Streamlit"
    )

    # ---- 加载模型（出错时给出中文提示并停止，避免整页报错） ----
    try:
        detector = load_detector(cfg["device"], cfg["min_face_size"],
                                 cfg["margin"], cfg["align"], cfg["keep_all"])
    except Exception as err:
        st.error(f"❌ MTCNN 检测器初始化失败：{err}")
        st.stop()

    try:
        extractor = load_extractor(cfg["backbone"], cfg["device"])
    except Exception as err:
        st.error(f"❌ 特征提取模型初始化失败：\n\n{err}")
        st.stop()

    db = load_database(str(DB_PATH), str(FACE_DIR))

    # ---- 顶部状态栏 ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("已注册人数", db.num_people)
    col2.metric("特征库条目", db.num_embeddings)
    col3.metric("判定阈值", f"{cfg['threshold']:.2f}")
    col4.metric("运行设备", str(extractor.device))

    tab_reg, tab_rec, tab_db, tab_live, tab_about = st.tabs(
        ["🧑‍🎓 人员注册", "🔍 人脸识别", "🗂️ 人脸库管理", "🎥 实时识别（加分项）", "📖 原理与说明"]
    )
    with tab_reg:
        render_register_tab(cfg, detector, extractor, db)
    with tab_rec:
        render_recognize_tab(cfg, detector, extractor, db)
    with tab_db:
        render_database_tab(cfg, db)
    with tab_live:
        render_realtime_tab(cfg, detector, extractor, db)
    with tab_about:
        render_about_tab(cfg, detector, extractor, db)

    st.divider()
    st.caption(
        "运行命令：`streamlit run app.py` ｜ 自检命令：`python self_test.py` ｜ "
        f"人脸库文件：`{DB_PATH.resolve()}`"
    )


if __name__ == "__main__":
    main()
