# -*- coding: utf-8 -*-
"""
self_test.py
============
项目一键自检脚本：在向老师演示之前先跑一遍，确认环境与代码都正常。

检查内容（分 5 组，越靠后越"重"）：
    1. 环境检查    ：Python 版本、torch / torchvision / facenet-pytorch / streamlit 是否安装、是否有 GPU
    2. 数学与判定  ：余弦相似度的正确性、阈值判定的边界（0.6 判定为已知 / 陌生人）
    3. 人脸库逻辑  ：注册、追加、查询、删除、pickle 存盘再读回是否一致
    4. 模型前向    ：MTCNN / 骨干网络能否正常构建，特征维度是否为 512、是否已 L2 归一化
    5. 端到端流程  ：用一张真实人脸图做"注册 → 识别"，验证整条流水线

用法::

    python self_test.py                      # 自动检查（没有真实图片时会跳过第 5 组）
    python self_test.py --image face.jpg     # 指定一张真实人脸图做端到端测试
    python self_test.py --backbone resnet18  # 指定骨干网络
    python self_test.py --skip-model         # 只检查不需要深度学习框架的部分

退出码：0 = 全部通过（含跳过）；1 = 有检查失败。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 控制台输出工具（Windows 控制台默认 GBK，这里强制 UTF-8 避免中文乱码）
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

RESULTS: List[Tuple[str, str, str]] = []   # (级别, 项目, 说明)
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def record(level: str, name: str, detail: str = "") -> None:
    icon = {PASS: "✅", FAIL: "❌", SKIP: "⚠️ "}[level]
    RESULTS.append((level, name, detail))
    print(f"{icon} {name}" + (f" —— {detail}" if detail else ""))


def check(condition: bool, name: str, ok_detail: str = "", fail_detail: str = "") -> bool:
    record(PASS if condition else FAIL, name, ok_detail if condition else fail_detail)
    return bool(condition)


def make_workdir(prefix: str) -> Path:
    """建一个临时工作目录。

    优先建在当前项目目录下（用 mkdir 而不是 tempfile.mkdtemp），
    因为个别受限环境/受保护目录下，tempfile 建出来的目录可能不可写；
    如果当前目录也不可写，再退回系统临时目录。
    """
    work = Path(f".selftest_{prefix}")
    try:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        # 真正写一个文件验证可写性
        probe = work / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return work
    except Exception:
        return Path(tempfile.mkdtemp(prefix=f"face_selftest_{prefix}_"))


# ---------------------------------------------------------------------------
# 1. 环境检查
# ---------------------------------------------------------------------------
def test_environment() -> dict:
    print("\n=========== 1. 环境检查 ===========")
    info = {"python": sys.version.split()[0],
            "torch": None, "torchvision": None, "facenet_pytorch": None,
            "streamlit": None, "cuda": False}

    version = tuple(int(v) for v in sys.version.split()[0].split(".")[:2])
    check(version >= (3, 10), "Python 版本 >= 3.10", f"当前 {sys.version.split()[0]}",
          f"当前 {sys.version.split()[0]}，请升级到 3.10+")

    # 打印解释器路径：绝大多数"明明装了却说没有这个模块"，都是装错 / 用错了解释器，
    # 所以这里必须把"到底在用哪个 python"直接显示出来（这也是排查时最容易漏掉的信息）。
    info["exe"] = sys.executable
    print(f"   解释器  : {sys.executable}")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    info["in_venv"] = in_venv
    if in_venv:
        record(PASS, "运行在虚拟环境中", f"环境目录 {sys.prefix}")
    else:
        record(SKIP, "当前解释器不在虚拟环境中",
               "若下面报缺 torch / streamlit / Pillow，多半就是用了全局 Python 而不是 .venv 里的那个；"
               "请改成执行 .venv\\Scripts\\python.exe self_test.py，或直接双击 run.bat")

    def _try_import(module_name: str, label: str) -> Optional[str]:
        try:
            module = __import__(module_name)
            ver = getattr(module, "__version__", "未知版本")
            info[module_name] = ver
            record(PASS, f"{label} 已安装", f"版本 {ver}")
            return ver
        except Exception as err:
            record(FAIL if module_name == "torch" else SKIP, f"{label} 未安装",
                   f"{err}（执行 pip install -r requirements.txt）")
            return None

    _try_import("torch", "PyTorch")
    if info["torch"]:
        _try_import("torchvision", "torchvision")
        _try_import("facenet_pytorch", "facenet-pytorch")
        try:
            import torch
            info["cuda"] = bool(torch.cuda.is_available())
            record(PASS, "设备检测",
                   f"CUDA 可用：{torch.cuda.get_device_name(0)}" if info["cuda"]
                   else "未检测到 CUDA，将使用 CPU（正常，只是慢一些）")
        except Exception as err:
            record(SKIP, "设备检测失败", str(err))
    _try_import("streamlit", "Streamlit")

    for path in ("face_detector.py", "feature_extractor.py", "face_database.py",
                 "similarity.py", "app.py"):
        check(Path(path).exists(), f"项目文件存在：{path}", fail_detail="文件缺失，请确认在项目根目录运行")
    return info


# ---------------------------------------------------------------------------
# 2. 数学与判定逻辑
# ---------------------------------------------------------------------------
def test_similarity() -> None:
    print("\n=========== 2. 相似度与阈值判定 ===========")
    from similarity import (DEFAULT_THRESHOLD, cosine_similarity, identify,
                            is_same_person, l2_normalize)

    check(abs(DEFAULT_THRESHOLD - 0.6) < 1e-9, "默认阈值为 0.6",
          f"当前默认值 {DEFAULT_THRESHOLD}")

    v = l2_normalize(np.random.default_rng(0).normal(size=512))
    check(abs(cosine_similarity(v, v) - 1.0) < 1e-5, "自己与自己相似度 = 1",
          f"实测 {cosine_similarity(v, v):.6f}")

    a = np.zeros(512, np.float32); a[0] = 1.0
    b = np.zeros(512, np.float32); b[1] = 1.0
    check(abs(cosine_similarity(a, b)) < 1e-6, "正交向量相似度 = 0",
          f"实测 {cosine_similarity(a, b):.6f}")

    rng = np.random.default_rng(1)
    names, feats = [], []
    for person in ("甲", "乙", "丙"):
        base = l2_normalize(rng.normal(size=512))
        for _ in range(3):
            names.append(person)
            feats.append(l2_normalize(base + 0.1 * rng.normal(size=512)))
    gallery = np.stack(feats)

    hit = identify(gallery[0], names, gallery, threshold=0.6)
    check(hit.is_known and hit.name == "甲", "库内特征能被正确识别为本人",
          f"结果 {hit.name}，相似度 {hit.similarity:.4f}")

    stranger = identify(l2_normalize(rng.normal(size=512)), names, gallery, threshold=0.6)
    check((not stranger.is_known) and stranger.name == "陌生人",
          "库外特征被判定为陌生人",
          f"最高相似度 {stranger.similarity:.4f} < 阈值 0.6")

    check(is_same_person(0.6, 0.6) and not is_same_person(0.5999, 0.6),
          "阈值边界判定正确（>= 阈值算通过）")

    empty = identify(l2_normalize(rng.normal(size=512)), [], np.zeros((0, 512), np.float32))
    check((not empty.is_known) and empty.name == "陌生人", "空人脸库不会崩溃，返回陌生人")


# ---------------------------------------------------------------------------
# 3. 人脸库增删查与持久化
# ---------------------------------------------------------------------------
def test_database() -> None:
    print("\n=========== 3. 人脸库（pickle 持久化）===========")
    from face_database import FaceDatabase

    # 临时目录建在当前项目目录下，避免个别受限环境（如受沙箱保护的临时目录）无法写盘
    work = make_workdir("db")
    try:
        db = FaceDatabase(db_path=work / "test.pkl", face_dir=work / "faces")
        check(db.num_people == 0, "初始人脸库为空")

        rng = np.random.default_rng(7)
        embs = np.stack([l2n(rng.normal(size=512)) for _ in range(5)])
        imgs = [rng.integers(0, 256, (160, 160, 3), dtype=np.uint8) for _ in range(5)]
        info = db.add_person("自检用户", embs, face_images=imgs, backbone="inception_resnet_v1")
        check(info.num_images == 5, "注册 5 张人脸特征", f"库内 {info.num_images} 条")
        saved = len(list((work / "faces" / "自检用户").glob("*.png")))
        if saved == 5:
            record(PASS, "注册时保存了 5 张人脸小图")
        else:
            record(SKIP, "人脸小图未能保存（不影响特征入库，属正常降级）",
                   f"实际 {saved} 张；原因：{db.meta.get('image_save_error', '未知')}")

        db.add_person("自检用户", embs[:2])
        check(db.get_embeddings("自检用户").shape[0] == 7, "追加样本后为 7 条特征",
              f"实际 {db.get_embeddings('自检用户').shape[0]} 条")

        names, gallery = db.get_gallery()
        check(gallery.shape == (7, 512), "扁平人脸库矩阵形状为 (7, 512)", f"实际 {gallery.shape}")

        db2 = FaceDatabase(db_path=work / "test.pkl", face_dir=work / "faces")
        same = np.allclose(db2.get_embeddings("自检用户"), db.get_embeddings("自检用户"))
        check(same, "pickle 存盘后读回的数据完全一致")

        # 换骨干网络会改变维度，此时必须报错提醒"重新注册"，而不是默默算出错误结果
        try:
            db2.add_person("维度不一致", np.zeros((1, 256), np.float32))
            record(FAIL, "特征维度不一致时应报错", "没有报错，需要检查 face_database.add_person")
        except ValueError:
            record(PASS, "特征维度不一致时给出明确报错")

        check(db2.delete_person("自检用户") and db2.num_people == 0, "删除人员成功")
        check(not (work / "faces" / "自检用户").exists(), "删除人员时同步清理了人脸图片目录")
        check(db2.delete_person("不存在的人") is False, "删除不存在的人返回 False（不报错）")
    except Exception:
        record(FAIL, "人脸库自检异常", traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def l2n(vec: np.ndarray) -> np.ndarray:
    """本地小工具：L2 归一化，避免自检脚本强依赖 similarity 模块。"""
    return (vec / max(float(np.linalg.norm(vec)), 1e-10)).astype(np.float32)


# ---------------------------------------------------------------------------
# 3.5 采集指引（机位分类 / 覆盖度 / 质量控制）
# ---------------------------------------------------------------------------
def test_enrollment_guide() -> None:
    print("\n=========== 3.5 采集指引（机位分类与覆盖度）===========")
    try:
        import enrollment_guide as guide
        from PIL import Image
    except Exception as err:  # pragma: no cover
        record(SKIP, "跳过采集指引自检", str(err))
        return

    # 用量出来的真实几何合成关键点（眼距 140；正视鼻-眼比 0.52、嘴-眼比 1.12），
    # 验证 7 种机位分类（含"抬头+轻微侧转"这种旧版会误判的组合）
    cases = guide.DEMO_CASES
    wrong: List[str] = []
    for name, (kwargs, size_ratio, expect) in cases.items():
        pose = guide.estimate_pose(guide.demo_landmarks(**kwargs))
        slot = guide.classify_slot(pose, size_ratio=size_ratio)
        if slot != expect:
            wrong.append(f"{name}(期望{expect}得{slot})")
    check(not wrong, f"{len(cases)} 种机位的姿态分类全部正确",
          fail_detail="分类错误：" + "、".join(wrong))

    # 角度换算必须准：给 12° 就要读出 ~12°（这是"按角度卡标准"的前提）
    up_pose = guide.estimate_pose(guide.demo_landmarks(pitch_deg=+12.0))
    check(abs(up_pose["pitch_deg"] - 12.0) < 1.0,
          "抬头角度能被准确读出（±1°）",
          f"输入 12°，实测 {up_pose['pitch_deg']:+.1f}°")
    half_left = guide.estimate_pose(guide.demo_landmarks(yaw_deg=+22.0))
    check(abs(half_left["yaw_deg"] - 22.0) < 1.5,
          "侧转角度能被准确读出（±1.5°）",
          f"输入 22°，实测 {half_left['yaw_deg']:+.1f}°")

    # 旧版的坑：抬头时带一点侧转（超过旧阈值）就不该把俯仰丢掉
    mixed = guide.estimate_pose(guide.demo_landmarks(yaw_deg=+8.0, pitch_deg=+12.0))
    check(guide.classify_slot(mixed, size_ratio=0.10) == "up",
          "「抬头 + 轻微侧转」仍判为抬头（旧版会误判成侧转）",
          f"yaw={mixed['yaw_deg']:+.1f}° pitch={mixed['pitch_deg']:+.1f}°")

    # "还差几度"的实时引导（黄金标准能不能落地的关键）
    near_miss = guide.estimate_pose(guide.demo_landmarks(yaw_deg=+12.0))
    advice = guide.angle_advice(near_miss, "half_left")
    check("再向" in advice and "转" in advice,
          "角度没到窗口时能说出「再转几度」", advice)

    # 覆盖度：只有 2 张正面时，应报出缺 6 个核心机位（7 机位标准）
    cov = guide.coverage([{"slot": "front"}, {"slot": "front"}])
    check(cov["missing_core"] == ["half_left", "half_right", "up", "down", "big_left", "big_right"],
          "覆盖度统计能列出缺失的核心机位",
          f"缺失={cov['missing_core']}，比例={cov['core_ratio']:.0%}")
    check((not cov["complete"]) and ("请拍" in cov["advice"]),
          "未覆盖齐全时给出下一步拍法建议", cov["advice"][:40])

    # 重复姿态检测
    dup = guide.duplicate_pose({"yaw_deg": 2.0, "pitch_deg": 0.0},
                               [{"pose": {"yaw_deg": 0.0, "pitch_deg": 0.0}}])
    check(dup, "与已有照片姿态几乎相同的照片能被识别为「重复机位」")
    not_dup = guide.duplicate_pose({"yaw_deg": 30.0, "pitch_deg": 0.0},
                                   [{"pose": {"yaw_deg": 0.0, "pitch_deg": 0.0}}])
    check(not not_dup, "换了机位（侧转 30°）不会被误判为重复")

    # assess_quality 是纯函数：构造一个假 Face（只管 box / landmarks）也能跑
    class _FakeFace:
        def __init__(self, box, landmarks):
            self.box = np.asarray(box, dtype=np.float32)
            self.landmarks = np.asarray(landmarks, dtype=np.float32)

    image = Image.new("RGB", (640, 480), (128, 128, 128))
    # 关键点直接用标准模板（demo_landmarks 的参数就是角度：0° = 正脸平视）
    fake_face = _FakeFace([200, 120, 440, 380], guide.demo_landmarks(yaw_deg=0, pitch_deg=0))
    quality = guide.assess_quality(image, fake_face)
    check(quality["slot"] == "front" and 0 <= quality["score"] <= 100,
          "质量评估返回机位与 0~100 的质量分",
          f"机位={quality['slot_title']} 质量={quality['score']:.0f} "
          f"人脸占比={quality['size_ratio']:.2f} "
          f"实测 yaw={quality['pose']['yaw_deg']:+.1f}° pitch={quality['pose']['pitch_deg']:+.1f}°")
    table = guide.plan_table([{"slot": "front"}])
    check(len(table) == len(guide.PLAN) and any("✅" in row["状态"] for row in table),
          f"采集计划表有 {len(guide.PLAN)} 行且能标出已覆盖机位")


# ---------------------------------------------------------------------------
# 3.7 实时录入状态机（机位收集与覆盖）
# ---------------------------------------------------------------------------
def test_live_enrollment() -> None:
    print("\n=========== 3.7 实时录入状态机（机位收集与覆盖）===========")
    try:
        import live_enrollment as le
    except Exception as err:  # pragma: no cover
        record(SKIP, "跳过实时录入自检", str(err))
        return

    # 用假检测器精确控制每一"帧"的机位，验证"凑齐 7 个核心机位"
    class _FakeFace:
        def __init__(self, box, landmarks):
            self.box = np.asarray(box, dtype=np.float32)
            self.landmarks = np.asarray(landmarks, dtype=np.float32)
            self.prob = 0.99

    class _FakeDetector:
        def __init__(self, poses):
            self.poses = poses
            self.i = 0

        def detect(self, image):
            pts = self.poses[self.i % len(self.poses)]
            self.i += 1
            # 人脸框固定 200x200（占 640x480 的 13%），避免被判成"靠近镜头"机位
            return [_FakeFace([220, 140, 420, 340], pts)]

    # 五官模板统一用 enrollment_guide.demo_landmarks（参数直接是角度，与黄金标准一致）
    _kwargs_list = [
        dict(yaw_deg=0.0),                 # 1 正脸
        dict(yaw_deg=+22.0),               # 2 左半侧
        dict(yaw_deg=-22.0),               # 3 右半侧
        dict(pitch_deg=+12.0),             # 4 轻微抬头
        dict(pitch_deg=-12.0),             # 5 轻微低头
        dict(yaw_deg=+52.0),               # 6 左大侧
        dict(yaw_deg=-52.0),               # 7 右大侧
    ]
    import enrollment_guide as _guide

    poses = [_guide.demo_landmarks(**kw) for kw in _kwargs_list]

    session = le.LiveEnrollmentSession(_FakeDetector(poses), per_slot=1, min_score=0)
    frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    snap = None
    for _ in range(len(poses)):
        snap = session.process_frame(frame)

    assert snap is not None
    slots = [it["analysis"]["slot"] for it in session.items()]
    check(session.coverage()["complete"] == True and len(slots) == 7,
          "7 机位黄金标准在 7 帧内被全部收齐",
          f"收录机位={slots}，覆盖={session.coverage()['filled_core']}")
    check("ALL 7 CORE" in " ".join(session.hud_lines()),
          "核心机位齐了之后 HUD 明确提示可以收工",
          f"HUD={session.hud_lines()}")
    check(snap["canvas"] is not None and snap["canvas"].shape == (480, 640, 3),
          "每帧都产出了可显示的预览画面（含人脸框与 HUD）",
          f"画面形状={None if snap['canvas'] is None else snap['canvas'].shape}")

    before = len(session.items())
    snap2 = session.process_frame(frame)
    check(len(session.items()) == before and not snap2["accepted"],
          "同一机位已有更好的帧时，新帧不会被收进（只留最好的）",
          f"提示：{snap2['note']}")

    session.clear()
    check(not session.items() and session.attempts == 0, "清空重录能重置状态")


# ---------------------------------------------------------------------------
# 3.8 浏览器 WebRTC 实时录入（帧处理器）
# ---------------------------------------------------------------------------
def test_webrtc_enrollment() -> None:
    print("\n=========== 3.8 浏览器 WebRTC 实时录入（帧处理器）===========")
    try:
        import av
        import live_enrollment as le
        from PIL import Image
    except Exception as err:  # pragma: no cover
        record(SKIP, "跳过 WebRTC 实时录入自检（缺 streamlit-webrtc / av）", str(err))
        return

    if not getattr(le, "WEBRTC_AVAILABLE", False):
        record(SKIP, "跳过 WebRTC 实时录入自检", "WEBRTC_AVAILABLE=False")
        return

    # 用真实人脸图造一帧画面（只验证链路与状态机，姿态用假检测器控制）
    files = sorted(Path("registered_faces").glob("*/*.png"))
    if not files:
        record(SKIP, "跳过 WebRTC 实时录入自检", "没有人脸图（先注册一个人）")
        return
    face = Image.open(files[0]).convert("RGB")
    if face.size != (220, 220):
        face = face.resize((220, 220))
    canvas = np.full((480, 640, 3), 130, dtype=np.uint8)
    canvas[130:350, 210:430] = np.asarray(face)
    frame_bgr = np.ascontiguousarray(canvas[:, :, ::-1])

    class _FakeFace:
        def __init__(self, box, landmarks):
            self.box = np.asarray(box, dtype=np.float32)
            self.landmarks = np.asarray(landmarks, dtype=np.float32)
            self.prob = 0.99

    class _FakeDetector:
        def __init__(self, poses):
            self.poses = poses
            self.i = 0

        def detect(self, image):
            pts = self.poses[self.i % len(self.poses)]
            self.i += 1
            return [_FakeFace([220, 140, 420, 340], pts)]

    import enrollment_guide as _guide

    poses = [_guide.demo_landmarks(**kw) for kw in (
        dict(yaw_deg=0.0),        # 正脸
        dict(yaw_deg=+22.0),      # 左半侧
        dict(yaw_deg=-22.0),      # 右半侧
        dict(pitch_deg=+12.0),    # 轻微抬头
        dict(pitch_deg=-12.0),    # 轻微低头
        dict(yaw_deg=+52.0),      # 左大侧
        dict(yaw_deg=-52.0),      # 右大侧
    )]

    try:
        import time as _time

        proc = le.create_enrollment_processor(per_slot=1, min_score=0,
                                              detect_interval=1, device="cpu")

        def _wait_analyzed(target: int, timeout: float = 10.0) -> None:
            """等后台分析线程把这一帧处理完。

            注意要同时看 busy：attempts 在**开始**分析时就 +1，
            只看它会在"分析到一半"时就返回（自检踩过这个坑）。
            """
            deadline = _time.time() + timeout
            while _time.time() < deadline:
                if proc.session.attempts >= target and not proc.busy:
                    return
                _time.sleep(0.02)

        proc.detector = _FakeDetector(poses)
        proc.session.detector = proc.detector

        shapes = []
        for i in range(len(poses)):
            out = proc.recv(av.VideoFrame.from_ndarray(frame_bgr, format="bgr24"))
            arr = out.to_ndarray(format="bgr24")
            shapes.append(arr.shape)
            _wait_analyzed(i + 1)          # 等这一帧被后台线程吃掉再喂下一帧

        slots = [it["analysis"]["slot"] for it in proc.session.items()]
        check(len(proc.session.items()) == 7 and proc.session.coverage()["complete"],
              "WebRTC 帧处理器：7 帧内集齐 7 机位黄金标准",
              f"收录机位={slots}")
        check(all(s == (480, 640, 3) for s in shapes),
              "recv 每帧都返回同尺寸的 BGR 帧（可直接回传浏览器）",
              f"形状={shapes[0]}")
        check(proc.fps > 0, "帧处理器统计了实时帧率", f"FPS={proc.fps:.1f}")
        check(proc._worker.is_alive(), "检测跑在后台线程里（recv 不被模型阻塞）")

        # 关键性能特性：recv() 的耗时应该远小于一次检测的耗时
        t0 = _time.perf_counter()
        for _ in range(10):
            proc.recv(av.VideoFrame.from_ndarray(frame_bgr, format="bgr24"))
        per_frame_ms = (_time.perf_counter() - t0) * 100.0
        check(per_frame_ms < 60 and proc.analyze_ms > per_frame_ms,
              "recv() 单帧耗时远小于模型分析耗时（视频通路不被拖住）",
              f"recv≈{per_frame_ms:.1f}ms/帧，分析≈{proc.analyze_ms:.0f}ms/帧，"
              f"丢弃 {proc.dropped} 帧")

        before = len(proc.session.items())
        proc.recv(av.VideoFrame.from_ndarray(frame_bgr, format="bgr24"))
        _wait_analyzed(proc.session.attempts + 1, timeout=3.0)
        check(len(proc.session.items()) == before,
              "同一机位已有更好的帧时不会被替换（只留最好的）")
    except Exception:
        record(FAIL, "WebRTC 实时录入自检异常", traceback.format_exc(limit=3))


# ---------------------------------------------------------------------------
# 4. 模型前向
# ---------------------------------------------------------------------------
def test_models(backbone: str, device: str) -> bool:
    print(f"\n=========== 4. 模型前向（骨干：{backbone}）===========")
    try:
        from face_detector import FaceDetector
        from feature_extractor import FeatureExtractor
    except Exception as err:
        record(SKIP, "跳过模型自检（缺少深度学习依赖）", str(err))
        return False

    try:
        detector = FaceDetector(device=device, align=True)
        record(PASS, "MTCNN 检测器构建成功", repr(detector))
    except Exception as err:
        record(FAIL, "MTCNN 构建失败", f"{err}")
        return False

    try:
        extractor = FeatureExtractor(backbone=backbone, device=device)
        record(PASS, "特征提取器构建成功", repr(extractor))
    except Exception as err:
        record(FAIL, "特征提取器构建失败（多半是权重下载问题）", f"{err}")
        return False

    check(extractor.embedding_dim == 512, "特征维度为 512",
          f"实际 {extractor.embedding_dim}")

    # 用随机图验证前向与归一化，不依赖真实人脸
    rng = np.random.default_rng(0)
    fake = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    feat = extractor.extract(fake)
    check(feat.shape == (extractor.embedding_dim,), "单张图前向输出形状正确", f"{feat.shape}")
    if extractor.l2_normalize:
        check(abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4, "输出特征已 L2 归一化",
              f"L2 范数 {float(np.linalg.norm(feat)):.6f}")

    batch = extractor.extract_batch([fake] * 3, batch_size=2)
    check(batch.shape[0] == 3, "批量提取返回 3 条特征", f"{batch.shape}")

    # MTCNN 对纯随机噪声不应该报错（检测不到人脸是正常的）
    try:
        faces = detector.detect(fake)
        record(PASS, "MTCNN 在无脸图片上正常返回", f"检测到 {len(faces)} 张人脸（预期 0）")
    except Exception as err:
        record(FAIL, "MTCNN 检测过程异常", str(err))
    return True


# ---------------------------------------------------------------------------
# 5. 端到端：注册 -> 识别
# ---------------------------------------------------------------------------
def _augment(image, count: int = 5) -> List:
    """把一张人脸图做轻微变换，模拟"同一个人不同照片"，用于端到端自检。"""
    from PIL import Image, ImageEnhance, ImageOps

    ops = [
        ("原图", lambda im: im),
        ("亮度+20%", lambda im: ImageEnhance.Brightness(im).enhance(1.2)),
        ("亮度-20%", lambda im: ImageEnhance.Brightness(im).enhance(0.8)),
        ("水平翻转", lambda im: ImageOps.mirror(im)),
        ("旋转+7°", lambda im: im.rotate(7, resample=Image.BILINEAR, fillcolor=(0, 0, 0))),
        ("旋转-7°", lambda im: im.rotate(-7, resample=Image.BILINEAR, fillcolor=(0, 0, 0))),
    ]
    variants = []
    for name, op in ops[:max(2, count)]:
        try:
            variants.append((name, op(image.convert("RGB"))))
        except Exception:
            continue
    return variants


def test_end_to_end(backbone: str, device: str, image_path: Optional[Path]) -> None:
    print("\n=========== 5. 端到端：注册 → 识别 ===========")
    try:
        from face_detector import FaceDetector
        from face_database import FaceDatabase
        from feature_extractor import FeatureExtractor
        from similarity import identify
    except Exception as err:
        record(SKIP, "跳过端到端自检（缺少深度学习依赖）", str(err))
        return

    if image_path is None:
        record(SKIP, "未提供真实人脸图",
               "用 python self_test.py --image 你的照片.jpg 可以跑完整流程")
        return
    if not Path(image_path).exists():
        record(FAIL, "指定的图片不存在", str(image_path))
        return

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    work = make_workdir("e2e")
    try:
        detector = FaceDetector(device=device, align=True)
        extractor = FeatureExtractor(backbone=backbone, device=device)
        db = FaceDatabase(db_path=work / "e2e.pkl", face_dir=work / "faces")

        # ---- 注册：用原图的若干轻微变体当"同一个人的 5 张照片" ----
        crops, embs, used = [], [], []
        for name, variant in _augment(image, count=5):
            faces = detector.detect(variant)
            if not faces:
                continue
            crop = detector.crop(variant, faces[0])
            crops.append(crop)
            embs.append(extractor.extract(crop))
            used.append(name)

        if len(embs) < 2:
            record(SKIP, "这张图上 MTCNN 没检测到足够的人脸",
                   "换一张正面清晰的照片再试（端到端流程本身没跑起来，不算失败）")
            return
        record(PASS, f"从原图生成 {len(embs)} 张注册图", "变体：" + "、".join(used))

        info = db.add_person("自检用户", np.stack(embs), face_images=crops,
                             backbone=backbone)
        check(info.num_images == len(embs), "注册成功", f"库内 {info.num_images} 条特征")

        # ---- 识别：用原图查询，应当命中"自检用户" ----
        faces = detector.detect(image)
        if not faces:
            record(FAIL, "识别阶段未检测到人脸", "与注册阶段结果不一致，请检查")
            return
        query = extractor.extract(detector.crop(image, faces[0]))
        names, gallery = db.get_gallery()
        result = identify(query, names, gallery, threshold=0.6, top_k=3)

        check(result.is_known and result.name == "自检用户",
              "端到端识别命中本人",
              f"识别为 {result.name}，相似度 {result.similarity:.4f}")
        check(result.similarity >= 0.6, "相似度不低于阈值 0.6",
              f"实测 {result.similarity:.4f}")

        # ---- 陌生人测试：用一张纯噪声"脸"查询（应当判为陌生人）----
        rng = np.random.default_rng(3)
        stranger_feat = extractor.extract(rng.integers(0, 256, (160, 160, 3), dtype=np.uint8))
        stranger = identify(stranger_feat, names, gallery, threshold=0.6)
        check(not stranger.is_known, "随机噪声被判定为陌生人",
              f"最高相似度 {stranger.similarity:.4f}")
    except Exception:
        record(FAIL, "端到端自检异常", traceback.format_exc(limit=4))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="人脸识别 Demo 自检脚本")
    parser.add_argument("--image", type=str, default=None,
                        help="一张真实人脸图片的路径，用于第 5 组端到端测试")
    parser.add_argument("--backbone", type=str, default="inception_resnet_v1",
                        choices=["inception_resnet_v1", "resnet18"], help="特征提取骨干网络")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    parser.add_argument("--skip-model", action="store_true",
                        help="只跑不需要深度学习框架的检查（1~3 组）")
    args = parser.parse_args()

    print("=" * 60)
    print(" 基于预训练模型的人脸识别 Demo —— 项目自检")
    print("=" * 60)

    info = test_environment()
    test_similarity()
    test_database()
    test_enrollment_guide()
    test_live_enrollment()
    test_webrtc_enrollment()

    if args.skip_model or not info["torch"]:
        if not args.skip_model:
            record(SKIP, "跳过第 4、5 组", "PyTorch 未安装，无法进行模型相关检查")
    else:
        models_ok = test_models(args.backbone, args.device)
        if models_ok:
            test_end_to_end(args.backbone, args.device,
                            Path(args.image) if args.image else None)

    # ---- 汇总 ----
    passed = sum(1 for level, _, _ in RESULTS if level == PASS)
    failed = [r for r in RESULTS if r[0] == FAIL]
    skipped = sum(1 for level, _, _ in RESULTS if level == SKIP)

    print("\n" + "=" * 60)
    print(f" 汇总：通过 {passed} 项，失败 {len(failed)} 项，跳过 {skipped} 项")
    if failed:
        print(" 失败明细：")
        for _, name, detail in failed:
            print(f"   - {name}：{detail}")
        print("\n 结论：存在失败项，请按上面的提示修复后再演示。")
        return 1
    print("\n 结论：全部检查通过，可以启动界面：streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
