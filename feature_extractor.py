# -*- coding: utf-8 -*-
"""
feature_extractor.py
====================
人脸特征（Embedding）提取模块。

人脸识别的核心思想：把人脸图映射成一个固定长度的特征向量（本文 512 维），
同一个人的向量在特征空间中距离很近，不同人的距离较远。
本模块用预训练模型直接输出这个向量，无需自己训练，属于"基于预训练模型的迁移学习"。

关于骨干网络（重要，答辩时会问）
--------------------------------
作业要求写的是"facenet-pytorch 的 InceptionResnetV1，骨干网络使用 ResNet18"，
这两者在实现上其实是两个不同的东西：

1. ``inception_resnet_v1``（默认，推荐）
   facenet-pytorch 提供的 InceptionResnetV1 网络，本身就是一个人脸专用骨干，
   在 VGGFace2 / CASIA-WebFace 上训练过，直接输出 512 维人脸特征。
   识别效果好，是本 Demo 的默认选择。

2. ``resnet18``
   严格意义上的 ResNet18 骨干：来自 torchvision，加载 ImageNet 预训练权重，
   砍掉最后的 1000 类全连接层，用倒数第二层的 512 维输出当特征。
   这更贴近"骨干网络使用 ResNet18"的字面要求，但它是"通用物体"预训练，
   没有见过人脸身份标签，所以人脸区分能力明显弱于方案 1。

因此本项目把两种骨干都实现了，通过参数（或界面侧边栏）切换，
默认用识别效果更好的 InceptionResnetV1；想演示"ResNet18 骨干"时切过去即可。
两种骨干输出维度都是 512，但它们的特征空间不通用，
切换骨干后必须 **重新注册人脸库**（face_database 会检测维度/骨干是否匹配并报警）。

预处理差异（也必须一致，否则特征会失效）：
    - InceptionResnetV1：FaceNet 的 fixed_image_standardization，(x - 127.5) / 128.0
    - ResNet18        ：ImageNet 归一化，先 /255 再减均值除标准差
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from face_detector import DEFAULT_IMAGE_SIZE, resolve_device
from image_utils import to_rgb_array
from similarity import l2_normalize

# FaceNet 固定标准化参数：把 0~255 的像素映射到约 [-1, 1]
FACENET_MEAN = 127.5
FACENET_STD = 128.0

# ImageNet 归一化参数（torchvision 官方值）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# 预训练权重的缓存目录
#
# facenet-pytorch 与 torchvision 默认把权重下载到 ~/.cache/torch/checkpoints，
# 该路径由环境变量 TORCH_HOME 控制（facenet_pytorch.models.inception_resnet_v1.get_torch_home）。
#
# 问题：部分 Windows 机器上用户主目录权限异常，创建 C:\Users\<用户名>\.cache 会被拒绝，
# 表现为 "预训练模型加载失败：[WinError 5] 拒绝访问。: 'C:\\Users\\xxx/.cache'"。
# 与其让同学去改系统权限，不如程序自适应：探测默认目录是否可写，不可写就自动
# 切换到项目目录内的 .cache/torch（本目录已在 .gitignore 中忽略）。
# ---------------------------------------------------------------------------
_TORCH_HOME_READY = False


def ensure_writable_torch_home(
    project_dir: Optional[Path] = None,
    verbose: bool = True,
) -> str:
    """确保 TORCH_HOME 指向一个**确实可写**的目录，并返回该路径。

    优先级：
        1. 用户已经设置好的 TORCH_HOME（尊重用户选择）；
        2. 默认的 ~/.cache/torch；
        3. 项目目录下的 .cache/torch（兜底）。

    每个候选目录都会用"真的写一个探针文件"的方式验证，避免只看 mkdir 是否成功。
    """
    global _TORCH_HOME_READY
    if _TORCH_HOME_READY and os.environ.get("TORCH_HOME"):
        return os.environ["TORCH_HOME"]

    def _usable(path: Path) -> bool:
        """创建目录并实际写一个文件，确认可写。"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except Exception:
            return False

    user_set = os.environ.get("TORCH_HOME")
    default_home = Path(os.path.expanduser("~/.cache/torch"))
    fallback_home = Path(project_dir or Path(__file__).resolve().parent) / ".cache" / "torch"

    candidates = [Path(user_set)] if user_set else [default_home, fallback_home]

    for candidate in candidates:
        if _usable(candidate):
            os.environ["TORCH_HOME"] = str(candidate)
            _TORCH_HOME_READY = True
            if verbose and user_set is None and candidate != default_home:
                print(
                    f"[feature_extractor] 默认缓存目录 {default_home} 不可写，"
                    f"已自动把预训练权重缓存切换到：{candidate}"
                )
            return str(candidate)

    _TORCH_HOME_READY = True
    if verbose:
        print(
            "[feature_extractor] 警告：找不到可写的权重缓存目录，预训练权重可能下载失败。\n"
            "                   可手动指定，例如：set TORCH_HOME=D:\\torch_cache"
        )
    return os.environ.get("TORCH_HOME", "")


class FeatureExtractor:
    """把 160x160 的人脸图编码成 512 维特征向量。"""

    SUPPORTED_BACKBONES = ("inception_resnet_v1", "resnet18")

    def __init__(
        self,
        backbone: str = "inception_resnet_v1",
        pretrained: Union[str, bool, None] = "vggface2",
        device: Optional[str] = None,
        l2_normalize: bool = True,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        """
        参数说明：
            backbone     : 'inception_resnet_v1' 或 'resnet18'
            pretrained   : InceptionResnetV1 取 'vggface2' / 'casia-webface' / None；
                           ResNet18 取 True/'imagenet' 表示加载 ImageNet 预训练权重，False 表示随机初始化
            device       : 'auto' / 'cpu' / 'cuda'
            l2_normalize : 是否把输出向量归一化成长度 1。
                           归一化后"余弦相似度"就等于"向量点积"，计算更快且阈值更稳定。
            image_size   : 模型输入边长（FaceNet 系列固定 160）
        """
        backbone = str(backbone).lower().strip()
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"不支持的骨干网络：{backbone}，可选 {self.SUPPORTED_BACKBONES}"
            )

        self.backbone = backbone
        self.pretrained = pretrained
        self.image_size = int(image_size)
        self.l2_normalize = bool(l2_normalize)
        self.device = resolve_device(device)

        # 要加载预训练权重前，先确保权重缓存目录可写（否则某些机器会因
        # ~/.cache 不可写而报 WinError 5，权重永远下不下来）
        if self._is_pretrained(pretrained):
            self.torch_home = ensure_writable_torch_home()
        else:
            self.torch_home = os.environ.get("TORCH_HOME", "")

        self.model = self._build_model(backbone, pretrained)
        self.model.eval()                      # 推理模式：关闭 Dropout / BN 更新
        self.model = self.model.to(self.device)

        # 探测输出维度：喂一个全零张量，避免硬编码出错
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            self.embedding_dim = int(self.model(dummy).shape[-1])

    # ------------------------------------------------------------------
    # 建模
    # ------------------------------------------------------------------
    @staticmethod
    def _is_pretrained(pretrained: Union[str, bool, None]) -> bool:
        """把各种写法统一成"要不要加载预训练权重"。"""
        if pretrained is None or pretrained is False:
            return False
        if isinstance(pretrained, str) and pretrained.lower() in ("none", "false", "random", ""):
            return False
        return True

    def _build_model(self, backbone: str, pretrained: Union[str, bool, None]) -> nn.Module:
        """构建骨干网络，并把可能出现的"下载权重失败"翻译成中文提示。"""
        load_pretrained = self._is_pretrained(pretrained)
        try:
            if backbone == "inception_resnet_v1":
                from facenet_pytorch import InceptionResnetV1

                # classify=False 表示不接分类头，只输出 512 维特征向量
                weight_name = pretrained if isinstance(pretrained, str) and load_pretrained else None
                model = InceptionResnetV1(pretrained=weight_name, classify=False)
            else:
                from torchvision import models

                weights = models.ResNet18_Weights.IMAGENET1K_V1 if load_pretrained else None
                model = models.resnet18(weights=weights)
                # 关键一步：把最后的全连接层换成 Identity，
                # 于是 forward 输出的是全局平均池化后的 512 维特征
                model.fc = nn.Identity()
            return model
        except Exception as exc:  # 网络不通 / 磁盘没空间 / 版本不兼容都会走到这里
            cache_dir = os.environ.get("TORCH_HOME", "~/.cache/torch")
            raise RuntimeError(
                "预训练模型加载失败：{}\n"
                "当前使用的权重缓存目录：{}\n"
                "常见原因与解决办法：\n"
                "  1) 首次运行需要联网下载权重（facenet-pytorch InceptionResnetV1 约 110MB，\n"
                "     torchvision ResNet18 约 45MB），请检查网络后重试；\n"
                "  2) 也可以手动下载权重，放进上面的缓存目录（子目录 checkpoints）：\n"
                "     InceptionResnetV1(vggface2) 文件名 20180402-114759-vggface2.pt，\n"
                "     下载地址 https://github.com/timesler/facenet-pytorch/releases/tag/v2.2.9\n"
                "     ResNet18 文件名 resnet18-f37072fd.pth\n"
                "  3) 想把缓存放到别处，可设置环境变量 TORCH_HOME，例如：\n"
                "     set TORCH_HOME=D:\\torch_cache\n"
                "  4) 只想验证代码流程、不追求识别精度时，可用 pretrained=None 走随机初始化权重。"
                .format(exc, cache_dir)
            ) from exc

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------
    def preprocess(self, faces: Union[np.ndarray, Sequence[np.ndarray]]) -> torch.Tensor:
        """把 uint8 的 RGB 人脸图（单张或多张）转成模型输入张量 NCHW。

        输入：单张 (H,W,3) 或 列表 [(H,W,3), ...]
        输出：torch.FloatTensor，形状 (N, 3, size, size)，已放到目标设备
        """
        if isinstance(faces, np.ndarray) and faces.ndim == 3:
            faces = [faces]
        if len(faces) == 0:
            return torch.zeros(0, 3, self.image_size, self.image_size, device=self.device)

        # 统一走 image_utils.to_rgb_array：灰度补通道、RGBA 截 alpha、非 uint8 截断
        arrs = [to_rgb_array(face) for face in faces]

        x = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float()  # NCHW

        # 尺寸不一致时用双线性插值对齐（正常情况下 FaceDetector 已经裁成 160）
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        if self.backbone == "inception_resnet_v1":
            # FaceNet 官方固定标准化：(像素 - 127.5) / 128
            x = (x - FACENET_MEAN) / FACENET_STD
        else:
            x = x / 255.0
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            x = (x - mean) / std

        return x.to(self.device)

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_batch(
        self,
        faces: Sequence[np.ndarray],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """批量提取特征，返回 (N, D) 的 float32 数组。

        progress_callback(done, total) 可用于界面进度条。
        """
        if len(faces) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        outputs: List[np.ndarray] = []
        total = len(faces)
        for start in range(0, total, int(batch_size)):
            chunk = faces[start:start + int(batch_size)]
            x = self.preprocess(chunk)
            feat = self.model(x)                       # (B, D)
            feat = feat.detach().cpu().numpy().astype(np.float32)
            if self.l2_normalize:
                feat = l2_normalize(feat, axis=-1)
            outputs.append(feat)
            if progress_callback is not None:
                progress_callback(min(start + len(chunk), total), total)

        return np.concatenate(outputs, axis=0)

    def extract(self, face: np.ndarray) -> np.ndarray:
        """提取单张人脸的特征，返回 (D,) 的 float32 向量。"""
        return self.extract_batch([face])[0]

    # ------------------------------------------------------------------
    # 面向界面的描述信息
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        """返回模型信息，方便在 Streamlit 侧边栏 / 结果区展示。"""
        return {
            "骨干网络": self.backbone,
            "预训练权重": str(self.pretrained),
            "特征维度": self.embedding_dim,
            "输入尺寸": f"{self.image_size}x{self.image_size}",
            "是否 L2 归一化": self.l2_normalize,
            "运行设备": str(self.device),
            "权重缓存目录": self.torch_home or "(默认)",
        }

    def __repr__(self) -> str:
        return (f"FeatureExtractor(backbone='{self.backbone}', pretrained='{self.pretrained}', "
                f"dim={self.embedding_dim}, device='{self.device}')")


# ---------------------------------------------------------------------------
# 模块自测：python feature_extractor.py
# 只验证网络结构与预处理是否正确（用随机图，不需要真实人脸）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name in FeatureExtractor.SUPPORTED_BACKBONES:
        pretrained = "vggface2" if name == "inception_resnet_v1" else True
        print(f"\n=== 测试骨干：{name} ===")
        try:
            extractor = FeatureExtractor(backbone=name, pretrained=pretrained)
        except RuntimeError as err:
            print("[跳过] 权重不可用：", str(err).splitlines()[0])
            continue

        print(extractor)
        print("模型信息：", extractor.describe())

        rng = np.random.default_rng(0)
        fake_face = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
        feat = extractor.extract(fake_face)
        print("单张特征形状：", feat.shape, "L2 范数：", float(np.linalg.norm(feat)))
        batch = extractor.extract_batch([fake_face] * 5, batch_size=2)
        print("批量特征形状：", batch.shape)
