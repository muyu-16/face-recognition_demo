# -*- coding: utf-8 -*-
"""
prefetch_models.py
==================
提前把两个骨干网络的预训练权重下载到本地缓存，避免"第一次打开界面时卡在下载"。

为什么单独做成脚本
------------------
1. 权重加起来约 150 MB（vggface2 约 107 MB + resnet18 约 45 MB），
   第一次运行必须联网；放在命令行里下载可以看到进度，比在网页里干等清楚得多。
2. 下载目录的"可写性兜底"逻辑在 feature_extractor.py 里
   （有些机器上 ~/.cache/torch 不可写，程序会自动改用项目内的 .cache/torch）。
   这里直接复用那套逻辑，保证脚本下载的权重和应用运行时用的是同一个目录。

用法：
    python prefetch_models.py            # 下载两个骨干（已缓存则跳过）
    python prefetch_models.py --check     # 只检查缓存状态，不下载
    python prefetch_models.py --only resnet18
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def torch_home() -> Path:
    """返回当前使用的 torch 缓存目录（和 feature_extractor 里保持一致）。"""
    try:
        import feature_extractor as fe

        fe.ensure_writable_torch_home()
    except Exception:
        pass
    return Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))


def cached_files() -> dict:
    """列出缓存目录里已有的权重文件。"""
    home = torch_home()
    ckpt = home / "checkpoints"
    files = {}
    if ckpt.is_dir():
        for path in ckpt.glob("*"):
            if path.is_file() and path.suffix in (".pt", ".pth"):
                files[path.name] = path.stat().st_size
    hub = home / "hub" / "checkpoints"
    if hub.is_dir():
        for path in hub.glob("*"):
            if path.is_file() and path.suffix in (".pt", ".pth"):
                files[path.name] = path.stat().st_size
    return files


def show_cache() -> None:
    home = torch_home()
    print(f"权重缓存目录：{home}")
    files = cached_files()
    if not files:
        print("   （还没有任何权重，首次运行需要联网下载约 150 MB）")
        return
    total = 0
    for name, size in sorted(files.items()):
        total += size
        print(f"   ✓ {name}  {human(size)}")
    print(f"   合计 {human(total)}")


def prefetch(which: str) -> int:
    """下载指定骨干的权重；返回 0 表示成功（或已缓存）。"""
    failures = 0

    if which in ("all", "inception_resnet_v1"):
        print("\n[1] InceptionResnetV1（VGGFace2，人脸识别预训练，约 107 MB）")
        try:
            t0 = time.time()
            from facenet_pytorch import InceptionResnetV1

            model = InceptionResnetV1(pretrained="vggface2")
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"    ✓ 就绪（{params:.1f}M 参数，耗时 {time.time() - t0:.0f}s）")
            del model
        except Exception as err:
            failures += 1
            print(f"    ✗ 失败：{type(err).__name__}: {err}")
            print("      手动下载后放到上面的 checkpoints 目录即可：")
            print("        https://github.com/timesler/facenet-pytorch/releases/download/"
                  "v2.2.9/20180402-114759-vggface2.pt")

    if which in ("all", "resnet18"):
        print("\n[2] ResNet18（ImageNet 预训练，作业字面要求的骨干，约 45 MB）")
        try:
            t0 = time.time()
            import torchvision.models as models

            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print(f"    ✓ 就绪（耗时 {time.time() - t0:.0f}s）")
            del model
        except Exception as err:
            failures += 1
            print(f"    ✗ 失败：{type(err).__name__}: {err}")
            print("      这个骨干是可选的：侧边栏默认用 InceptionResnetV1，"
                  "不切换就不需要它。")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="预下载人脸识别用的预训练权重")
    parser.add_argument("--check", action="store_true", help="只检查缓存，不下载")
    parser.add_argument("--only", choices=["inception_resnet_v1", "resnet18", "all"],
                        default="all", help="只下载指定骨干")
    args = parser.parse_args()

    print("=" * 62)
    print("  预训练权重准备")
    print("=" * 62)
    show_cache()

    if args.check:
        print("\n（--check 模式：未做任何下载）")
        return 0

    print("\n开始检查/下载权重（已缓存的会秒过）……")
    code = prefetch(args.only)
    print("\n" + "-" * 62)
    show_cache()
    if code:
        print("\n⚠️  有下载失败项：不影响启动，程序会在真正用到时再试一次；"
              "如果一直失败，按上面提示手动放置权重文件。")
    else:
        print("\n✅ 权重已就绪，启动后不会再有长时间等待。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
