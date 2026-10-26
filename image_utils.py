# -*- coding: utf-8 -*-
"""
image_utils.py
==============
图像处理的小工具集合（纯 numpy / PIL，无深度学习依赖）。

本模块集中放"把各种来源统一转成 RGB 的 PIL.Image"以及"把 numpy 数组统一转成
RGB/uint8 数组"的逻辑，避免它散落在 face_detector / enrollment_guide /
feature_extractor / app 多处重复实现：
不同模块各自的 to_pil / _to_pil 行为不一致时，最容易出现"同一个函数换了个文件就
行为不同"的坑。这里只维护一份。
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
from PIL import Image

# 可接受的输入类型：PIL 图、numpy 数组、文件路径（含 Streamlit 上传对象，它支持 .read()）
ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def to_rgb_array(arr: np.ndarray) -> np.ndarray:
    """把任意形状/类型的数组统一转成 **RGB、uint8** 的 (H, W, 3) 数组。

    统一处理三种常见情况：
      * 灰度(2D)  -> 复制成三通道；
      * RGBA(4通道)-> 丢掉 alpha 通道；
      * 非 uint8   -> 先截断到 0~255 再转成 uint8。
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:                       # 灰度图 -> RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:               # 浮点图（0~1 或 0~255）转成 uint8
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_rgb_pil(image: ImageLike) -> Image.Image:
    """把 路径 / PIL / numpy / 上传对象 统一转换成 RGB 的 PIL.Image。

    统一处理三种常见情况：
      * PIL.Image          -> 保证是 RGB 模式；
      * numpy 数组         -> 交给 to_rgb_array 规范化后再包成 PIL；
      * 路径 / 上传对象     -> 用 PIL.Image.open 打开（Streamlit 上传对象也实现了 read()）。
    返回的图一定是 "RGB" 模式，dtype 为 uint8，可直接交给 MTCNN / 特征提取器。
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image

    if isinstance(image, np.ndarray):
        return Image.fromarray(np.ascontiguousarray(to_rgb_array(image))).convert("RGB")

    # 路径字符串 / Path / Streamlit UploadedFile（都支持 open 或 read）
    return Image.open(image).convert("RGB")
