# -*- coding: utf-8 -*-
"""
webrtc_common.py
================
实时视频（WebRTC）相关的公共基础设施。

realtime.py 与 live_enrollment.py 都用到同一套"可选依赖探测"逻辑：
    * cv2（清晰度/绘制文字用，缺失则降级）；
    * av + streamlit_webrtc（实时视频流，缺失则整个实时功能不可用）。
两者原本各自复制了一遍 import 与 RTC 配置，既冗长又容易两边改了一边漏了另一边。
这里集中维护一份，两个模块统一从这里 import。
"""

from __future__ import annotations

# ---- OpenCV：可选，缺失时只影响 FPS / HUD 文字的叠加，不影响识别逻辑 ----
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# ---- WebRTC 实时视频流：可选，缺失时整个实时功能降级为提示 ----
try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        webrtc_streamer,
    )

    try:  # 0.4x 起 mode 要求传 WebRtcMode 枚举，老版本才接受字符串
        from streamlit_webrtc import WebRtcMode

        _MODE_SENDRECV = WebRtcMode.SENDRECV
    except Exception:  # pragma: no cover
        _MODE_SENDRECV = "sendrecv"

    WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    webrtc_streamer = None
    _MODE_SENDRECV = "sendrecv"
    WEBRTC_AVAILABLE = False

# 只需要一个 STUN 服务器即可在局域网/本机建立连接
_RTC_CONFIG = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if WEBRTC_AVAILABLE
    else None
)
