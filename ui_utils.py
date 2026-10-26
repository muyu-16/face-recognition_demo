# -*- coding: utf-8 -*-
"""
ui_utils.py
===========
Streamlit 界面层的公共小工具。

目前只放一件事：统一"自适应宽度地显示一张图"的写法。
app.py 与 live_enrollment.py 都需要显示图片，原先两边各写了一份
try/except 版本回退链，改一处忘一处很容易漏；
这里集中维护一份，调用方只管传图和标题。
"""

from __future__ import annotations

from typing import Optional


def show_image(image, caption: Optional[str] = None) -> None:
    """自适应宽度地显示一张图（Streamlit 1.50+ 的 `width="stretch"` 写法）。

    项目 requirements.txt 要求 streamlit>=1.50，全站统一使用新版参数，
    因此这里不再保留旧版 `use_container_width` 的回退分支。
    """
    import streamlit as st

    st.image(image, caption=caption, width="stretch")
