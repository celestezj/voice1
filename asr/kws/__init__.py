# -*- coding: utf-8 -*-
"""打断词检测（keyword spotting，KWS）：旁路主动打断排队任务的专用轻量模块。

与 ASR 主后端正交：独立极轻量模型专司"关键词命中"，不走主识别队列。
引擎 `interrupt_words=None`（默认）→ 不加载，打断旁路完全关闭，零开销。
"""
from .interrupt import InterruptDetector, get_interrupt_detector

__all__ = ["InterruptDetector", "get_interrupt_detector"]
