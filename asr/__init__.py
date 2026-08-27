# -*- coding: utf-8 -*-
"""voice1 实时语音识别包。"""
from .core.engine import RealtimeASR
from .core.jobs import SentenceResult

__all__ = ["RealtimeASR", "SentenceResult"]
