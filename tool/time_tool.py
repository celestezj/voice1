# -*- coding: utf-8 -*-
"""首批示例工具：get_time（当前日期时间，零网络本地执行）。"""
from __future__ import annotations

from datetime import datetime

from .base import tool

_WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@tool(
    "get_time",
    "获取当前日期和时间（含星期）",
    explanation="返回本地日期、时间与星期，如 2026-09-16 周三 14:30:05。需要时间信息时调用，零网络毫秒级返回。",
    timeout=3.0,
    max_result=200,
)
def get_time(params: dict) -> str:
    now = datetime.now()
    return "%s %s %s" % (
        now.strftime("%Y-%m-%d"),
        _WEEK[now.weekday()],
        now.strftime("%H:%M:%S"),
    )
