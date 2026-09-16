# -*- coding: utf-8 -*-
"""首批示例工具：get_weather（复用 assistant/qweather 技能，直接 HTTP 调，不走 MCP 更轻）。

复用的东西（全在 assistant/qweather/scripts/，不重复造轮子）：
  - JWT / API-Key 鉴权、经纬度定位（配置优先 + IP 反查兜底）、城市反查
  - weather_to_ai_summary.gen_summary 生成 AI 易读摘要
本模块只负责"把 qweather 包成 Tool 接口"：city 参数（可选）经 geo 接口查经纬度，
默认查配置位置。凭据缺失/城市找不到/接口错误都返回可读错误文本，绝不抛异常。
"""
from __future__ import annotations

import os
import sys

from .base import tool

# 技能模块位置：tool/ 在仓库根，assistant/qweather/scripts 是它的兄弟目录
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QW_SCRIPTS = os.path.join(_REPO_ROOT, "assistant", "qweather", "scripts")
_QW_CONFIG = os.path.join(_REPO_ROOT, "assistant", "qweather", "config.json")


def _import_qweather():
    """导入 assistant/qweather 模块（懒加载：本工具被调用才触发）。"""
    sys.path.insert(0, _QW_SCRIPTS)
    import get_location as gl
    import weather as qw
    import weather_to_ai_summary as ws
    return qw, gl, ws


def _city_latlon(qw, auth, city):
    """城市名 → (lat, lon, name)。geo 接口失败/找不到返回 None。"""
    try:
        data = qw._qweather_get(qw.GEO_HOST, "/v2/city/lookup", {"location": city}, auth)
        locs = data.get("location") or []
        if not locs:
            return None
        loc = locs[0]
        return float(loc["lat"]), float(loc["lon"]), loc.get("name")
    except Exception:
        return None


def _query(city, days):
    """核心查询：返回摘要文本，或可读错误文本（不抛异常）。"""
    if not os.path.isfile(_QW_CONFIG):
        return "无法查询天气：缺少和风天气配置文件 %s（请配置 assistant/qweather/config.json）" % _QW_CONFIG
    try:
        qw, gl, ws = _import_qweather()
    except Exception as e:
        return "无法查询天气：qweather 技能模块加载失败：%s" % e

    config = gl.load_config(_QW_CONFIG)
    priv = config.get("qweather_private_key")
    if priv and not os.path.isabs(priv):
        config["qweather_private_key"] = os.path.join(os.path.dirname(_QW_CONFIG), priv)
    auth = qw.build_auth(config)
    if auth is None:
        return ("无法查询天气：未配置和风天气凭据（assistant/qweather/config.json 中填 "
                "qweather_key 或用 JWT：qweather_kid / qweather_sub / qweather_private_key）")

    class _A:
        pass

    args = _A()
    args.key = None
    args.config = _QW_CONFIG
    args.lat = args.lon = None
    args.days = days
    args.hours = 0
    label = None
    if city:
        loc = _city_latlon(qw, auth, city)
        if not loc:
            return "无法查询天气：找不到城市「%s」，请换更完整的地名（如「北京市」）再试" % city
        args.lat, args.lon, label = loc[0], loc[1], loc[2]

    try:
        result = qw.collect(config, args)
        text = ws.gen_summary(result)
    except Exception as e:
        return "查询天气失败：%s" % e
    if label:
        text = "城市：%s\n%s" % (label, text)
    return text


@tool(
    "get_weather",
    "查询天气（当前/未来几天，含温度、降水、风；默认当前配置位置，可指定城市）",
    {
        "city": "[可选] 城市名，如 北京 或 北京市；不填则查询当前配置位置",
        "days": "[可选] 预报天数：3 或 7，默认 7（整周覆盖，用户可能追问后面几天）",
    },
    explanation=("复用和风天气 qweather：城市→经纬度→当前+预报→AI 易读摘要。"
                 "需要天气/温度/降水信息时调用。默认取整周（7 天）预报，"
                 "用户追问后面几天/别的城市时不用重查就能答；确实超出范围再重新调用。"),
    timeout=15.0,
    max_result=1200,
)
def get_weather(params: dict) -> str:
    city = (params.get("city") or "").strip()
    try:
        days = int(params.get("days") or 7)
        days = days if days in (3, 7) else 7
    except (TypeError, ValueError):
        days = 7
    return _query(city, days)
