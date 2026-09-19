# -*- coding: utf-8 -*-
"""llm+tools 金价查询：get_gold_history（复用 assistant/gold 数据管线，不走 MCP 更轻）。

参照 soviet-joke skill 的模式——确定性逻辑（拉数/统计/免责声明）全在数据脚本里，
Tool 只做薄封装：参数校验 → 调数据层 → 返回可读文本，模型只呈现、不造数。

复用的东西（全在 assistant/gold/gold_mcp.py，不重复造轮子）：
  - fetch_au0 / fetch_xau_realtime：新浪免费源，零 API key，30h 磁盘缓存
  - compute_stats / downsample：统计 + 技术指标 + 等间距趋势样本
  - DISCLAIMER：金融合规免责声明（勿删，任何含投资成分的输出都必须带上）
本模块只负责"把 gold 数据管线包成 Tool 接口"：period/market 参数校验、
错误都返回可读文本，绝不抛异常。
"""
from __future__ import annotations

import json
import os
import sys

from .base import tool

# 技能模块位置：tool/ 在仓库根，assistant/gold 是它的兄弟目录
# （代理策略统一在 tool/__init__.py `apply_network_policy()`，见 `load_tools`）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLD_DIR = os.path.join(_REPO_ROOT, "assistant", "gold")


def _import_gold():
    """导入 assistant/gold/gold_mcp（懒加载：本工具被调用才触发）。"""
    sys.path.insert(0, _GOLD_DIR)
    import gold_mcp
    return gold_mcp


def _query(period, market):
    """核心查询：返回可读文本，或可读错误文本（不抛异常）。"""
    if not os.path.isdir(_GOLD_DIR):
        return "无法查询金价：缺少数据模块 assistant/gold"
    try:
        gm = _import_gold()
    except Exception as e:
        return "无法查询金价：assistant/gold 模块加载失败：%s" % e

    market = (market or "au9999").lower()
    period = (period or "1y").lower()

    if market == "xauusd":
        try:
            q = gm.fetch_xau_realtime()
        except Exception as e:
            return "查询国际金价失败：%s" % e
        q["disclaimer"] = gm.DISCLAIMER
        return json.dumps(q, ensure_ascii=False, indent=2)

    if market != "au9999":
        return "无法查询金价：未知市场 %r（可选 au9999 国内 / xauusd 国际实时）" % market
    if period not in gm.PERIODS:
        return "无法查询金价：未知周期 %r（可选 %s）" % (period, "/".join(sorted(gm.PERIODS)))

    try:
        window, cached = gm._window_rows(period)
        if not window:
            return "无法查询金价：周期 %s 无数据" % period
        stats = gm.compute_stats(window)
        sample = gm.downsample(window)
    except Exception as e:
        return "查询金价失败：%s" % e

    lines = []
    lines.append("国内沪金主力 AU0（%s）· 周期 %s · 数据至 %s%s" % (
        gm.UNIT_AU, period, window[-1]["date"], "（缓存命中）" if cached else ""))
    lines.append("最新收盘 %s，较期初（%s %s）%+.2f%%，年化 %+.2f%%，年化波动 %.2f%%，最大回撤 %.2f%%" % (
        stats["last_close"], stats["start"]["date"], stats["start"]["close"],
        stats["period_change_pct"], stats["annualized_return_pct"],
        stats["annualized_volatility_pct"], stats["max_drawdown_pct"]))
    lines.append("区间最高 %s · 区间最低 %s" % (stats["period_high"], stats["period_low"]))
    ma = []
    for k in ("ma20", "ma60", "ma200"):
        if stats.get(k) is not None:
            ma.append("%s=%s" % (k.upper(), stats[k]))
    rsi = stats.get("rsi14")
    macd = stats.get("macd") or {}
    extra = "；".join(filter(None, [
        "MA " + " ".join(ma) if ma else "",
        ("RSI14=%s" % rsi) if rsi is not None else "",
        ("MACD DIF=%s DEA=%s 柱=%s" % (macd.get("dif"), macd.get("dea"), macd.get("histogram")))
        if macd.get("dif") is not None else "",
    ]))
    if extra:
        lines.append(extra)
    lines.append("趋势样本（等间距 %d 点）：" % len(sample))
    for p in sample:
        lines.append("  %s %s" % (p["date"], p["close"]))
    lines.append("注：国际现货金（XAU/USD）仅实时无历史，见 market=xauusd。")
    lines.append(gm.DISCLAIMER)
    return "\n".join(lines)


@tool(
    "get_gold_history",
    "查询金价历史与统计（国内沪金 AU0 全历史 + 国际现货金实时；含技术指标与免责声明）",
    {
        "period": "[可选] 周期：1m(1月)/6m(半年)/1y(1年)/2y(2年)/5y(5年)/all(极全)，默认 1y",
        "market": "[可选] au9999(国内沪金，默认) / xauusd(国际现货，仅实时)",
    },
    explanation=("复用 assistant/gold 数据管线（新浪免费源，零 API key，30h 缓存）。"
                 "需要黄金价格/历史走势/涨跌统计/技术指标时调用。服务端已预算好统计，"
                 "直接引述即可，不要编造数据里没有的数字。"),
    timeout=25.0,   # 网络抓取可能 15~20s（缓存命中则秒回）
    max_result=2000,
)
def get_gold_history(params: dict) -> str:
    period = (params.get("period") or "1y").strip()
    market = (params.get("market") or "au9999").strip()
    return _query(period, market)
