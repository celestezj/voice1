# -*- coding: utf-8 -*-
"""LLM 模式工具包（`--brain llm` 场景，docs/llm-tools.md）。

每个工具一个 py 文件丢进本包（tool/）即插即用，零改码：
用 `from .base import tool` + `@tool(...)` 装饰函数即可，`load_tools()` 会自动扫描收集。

扫描方式：`pkgutil.iter_modules` 遍历包内所有模块 → 收集模块命名空间里的 `Tool` 实例。
模块导入失败（依赖缺失/配置坏）→ 记告警跳过该模块，不致命——一个工具坏不影响别的。
"""
from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys

from .base import Tool, tool  # noqa: F401  工具模块统一从这里拿

log = logging.getLogger("tools")


def apply_network_policy():
    """工具包网络策略（2026-09-18，用户实测）：本机 Windows 系统代理（Clash 之类）配置了
    但进程不在跑时，`requests` 读注册表代理会连不上国内源（新浪金价/和风天气）——这是
    `requests` 默认 `trust_env=True` 的行为，代码里**没有写死任何代理地址**。
    处理：**未显式配置 HTTP_PROXY/HTTPS_PROXY → 设 `NO_PROXY=*` 绕过系统代理直连**
    （金价/天气都是国内可达源，直连更稳）；**显式配置了代理 → 尊重代理**（用户网络环境
    需要走代理时，设 HTTP_PROXY/HTTPS_PROXY 环境变量即可）。`load_tools()` 加载时执行一次，
    对进程内后续全部 requests 生效。
    """
    if not (os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
            or os.environ.get("http_proxy") or os.environ.get("https_proxy")):
        os.environ.setdefault("NO_PROXY", "*")
        os.environ.setdefault("no_proxy", "*")


def _scan_module(modname: str) -> list:
    """导入单个工具模块，返回其中的 Tool 实例列表。导入失败不抛（记告警跳过）。"""
    try:
        mod = importlib.import_module("%s.%s" % (__name__, modname))
    except Exception as e:
        log.warning("[tools] 模块 %s 导入失败，跳过：%s", modname, e)
        return []
    tools = []
    for obj in vars(mod).values():
        if isinstance(obj, Tool):
            tools.append(obj)
    return tools


def load_tools(names=None, logger=None) -> dict:
    """加载工具，返回 {name: Tool}。

    names: None / "all" / "" = 全部；否则逗号分隔的名字列表，只加载匹配者
    （未匹配到任何工具时打印可加载清单，帮助排查拼写）。
    logger: 可选的诊断打印对象（print 到控制台），None 则不打。
    """
    glog = logger or log.info
    apply_network_policy()          # 未配置代理 → NO_PROXY=* 绕注册表系统代理直连
    found = {}
    for _, modname, _ in pkgutil.iter_modules(__path__):
        for t in _scan_module(modname):
            if t.name in found:
                glog("[tools] 警告：工具名 %s 重复，后者覆盖" % t.name)
            found[t.name] = t
    if not names or str(names).strip() in ("", "all"):
        sel = dict(found)
    else:
        wanted = [n.strip() for n in str(names).split(",") if n.strip()]
        sel = {}
        for w in wanted:
            if w in found:
                sel[w] = found[w]
            else:
                glog("[tools] 未知工具名：%s（可用：%s）" % (w, ", ".join(sorted(found)) or "无"))
    if logger:
        logger("[tools] 已加载 %d/%d 个工具：%s" % (
            len(sel), len(found), ", ".join(sorted(sel)) or "无"))
    return sel
