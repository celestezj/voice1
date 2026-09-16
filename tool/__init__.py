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
import pkgutil
import sys

from .base import Tool, tool  # noqa: F401  工具模块统一从这里拿

log = logging.getLogger("tools")


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
