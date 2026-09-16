# -*- coding: utf-8 -*-
"""LLM 模式工具注册表基础：Tool / @tool / 执行守卫。

设计（docs/llm-tools.md §5.1）：每个工具一个 py 文件丢进 tool/ 即插即用，零改码——
`__init__.py` 自动扫描包内所有模块、收集 `@tool` 注册的 Tool 实例。
工具执行统一走 `Tool.run()`：独立线程 + timeout 守卫（防某工具挂死拖住 LLM 流线程），
异常回灌错误文本、超长结果截断——LLM 只会看到"干净的字符串"。
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

_RESERVED = {"explanation", "timeout", "max_result"}


@dataclass
class Tool:
    """一个可被 LLM 调用的工具。

    fn: params dict -> 结果字符串。必须快（默认 10s 超时守卫）、纯函数优先、
        网络调用自己控超时。返回的字符串会原样送 LLM 加工（上限 max_result 字）。
    """

    name: str                       # XML 标签名（小写，如 get_weather）
    description: str                # 一行简介（进提示词）
    params: dict = field(default_factory=dict)      # 参数名 -> "类型 说明"（"[可选]" 前缀表可选）
    explanation: str = ""           # 详细用法说明（可选，进提示词）
    timeout: float = 10.0           # 执行超时（秒），防语音流卡死
    max_result: int = 800           # 结果最长字符，防爆上下文
    fn: Optional[Callable[[dict], str]] = None

    def run(self, kwargs: dict) -> tuple[bool, str]:
        """执行工具，带超时/异常/截断守卫。返回 (ok, text)；ok=False 时 text 为错误文本。

        独立 daemon 线程 + join(timeout)：fn 挂在网络/死循环上超时也能回来，
        超时文本回灌 LLM 让它优雅处理，不拖死语音流线程。
        """
        if self.fn is None:
            return False, "[工具错误: 工具 %s 未实现]" % self.name
        box = {}

        def _call():
            try:
                box["ret"] = self.fn(dict(kwargs))
            except Exception:
                box["err"] = traceback.format_exc(limit=1).strip().splitlines()[-1]

        t = threading.Thread(target=_call, name="tool:%s" % self.name, daemon=True)
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            return False, "[工具错误: %s 执行超时（>%ss），请换个问法或稍后再试]" % (self.name, self.timeout)
        if "err" in box:
            return False, "[工具错误: %s 出错：%s]" % (self.name, box["err"])
        text = box.get("ret")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if len(text) > self.max_result:
            text = text[: self.max_result] + "…（结果已截断）"
        return True, text

    def to_prompt_doc(self) -> str:
        """生成提示词里的一条工具文档（一行，照 docs/llm-tools.md §5.2）。"""
        parts = ["- <%s" % self.name]
        for pname, pdoc in self.params.items():
            parts.append('%s="%s"' % (pname, pdoc))
        parts.append("/> : %s" % self.description)
        return " ".join(parts)


def tool(name: str, description: str, params: dict = None, **kwargs):
    """@tool 装饰器：把函数注册成 Tool 直接返回（模块命名空间里就是 Tool 实例）。

    - params 可用位置 dict，也可用关键字直接给（保留字 explanation/timeout/max_result 除外）：
        @tool("get_weather", "查询天气", city="城市名", days="[可选] 预报天数")
      等价于 params={"city": "城市名", "days": "[可选] 预报天数"}。
    - 保留字：explanation / timeout / max_result。
    """
    if params is None:
        params = {k: v for k, v in kwargs.items() if k not in _RESERVED}
    explanation = kwargs.get("explanation", "")
    timeout = kwargs.get("timeout", 10.0)
    max_result = kwargs.get("max_result", 800)

    def deco(fn):
        return Tool(
            name=name,
            description=description,
            params=params,
            explanation=explanation,
            timeout=timeout,
            max_result=max_result,
            fn=fn,
        )

    return deco
