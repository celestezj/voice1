# -*- coding: utf-8 -*-
"""XML 流式解析器（Alife XmlStreamParser 移植，docs/llm-tools.md §5.3）。

用途：`--brain llm` 模式下，模型在输出文本流里写 `<get_weather city="北京"/>`
这样的自闭合 XML 标签调用工具。本解析器逐字符喂入（feed(delta)），**边流边解析**：

- 已知工具标签（自闭合 `<name/>` 或成对 `<name>…</name>`）→ 产出 `ToolCall` 事件，
  标签本身**不进正文**；
- 未知成对标签 → 剥掉标签、内部文本保留为正文（如 `<answer>…</answer>` 透明容器）；
- 未知自闭合标签 / 注释（`<!-- … -->`）→ 整个丢弃；
- 实体（`&amp; &lt; &gt; &quot; &#34;` 等）→ 解码成字符进正文/属性值；
- 未闭合标签在 `flush()` 时丢弃（不执行工具，残留正文按容器透明处理）。

与现有流式出字解耦：解析器只认标签，正文照原 `_assistant_buf` 管线走。心态标记
`【心态：xxx】` 是普通文本（无尖括号），原样通过。

线程：只在 `_llm_loop` 流线程使用，无共享状态，天然线程安全。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'",
    "&#34;": '"', "&#38;": "&", "&#60;": "<", "&#62;": ">",
    "&#160;": " ", "&nbsp;": " ",
}
_NUM_ENT_RE = re.compile(r"&#(x?)([0-9a-fA-F]+);")


@dataclass
class ToolCall:
    """一个已完整闭合的工具调用。"""

    name: str
    attrs: dict = field(default_factory=dict)
    raw: str = ""            # 原始标签文本（诊断用）


class _Frame:
    __slots__ = ("name", "attrs", "parts")

    def __init__(self, name, attrs):
        self.name = name
        self.attrs = attrs
        self.parts = []      # 本标签内部已积累的正文片段


class ToolXmlParser:
    def __init__(self, tool_names=()):
        self._tools = set(tool_names) if tool_names else set()
        self.reset()

    def reset(self):
        self._stack = []        # 打开的成对标签（_Frame 栈）
        self._text_parts = []   # 顶层（栈空时）正文片段
        self._tag = None        # 正在解析的单个标签（dict）或 None
        self._ann = False       # 注释中
        self._ann_buf = []
        self._esc = False       # 实体解析中
        self._esc_buf = []
        self._esc_target = "text"   # 实体解码后送哪："text" 正文 / "attr" 属性值
        self._calls = []        # 本次 feed/flush 已完整闭合的工具调用

    # ---------------- 正文输出 ----------------
    def _push_text(self, s, out):
        """正文片段：顶层（栈空）立即可返回；在成对标签内则挂帧上、等帧闭合才释放。"""
        if self._stack:
            self._stack[-1].parts.append(s)
        else:
            self._text_parts.append(s)
            out.append(s)

    def _new_tag(self):
        return {"name": None, "mode": 0, "attrs": {}, "attr_name": None,
                "in_value": False, "name_buf": [], "attr_name_buf": [],
                "attr_value_buf": [], "raw": []}

    # ---------------- 主循环 ----------------
    def feed(self, delta: str):
        """喂入一段流式增量，返回 (新正文文本, 新工具调用列表)。"""
        out = []
        for ch in delta:
            self._step(ch, out)
        calls = list(self._calls)
        self._calls = []
        return "".join(out), calls

    def _step(self, ch, out):
        if self._ann:
            if ch == ">":                          # '>' 不进缓冲：判缓冲是否已以 "--" 收尾
                if "".join(self._ann_buf).endswith("--"):
                    self._ann = False
                    self._ann_buf = []
            else:
                self._ann_buf.append(ch)
            return

        if self._esc:
            self._esc_buf.append(ch)
            if ch == ";":
                self._flush_esc(out)
            elif ch in ('"', "<"):          # 畸形实体：原样吐出后重处理该字符
                self._flush_esc(out)
                self._step(ch, out)
            return

        if ch == "&":
            self._esc = True
            self._esc_buf = ["&"]
            self._esc_target = "attr" if (self._tag and self._tag["in_value"]) else "text"
            return

        if self._tag is None:
            if ch == "<":
                self._tag = self._new_tag()
            else:
                self._push_text(ch, out)
            return

        tag = self._tag
        tag["raw"].append(ch)

        if tag["in_value"]:
            if ch == '"':
                val = "".join(tag["attr_value_buf"])
                tag["attrs"][tag["attr_name"]] = val
                tag["attr_name"] = None
                tag["in_value"] = False
                tag["attr_value_buf"] = []
            else:
                tag["attr_value_buf"].append(ch)
            return

        if tag["name"] is None:             # 解析标签名
            if ch in " =":
                tag["name"] = "".join(tag["name_buf"]).lower() or None
                tag["name_buf"] = []
            elif ch == "/":                 # </name … 闭标签 或 <name/ 自闭合
                tag["name"] = "".join(tag["name_buf"]).lower() or None
                tag["name_buf"] = []
                tag["mode"] = 1 if tag["name"] is None else 2
            elif ch == ">":
                tag["name"] = "".join(tag["name_buf"]).lower() or None
                tag["name_buf"] = []
                self._finish_tag(out)
            elif ch == "!":                 # <!-- 注释（仅紧跟 < 时）
                if not tag["name_buf"]:
                    self._tag = None
                    self._ann = True
                    self._ann_buf = []
                else:
                    tag["name_buf"].append(ch)
            else:
                tag["name_buf"].append(ch)
            return

        # 标签名已定 → 解析属性/分隔
        if ch in " =":
            self._flush_attr_name(tag)
        elif ch == "/":
            self._flush_attr_name(tag)
            tag["mode"] = 2
        elif ch == ">":
            self._flush_attr_name(tag)
            self._finish_tag(out)
        elif ch == '"':
            if tag["attr_name"] is not None:
                tag["in_value"] = True
        elif ch == "!":
            self._tag = None
            self._ann = True
            self._ann_buf = []
        else:
            tag["attr_name_buf"].append(ch)

    def _flush_attr_name(self, tag):
        if tag["attr_name"] is None and tag["attr_name_buf"]:
            tag["attr_name"] = "".join(tag["attr_name_buf"]).lower()
            tag["attr_name_buf"] = []

    def _flush_esc(self, out):
        content = "".join(self._esc_buf)
        self._esc = False
        self._esc_buf = []
        target = self._esc_target
        decoded = _ENTITIES.get(content)
        if decoded is None:
            m = _NUM_ENT_RE.fullmatch(content)
            if m:
                try:
                    decoded = chr(int(m.group(2), 16 if m.group(1) else 10))
                except (ValueError, OverflowError):
                    decoded = None
        if decoded is not None:
            if target == "text":
                self._push_text(decoded, out)
            else:
                self._tag["attr_value_buf"].append(decoded)
        else:
            if target == "text":
                self._push_text(content, out)
            else:
                self._tag["attr_value_buf"].append(content)

    def _finish_tag(self, out):
        tag = self._tag
        self._tag = None
        name, mode, raw = tag["name"], tag["mode"], "".join(tag["raw"])
        if not name:
            return

        if mode == 2:                       # <name …/> 自闭合
            if name in self._tools:
                self._calls.append(ToolCall(name, dict(tag["attrs"]), raw))
            return

        if mode == 1:                       # </name> 闭标签
            idx = None
            for i in range(len(self._stack) - 1, -1, -1):
                if self._stack[i].name == name:
                    idx = i
                    break
            if idx is None:                 # 孤儿闭标签 → 丢弃
                return
            popped = self._stack[idx:]
            del self._stack[idx:]
            content = "".join(p for f in popped for p in f.parts)   # 含未闭合的中间孤儿帧
            if popped[0].name in self._tools:
                # 成对工具标签：内部文本丢弃（不是给 TTS 的正文），attrs 回灌
                self._calls.append(ToolCall(popped[0].name, dict(popped[0].attrs), raw))
            else:
                # 未知成对标签：透明容器，内部文本并入父/顶层正文
                if self._stack:
                    self._stack[-1].parts.append(content)
                else:
                    self._text_parts.append(content)
                    out.append(content)
            return

        # mode 0：<name …> 开标签（成对结构入栈）
        self._stack.append(_Frame(name, dict(tag["attrs"])))

    # ---------------- 收尾 ----------------
    def flush(self):
        """流结束收尾：残留的未完成实体/标签按容器透明处理；返回剩余正文与调用。

        未闭合的工具标签**不执行**（attrs 可能不完整），其内部文本丢弃。
        """
        out = []
        if self._esc:
            if self._esc_target == "text":
                self._push_text("".join(self._esc_buf), out)
            self._esc = False
            self._esc_buf = []
        self._ann = False
        self._ann_buf = []
        self._tag = None
        for frame in self._stack:
            content = "".join(frame.parts)
            if frame.name not in self._tools:
                self._text_parts.append(content)
                out.append(content)
        self._stack = []
        calls = list(self._calls)
        self._calls = []
        return "".join(out), calls
