# -*- coding: utf-8 -*-
"""llm+tools 苏联笑话：get_soviet_joke（复用 assistant 的 soviet-joke skill 语料，不走 MCP 更轻）。

参照 soviet-joke skill / tool/gold.py 的模式——语料《苏联东欧政治笑话选编》(1975) 原文在
assistant/.claude/skills/soviet-joke/corpus.md，确定性逻辑（随机挑 / 剥 `X、` 序号 / 避让
已讲过的 / 干净输出）全在语料读取里，Tool 只做薄封装：参数校验 → 从语料挑一条 → 返回可读
文本，模型只呈现、不造数、不评论现当代政治。

与 skill 的 tell.py **同源同构**（同一份 corpus.md、同一套格式不变量：主题剥 `X、` 序号、
正文逐字引用、末尾无多余空行），但支持按主题挑选 / 避让已讲过的标题——tell.py 是给
agent 模式"单条随机"的 CLI，本工具是给 llm 模式带参数的编程接口。语料是历史文献（1975
年小册子的转写，77 条），任何含该语料的输出都应作"历史笑话"呈现，不映射当代。

本模块只负责"把 corpus.md 包成 Tool 接口"：theme/avoid 参数校验、错误都返回可读文本，
绝不抛异常。
"""
from __future__ import annotations

import os
import random
import re

from .base import tool

# 语料位置：tool/ 在仓库根，assistant/.claude/skills/soviet-joke 是它的兄弟目录
# （代理策略统一在 tool/__init__.py `apply_network_policy()`，见 `load_tools`）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_REPO_ROOT, "assistant", ".claude", "skills", "soviet-joke", "corpus.md")

# 形如 "一、" 的序号前缀（tell.py 同款，主题标签必须剥掉）
_NUM_PREFIX = re.compile(r"^[一二三四五六七八九十]、")

# 主题：corpus.md 的四个 `# ` 分节（剥序号后），带可读别名（序号/关键词）
_THEMES = {
    "关于苏修新资产阶级特权阶级": ["1", "一", "特权", "阶级"],
    "关于假共产主义": ["2", "二", "共产"],
    "关于苏修推行霸权主义": ["3", "三", "霸权"],
    "“苏联和东欧人民反对修正主义的统治”": ["4", "四", "反对修正", "修正"],
}


def _load_jokes():
    """解析 corpus.md → [(clean_theme, title, [body_lines])]。结构/剥序号与 tell.py 同构。

    clean_theme 已剥 `X、` 前缀；body 保留原始行（含正文内部的空行），首尾空行由 _format 去掉。
    """
    with open(_CORPUS, encoding="utf-8") as f:
        raw = f.read()
    lines = raw.split("\n")

    jokes = []
    theme = ""
    title = None
    body = []

    def flush():
        nonlocal title, body
        if title is not None and body:
            jokes.append((_NUM_PREFIX.sub("", theme).strip(), title, body))
        title = None
        body = []

    for ln in lines:
        st = ln.strip()
        if st.startswith("# "):           # 主题分节（`# `）
            flush()
            theme = st[2:].strip()
        elif st.startswith("## "):        # 单条笑话（`## `）
            flush()
            title = st[3:].strip()
        elif title is not None:
            body.append(ln)               # 保留原始行（含正文内部的空行）

    flush()
    return jokes


def _format(clean_theme, title, body):
    """拼成规范输出（tell.py 同款不变量）：主题剥序号、正文逐字、末尾无多余空行。"""
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    body_text = "\n".join(body).rstrip("\n")
    return "【%s · 《%s》】\n\n%s" % (clean_theme, title, body_text)


def _query(theme, avoid):
    """核心查询：返回一条笑话文本，或可读错误文本（不抛异常）。"""
    if not os.path.isfile(_CORPUS):
        return "无法讲笑话：缺少语料 assistant/.claude/skills/soviet-joke/corpus.md"
    try:
        jokes = _load_jokes()
    except Exception as e:
        return "无法讲笑话：语料读取失败：%s" % e
    if not jokes:
        return "无法讲笑话：语料为空"

    pool = jokes
    if theme:
        target = None
        for key, aliases in _THEMES.items():
            if theme in aliases or theme in key or key in theme:
                target = key
                break
        if target is None:
            return "无法讲笑话：未知主题 %r（可选 一/二/三/四 或 特权/共产/霸权/反对修正）" % theme
        pool = [j for j in pool if j[0] == target]
        if not pool:
            return "无法讲笑话：主题 %r 下暂无笑话" % theme

    if avoid:
        banned = [a.strip() for a in avoid.split(",") if a.strip()]

        def _avoided(j):
            title = j[1]
            return any(b in title or title in b for b in banned)

        pool = [j for j in pool if not _avoided(j)]
        if not pool:
            return "无法讲笑话：已避让 %r，该范围下没有未讲过的了" % avoid

    clean_theme, title, body = random.choice(pool)
    return _format(clean_theme, title, list(body))   # 拷贝 body，别污染语料内存


@tool(
    "get_soviet_joke",
    "讲一条苏联笑话（1975《苏联东欧政治笑话选编》历史语料，逐字引用不改造）",
    {
        "theme": "[可选] 按主题挑：一/二/三/四（特权阶级/假共产主义/霸权主义/反对修正主义）或关键词，默认全库随机",
        "avoid": "[可选] 已讲过的笑话标题（或关键词），逗号分隔可多个，避免重复（用户说'再来一个'时用）",
    },
    explanation=("复用 assistant 的 soviet-joke skill 语料（历史文献，1975 年小册子转写，77 条）。"
                 "用户想听苏联笑话 / 东欧冷笑话 / 关于勃列日涅夫或赫鲁晓夫的笑话时调用。"
                 "输出是【主题 · 《标题》】+ 正文；讲完一条后用户说'再来一个' → 用 "
                 "avoid=<上一条《》里的标题> 再调，保证不重复。语料是历史笑话，当历史段子讲，"
                 "别拿它影射当代。"),
    present=("用户要听的是笑话原文本身：把【主题 · 《标题》】头和正文**完整逐字讲出来**"
             "（原文一字不改、一句都不能省；调用工具前你已经说过开场过渡了，拿到结果**直接进"
             "正文**，**不要**再重复一遍开场）；**不要**只评论/复述梗概、**不要**另编笑话。"
             "这条比平时长没关系，用户就是要听完整版。"),
    timeout=5.0,      # 本地文件读取，快
    max_result=1200,  # 最长正文 490 字 + 头，1200 留余量防截断笑点
)
def get_soviet_joke(params: dict) -> str:
    theme = (params.get("theme") or "").strip()
    avoid = (params.get("avoid") or "").strip()
    return _query(theme or None, avoid or None)
