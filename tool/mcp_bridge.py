# -*- coding: utf-8 -*-
"""MCP 工具桥接：把 MCP server（stdio / streamable HTTP）枚举的工具转成 tool/ 的 Tool 实例。

背景：llm+tools 模式走 XML 内联工具（`<get_weather city="北京"/>`），工具是同步 `Tool.fn`。
MCP server 是 JSON-RPC（stdio 子进程 / streamable HTTP），SDK 是 asyncio 客户端。本模块做
桥接：**一个常驻后台事件循环线程**持有各 server 的 ClientSession 连接，把 MCP 工具的
input_schema（JSON Schema）转成 Tool.params（与本地工具同构），fn 用
`asyncio.run_coroutine_threadsafe` 桥到后台循环调 `session.call_tool`，结果统一格式化成文本
（content 里的 text 块优先，structuredContent 兜底）。

用法（一般不用直接调，`tool/__init__.py::load_tools` 在 `--tools` 含 mcp/all 时自动接）：

    from .mcp_bridge import load_mcp_tools, close_mcp_tools
    tools = load_mcp_tools()        # 读 tool/mcp.local.json，逐个 server 连接 + 枚举工具
    tools["search_query"].run({"q": "..."})   # 与本地 Tool 同构，模型照常 XML 调用
    close_mcp_tools()               # finally / atexit 收尾，断开全部连接

配置 `tool/mcp.local.json`（gitignored，含机器路径/凭据，参照 .mcp.json 同款格式，两个 server
示例）：

    {
      "mcpServers": {
        "search": {                 # stdio 子进程：command 必填
          "command": "C:/path/to/.venv-search/Scripts/python.exe",
          "args": ["-m", "search_mcp"],
          "env": {"SOME_VAR": "1"}  # 可选，透传子进程环境（继承当前环境再覆盖）
        },
        "weather": {                # streamable HTTP：url 必填（二选一）
          "url": "http://127.0.0.1:8080/mcp"
        }
      }
    }

- 工具名暴露为 `<server>_<工具名>`（如 search 里的 query → `search_query`），XML 标签名做了
  安全化（非法字符 → `_`，数字开头补 `m_`）。
- 每个 server 一个后台协程常驻（`async with` 保活连接），`close_mcp_tools()` 置 stop 事件
  优雅断开。
- 配置路径可用环境变量 `VOICE1_MCP_CONFIG` 覆盖（headless 测试 / 非默认布局用）。
- 网络策略：HTTP 型 server 复用 tool/__init__.py 的 apply_network_policy（默认 NO_PROXY=*）。
"""
from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager

from .base import Tool

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_HERE, "mcp.local.json")

# 连接/调用超时（秒）
_CONNECT_TIMEOUT = 15.0
_CALL_TIMEOUT = 60.0
_SHUTDOWN_GRACE = 3.0

# XML 标签名只允许 [A-Za-z0-9_.-] 且不能数字/横线开头
_XML_OK = re.compile(r"[^A-Za-z0-9_\-.]")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_sessions = {}      # server名 -> ClientSession（连接建立后登记，供 close 用）
_stop_events = {}   # server名 -> asyncio.Event（close 时置位，让常驻协程优雅退出）
_timeouts = {}      # server名 -> 调用超时（秒），来自配置的 timeout 字段
_tools = {}         # 暴露名 -> Tool
_errors = {}        # server名 -> 连接错误文本（诊断用）


def _log(logger, msg):
    (logger or (lambda m: None))(msg)


# ---------------------------------------------------------------- 后台循环
def _ensure_loop():
    """懒启动后台事件循环线程（daemon，进程退出自动收）。"""
    global _loop, _loop_thread
    if _loop is not None:
        return
    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True,
                                    name="mcp-bridge-loop")
    _loop_thread.start()


def _sanitize(name: str) -> str:
    """暴露名安全化：非法字符 → `_`，数字/横线开头补 `m_`（XML 标签名约束）。"""
    s = _XML_OK.sub("_", name)
    if not s or s[0].isdigit() or s[0] in "-.":
        s = "m_" + s
    return s


# ---------------------------------------------------------------- schema → params
def _schema_to_params(schema):
    """MCP input_schema（JSON Schema object）→ tool/ 的 params dict。

    约定与本地 Tool 一致：`"参数名": "[可选] 类型 说明"`。
    """
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}
    required = set(schema.get("required") or [])

    def _ptype(pdoc):
        if isinstance(pdoc, dict):
            t = pdoc.get("type")
            if t:
                return t
            any_of = pdoc.get("anyOf")
            if isinstance(any_of, list):
                ts = [_ptype(x) for x in any_of if isinstance(x, dict) and x.get("type")]
                if ts:
                    return "|".join(dict.fromkeys(ts))
            enum = pdoc.get("enum")
            if isinstance(enum, list) and enum:
                return "枚举:" + "/".join(str(e) for e in enum)
        return "string"

    params = {}
    for pname, pdoc in props.items():
        if not isinstance(pdoc, dict):
            continue
        desc = (pdoc.get("description") or "").strip()
        ptype = _ptype(pdoc)
        marker = "" if pname in required else "[可选] "
        params[pname] = ("%s%s" % (marker, ptype)
                         + (" " + desc if desc else ""))
    return params


# ---------------------------------------------------------------- 结果格式化
def _format_result(result) -> str:
    """CallToolResult → 可读文本。text 块优先，structuredContent 兜底。"""
    content = getattr(result, "content", None) or []
    texts = []
    for block in content:
        t = getattr(block, "text", None)
        if t:
            texts.append(t)
    if texts:
        return "\n".join(texts)
    structured = getattr(result, "structuredContent", None)
    if structured:
        return json.dumps(structured, ensure_ascii=False)
    return "" if result is None else str(result)


def _resolve_stdio(cfg):
    """把 .mcp.json 风格的 stdio 配置解析成 StdioServerParameters（与 agent.py 同款解耦）。

    - `python/py/python3/pythonw` → 本进程解释器 `sys.executable`（用 voice-asr 的 python，
      避免 PATH 里别的 python 缺 mcp 依赖——author.py 等脚本靠它 import mcp）；
    - 相对 `command`/`args` 视为相对**仓库根**（tool/ 的上级，mcp.local.json 所在仓库的根），
      转绝对路径，跟"从哪启动 voice_dialogue"解耦；
    - `-m <模块名>` 形态：`-m` 之后的参数是模块名（如 `-m search_mcp`），不是文件路径，保持
      原样——否则被误当相对路径转绝对（agent.py 实测过 search MCP 挂载失败根因）。
    """
    from mcp import StdioServerParameters
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tool/ 的上级
    cmd = (cfg.get("command") or "").strip()
    if cmd.lower() in ("python", "python3", "pythonw", "py"):
        cmd = sys.executable
    elif cmd and not os.path.isabs(cmd):
        cmd = os.path.abspath(os.path.join(root, cmd))
    args = []
    module_next = False
    for a in (cfg.get("args") or []):
        if module_next:
            module_next = False
            args.append(a)
        elif a == "-m":
            module_next = True
            args.append(a)
        else:
            args.append(os.path.abspath(os.path.join(root, a))
                        if a and not os.path.isabs(a) and not a.startswith("-") else a)
    return StdioServerParameters(command=cmd, args=args, env=cfg.get("env"))


# ---------------------------------------------------------------- 每个 server 的常驻连接
async def _serve_server(name, cfg):
    """为一个 server 建连接 → 枚举工具 → 常驻保活，直到 close 置位。"""
    stop = asyncio.Event()
    _stop_events[name] = stop
    timeout = cfg.get("timeout")
    if isinstance(timeout, (int, float)) and timeout > 0:
        _timeouts[name] = float(timeout)
    try:
        if "url" in cfg:
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client(cfg["url"]) as (r, w):
                async with _open_session(name, r, w):
                    await stop.wait()
        else:
            from mcp.client.stdio import stdio_client
            params = _resolve_stdio(cfg)
            async with stdio_client(params) as (r, w):
                async with _open_session(name, r, w):
                    await stop.wait()
    except Exception as e:   # 单个 server 失败不拖垮其他
        _errors[name] = "%s" % e
        _log(_glog, "[mcp] server %r 连接/枚举失败：%s" % (name, e))


@asynccontextmanager
async def _open_session(name, reader, writer):
    """ClientSession 生命周期：initialize（2.x 不自动握手，必须先显式 init）→ list_tools
    → 转 Tool 实例 → 注册 session → 让出给保活（连接在此挂住，直到外层 stop.wait() 返回）。"""
    from mcp.client.session import ClientSession
    async with ClientSession(reader, writer) as session:
        await session.initialize()
        listed = await session.list_tools()
        for mtool in listed.tools:
            _install_tool(name, mtool, session)
        _sessions[name] = session
        _log(_glog, "[mcp] server %r 已连接，工具：%s" % (
            name, ", ".join(sorted(getattr(t, "name", "") for t in listed.tools)) or "无"))
        yield


def _install_tool(server_name, mtool, session):
    """把一个 MCP 工具转成 Tool 实例并登记到 _tools[暴露名]。"""
    display = _sanitize("%s_%s" % (server_name, getattr(mtool, "name", "tool")))
    desc = (getattr(mtool, "description", None) or "").strip() or "%s 的 MCP 工具" % server_name
    mcp_tool_name = getattr(mtool, "name", display)
    schema = getattr(mtool, "input_schema", None) or {}
    params = _schema_to_params(schema)
    timeout = _timeouts.get(server_name, _CALL_TIMEOUT)

    def fn(p: dict) -> str:
        return _call_tool(server_name, mcp_tool_name, p)

    t = Tool(
        name=display,
        description=desc,
        params=params,
        explanation=("%s 是 MCP server「%s」暴露的工具（MCP 桥接，见 tool/mcp_bridge.py）。"
                     "按上面参数给值即可调用。" % (display, server_name)),
        timeout=timeout,
        max_result=8000,
        fn=fn,
    )
    _tools[display] = t


def _call_tool(server_name, mcp_tool_name, params) -> str:
    """同步 fn → 后台循环的 call_tool。失败返回可读错误文本（绝不抛异常，模型只呈现）。"""
    timeout = _timeouts.get(server_name, _CALL_TIMEOUT)
    if _loop is None or server_name not in _sessions:
        return "[工具错误: MCP server %r 未连接（%s）]" % (
            server_name, _errors.get(server_name, "已关闭"))
    fut = asyncio.run_coroutine_threadsafe(
        _do_call(server_name, mcp_tool_name, params), _loop)
    try:
        result = fut.result(timeout=timeout)
        return _format_result(result)
    except concurrent.futures.TimeoutError:
        return "[工具错误: %s_%s 调用超时（>%ss）]" % (server_name, mcp_tool_name, timeout)
    except Exception as e:
        return "[工具错误: %s_%s：%s]" % (server_name, mcp_tool_name, e)


async def _do_call(server_name, mcp_tool_name, params):
    session = _sessions.get(server_name)
    if session is None:
        raise RuntimeError("session 已断开")
    result = await session.call_tool(mcp_tool_name, arguments=params)
    if getattr(result, "isError", False):
        raise RuntimeError(_format_result(result) or "MCP server 返回错误")
    return result


# ---------------------------------------------------------------- 对外入口
_glog = None   # 模块级 logger，供后台线程（无参签名）用


def load_mcp_tools(cfg_path=None, logger=None):
    """读取 mcp.local.json，连接全部 server 并枚举工具。返回 {暴露名: Tool}。

    cfg_path 缺省取 VOICE1_MCP_CONFIG 或 tool/mcp.local.json；文件不存在/无 mcpServers →
    返回 {}（不抛异常）。已连接过再调（重复 --tools 场景）返回已有结果不重复连接。
    """
    global _glog
    _glog = logger
    glog = logger or (lambda m: None)

    if _tools:
        return dict(_tools)   # 幂等：重复调用直接返回已枚举结果

    path = cfg_path or os.environ.get("VOICE1_MCP_CONFIG") or _DEFAULT_CONFIG
    if not os.path.exists(path):
        glog("[mcp] 未找到 %s，跳过 MCP 工具（复制 tool/mcp.local.json 的示例配置即可启用）"
             % path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        glog("[mcp] 读取 %s 失败：%s" % (path, e))
        return {}
    servers = (cfg or {}).get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        glog("[mcp] %s 无 mcpServers 配置，跳过" % path)
        return {}

    _ensure_loop()
    asyncio.run_coroutine_threadsafe(_connect_all(servers, glog), _loop).result(
        timeout=_CONNECT_TIMEOUT + 2)
    glog("[mcp] 已加载 %d 个 MCP 工具（%s）" % (
        len(_tools), ", ".join(sorted(_tools)) or "无"))
    return dict(_tools)


async def _connect_all(servers, glog):
    """为每个 server 起常驻连接协程，等全部首次就绪（成功或失败）。

    遵循 .mcp.json 约定：`disabled: true` 的 server 跳过（不连接不枚举）；`timeout` 字段
    作为该 server 工具的执行/调用超时。
    """
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            glog("[mcp] server %r 配置不是对象，跳过" % name)
            _errors[name] = "配置不是对象"
            continue
        if cfg.get("disabled"):
            glog("[mcp] server %r 配置 disabled=true，跳过（改 tool/mcp.local.json 去掉该字段启用）" % name)
            _errors[name] = "disabled（配置跳过）"
            continue
        if not (cfg.get("url") or cfg.get("command")):
            glog("[mcp] server %r 缺少 url 或 command，跳过" % name)
            _errors[name] = "缺少 url/command"
            continue
        asyncio.create_task(_serve_server(name, cfg))
    deadline = time.monotonic() + _CONNECT_TIMEOUT
    while time.monotonic() < deadline:
        if len(_sessions) + len(_errors) >= len(servers):
            break
        await asyncio.sleep(0.05)


def close_mcp_tools():
    """断开全部 MCP 连接、停后台循环。幂等，可反复调 / atexit 兜底。"""
    global _loop, _loop_thread
    if _loop is None:
        return
    async def _shutdown():
        for ev in list(_stop_events.values()):
            ev.set()
        await asyncio.sleep(0.2)
        tasks = [t for t in asyncio.all_tasks(_loop)
                 if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    fut = asyncio.run_coroutine_threadsafe(_shutdown(), _loop)
    try:
        fut.result(timeout=_SHUTDOWN_GRACE)
    except Exception:
        pass
    _loop.call_soon_threadsafe(_loop.stop)
    if _loop_thread:
        _loop_thread.join(timeout=_SHUTDOWN_GRACE)
    _loop = None
    _loop_thread = None
    _sessions.clear()
    _stop_events.clear()
    _timeouts.clear()
    _tools.clear()
    _errors.clear()


atexit.register(close_mcp_tools)
