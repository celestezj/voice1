# -*- coding: utf-8 -*-
"""ClaudeAgentClient：本地 claude code（claude-agent-sdk）常驻会话适配器。

agent 模式的大脑：controller 把 ASR 文本交给它，它把 agent 的**最终结论**返回
（`ResultMessage.result`；中间的工具调用/思考文本不进 TTS）。

设计要点（详见 docs/agent-integration.md）：
- **常驻会话**：SDK 自己持一个常驻 claude 进程，多轮 `query()` 复用，不冷启动
  （connect 冷启动只一次，实测 ≈0.6s；热查询 ≈2s）。
- **打断 = ESC**：`abort()` → `client.interrupt()`，中断当前回合、**进程/会话存活**、
  历史保留，绝不死进程换上下文。被中断回合以 `ResultMessage(subtype=
  'error_during_execution')` 干净收尾。
- **旁路 controller 自实现历史/压缩/系统提示词**：上下文在 claude 会话里；人格用
  assistant 目录的 CLAUDE.md——**实测 cwd 的 CLAUDE.md 不会自动当人格加载**，连接时
  读文件内容显式传 `system_prompt`。
- **跨进程续会话**：session_id（UUID）落盘（默认 sessions/agent_session_id.txt，
  已 gitignore），重启带 `resume=True` 续上次上下文（对应 `claude --resume`）。
- **延迟治理（2026-09-10 实测）**：① `max_thinking_tokens=0` 关思考预算——模型走方舟
  `ark-code-latest` 且 CLI 不识别（`unrecognized_model`）时按超大默认 thinking 先"想"约 45s，
  实测关掉后同查询 48.8s→1.9s；② `_do_query` 带看门狗超时（`query_timeout`，默认 90s），
  超时中断回合并报错，绝不无限挂起；③ stderr 环形缓存 + 报错时 dump，不再全吞；④
  `close()` 先直接 interrupt 在途回合，不留脏回合给下次 resume。

线程模型：本类持有**常驻 asyncio 事件循环线程**。`submit()/abort()/close()` 线程安全
（`asyncio.run_coroutine_threadsafe` 桥接；内部 worker 串行化，保证一次只有一个 query
在飞、abort 先收尾再续）。回调（on_result/on_partial）在**循环线程**执行，须快速返回、
不能阻塞循环——controller 侧只做持锁快操作。
"""
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from collections import deque

from claude_agent_sdk import (ClaudeSDKClient, ClaudeAgentOptions,
                              AssistantMessage, ResultMessage, StreamEvent)

# 默认 agent 工作目录（assistant 人格目录，repo 根的 assistant/）——主程序可 --agent-dir 覆盖
_DEFAULT_AGENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assistant")
# 默认放行工具白名单：SDK 会话无交互终端，permission_mode=default 下未放行的工具调用
# 会被系统自动拒绝（agent 只能让用户"去终端点允许"，但无终端可点）。技能要真跑起来须在此
# 放行底层工具。传显式 allowed_tools 时覆盖。
#
# 实测（claude 2.1.266 Windows）：
# - shell 有两个都可用——**PowerShell** 与 **Bash**（Git Bash）。模型可能任选其一：
#   读了 SKILL.md（写 `bash fetch.sh`）的模型会走 Bash；没细读的会直接 PowerShell 跑 python。
#   当初白名单只有 PowerShell，走 Bash 的会话被自动拒绝 → 模型报"运行脚本被拦住了"
#   （措辞与强制禁 shell 探针一字不差）。故两个 shell 都放行。
# - **Skill** 工具实测即使不在白名单也不会被拒（技能发现即放行），显式列出更稳。
# - **WebFetch/WebSearch** 兜底：模型在脚本被拒/失败时会退到"用网页查"；放行避免二次拒绝。
_DEFAULT_ALLOWED_TOOLS = [
    "PowerShell", "Bash",          # 双 shell：模型可能任选其一（Bash 实测 Windows 也能跑）
    "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch",       # 网页兜底（脚本被禁时模型会退到网页查）
    "Skill",                       # 技能加载（默认发现；显式列出防误拒）
]
_DEFAULT_PERSONA_FILE = "CLAUDE.md"        # 人格文件（人格唯一事实源，显式传 system_prompt）
_DEFAULT_SESSION_FILE = os.path.join(      # session_id 落盘（--agent-resume 续会话用）
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sessions", "agent_session_id.txt")


class ClaudeAgentClient:
    """常驻 claude 会话的薄封装（asyncio 循环线程 + 串行 worker）。"""

    def __init__(self, *, cwd=None, persona_file=None, session_id_file=None,
                 resume=False, permission_mode="default", allowed_tools=None,
                 disallowed_tools=None, model=None, connect_timeout=90.0,
                 include_partial_messages=True,
                 max_thinking_tokens=0, query_timeout=90.0,
                 disable_mcp=False,        # True=不挂任何 MCP；默认读 <agent 目录>/.mcp.json 全部挂上
                 on_result=None, on_partial=None, on_error=None, debug=False):
        self._cwd = cwd or _DEFAULT_AGENT_DIR
        self._persona_file = persona_file or os.path.join(self._cwd, _DEFAULT_PERSONA_FILE)
        self._session_id_file = session_id_file or _DEFAULT_SESSION_FILE
        self._resume = bool(resume)
        self._permission_mode = permission_mode
        self._allowed_tools = (list(allowed_tools) if allowed_tools is not None
                               else list(_DEFAULT_ALLOWED_TOOLS))
        self._disallowed_tools = list(disallowed_tools or [])
        self._model = model
        self._connect_timeout = connect_timeout
        self._include_partial = bool(include_partial_messages)
        # 思考预算（默认 0=关闭）：模型走方舟 ark-code-latest 且 CLI 不认识它
        # （stderr 见 [claude-code:unrecognized_model]）→ 按超大默认 thinking 预算先"想"
        # 约 45s 才开口，实测同查询关 thinking 后 48.8s→1.9s（2026-09-10 复现）。
        # 语音助手延迟优先，默认关；要质量可给预算值（如 2048）。
        self._max_thinking_tokens = int(max_thinking_tokens)
        # 单回合看门狗（秒）：receive 等 ResultMessage 超时 → 中断并报错，绝不无限挂起
        # （曾实测 resumed 会话被中断残留污染后静默 2-3 分钟无任何事件）。
        self._query_timeout = float(query_timeout)
        # MCP：默认从 <agent 目录>/.mcp.json 自动发现全部 server（和 Claude Code 同一配置来源），
        # 新 MCP 只需往 .mcp.json 加一段，无需改码/加参数；disable_mcp=True 则一律不挂
        # （要只关某一个，直接改 .mcp.json 删掉那段即可）。
        self._disable_mcp = bool(disable_mcp)
        self._stderr_buf = deque(maxlen=300)     # CLI 输出环形缓存（诊断盲区兜底）
        self._on_result = on_result
        self._on_partial = on_partial
        self._on_error = on_error
        self._debug = debug

        self._session_id = None            # 本会话 UUID（resume 时读旧值）
        self._loop = None                  # 常驻事件循环（循环线程内）
        self._loop_thread = None
        self._client = None
        self._pending = None               # asyncio.Queue（循环内，worker 消费）
        self._worker_task = None
        self._inflight = None              # 当前在飞的 _do_query 任务（循环内，仅 worker 触碰）
        self._ready = threading.Event()
        self._connect_error = None
        self._closed = False

    # ---------------- 会话 id（UUID；落盘供 --agent-resume） ----------------
    @property
    def session_id(self):
        return self._session_id

    @property
    def cwd(self):
        return self._cwd

    def _load_or_create_session_id(self):
        """resume=True 且落盘文件存在 → 读回；否则新建 UUID 并落盘（中途崩溃也可续）。"""
        sid = None
        if self._resume:
            try:
                with open(self._session_id_file, "r", encoding="utf-8") as f:
                    sid = f.read().strip()
                uuid.UUID(sid)             # 校验格式（SDK 强制合法 UUID）
            except Exception:
                sid = None
        if not sid:
            sid = str(uuid.uuid4())
            try:
                d = os.path.dirname(self._session_id_file)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self._session_id_file, "w", encoding="utf-8") as f:
                    f.write(sid)
            except Exception as e:
                print("[agent] 会话 id 落盘失败（重启将无法续会话）: %s" % e, flush=True)
        return sid

    def _load_persona(self):
        """读人格文件（CLAUDE.md）作为 system_prompt（cwd 的 CLAUDE.md 不会自动加载）。"""
        try:
            with open(self._persona_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print("[agent] 人格文件 %s 读取失败（将用空 system_prompt）: %s"
                  % (self._persona_file, e), flush=True)
            return ""

    # ---------------- 生命周期 ----------------
    def start(self):
        """起常驻循环线程并连接 claude。冷启动只发生这一次（阻塞到连上）。"""
        if self._closed:
            raise RuntimeError("agent 已关闭")
        self._session_id = self._load_or_create_session_id()
        self._loop_thread = threading.Thread(target=self._run_loop, name="agent-loop",
                                             daemon=True)
        self._loop_thread.start()
        if not self._ready.wait(self._connect_timeout):
            raise RuntimeError("claude agent 连接超时（%.0fs）" % self._connect_timeout)
        if self._connect_error:
            raise RuntimeError("claude agent 连接失败: %s" % self._connect_error)

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._connect())
            self._ready.set()
            loop.run_forever()
        except Exception as e:             # 连接失败也要通知 start()（不悬挂）
            self._connect_error = e
            self._ready.set()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def _build_mcp_servers(self):
        """组装要挂给 claude 会话的 MCP server 配置。

        默认源 = <agent 目录>/.mcp.json（与 Claude Code 同一套配置），新增 MCP 往里加一段即可，
        无需改代码/加参数；要只关某一个，直接删 .mcp.json 里那段。`--no-mcp`（disable_mcp=True）
        则整个 MCP 功能都不挂。`command` 若是 python/python3/pythonw/py 统一换成跑本 agent 的
        python（voice-asr），避免 Windows 上 conda/base 串包；非 python 命令（node 等）原样保留。
        """
        if self._disable_mcp:
            return None
        cfg_path = os.path.join(self._cwd, ".mcp.json")   # 例：assistant/.mcp.json
        raw = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    raw = json.load(f).get("mcpServers", {}) or {}
            except Exception as e:
                print("[agent] 读取 MCP 配置失败 %s：%s" % (cfg_path, e), flush=True)
        selected = {}
        for name, cfg in (raw or {}).items():
            cfg = dict(cfg)
            cmd = (cfg.get("command") or "").strip().lower()
            if cmd in ("python", "python3", "pythonw", "py"):
                cfg["command"] = sys.executable
            # 相对 args 视为相对 agent 目录（.mcp.json 所在处），写成绝对路径，跟启动目录解耦
            # （防从别处 `python voice_dialogue.py` 时 MCP 子进程找不到脚本）。
            args = cfg.get("args") or []
            cfg["args"] = [os.path.abspath(os.path.join(self._cwd, a)) if a and not os.path.isabs(a)
                           and not a.startswith("-") else a for a in args]
            selected[name] = cfg
        if self._debug:
            print("[agent] MCP servers: %s" % (", ".join(selected) or "（无）"), flush=True)
        return selected or None

    async def _connect(self):
        persona = self._load_persona()

        def _stderr_sink(line):
            # 环形缓存 CLI 输出（曾全吞导致查不出卡因）：默认不刷屏，
            # 报错/超时时 _notify_error 会 dump 尾部；--debug 则实时打印。
            self._stderr_buf.append(line)
            if self._debug:
                print("[agent-cli]", line, flush=True)

        mcp_servers = self._build_mcp_servers()
        # MCP 工具名是动态的（mcp__<server>__<tool>），静态白名单覆盖不到——SDK 会话无
        # 交互终端，permission_mode=default 下未预放行的工具调用会被自动拒绝（与当初
        # shell 被拦同因，agent 只会说"工具没放行"）。按启用的 server 名补
        # `mcp__<name>__*` 模式自动放行：只放行 .mcp.json 里实际配的 server、不放开
        # 用户全局 MCP；新增 MCP 无需改码（server 名来自 json 键）。
        allowed_tools = list(self._allowed_tools)
        if mcp_servers:
            allowed_tools += ["mcp__%s__*" % name for name in mcp_servers]

        opts = ClaudeAgentOptions(
            cwd=self._cwd,
            system_prompt=persona,                     # cwd 的 CLAUDE.md 不自动加载，须显式传
            session_id=None if self._resume else self._session_id,
            resume=self._session_id if self._resume else None,
            permission_mode=self._permission_mode,
            allowed_tools=allowed_tools,
            disallowed_tools=self._disallowed_tools,
            mcp_servers=mcp_servers,                     # 见 _build_mcp_servers（默认 .mcp.json 全量）
            model=self._model,
            include_partial_messages=self._include_partial,  # True：控制台流式出字；TTS 仍取最终结论
            max_thinking_tokens=self._max_thinking_tokens,  # 关默认超大思考预算（真凶，见 __init__）
            stderr=_stderr_sink,                        # 缓存 + 可选打印，不再吞
        )
        self._client = ClaudeSDKClient(options=opts)
        await self._client.connect()
        self._pending = asyncio.Queue()
        self._inflight = None
        self._worker_task = asyncio.create_task(self._worker())
        if self._debug:
            print("[agent] 已连接（session=%s，cwd=%s%s）"
                  % (self._session_id, self._cwd, "，resume 续上次会话" if self._resume else ""),
                  flush=True)

    def close(self):
        """优雅关闭：中断在途回合 + 断开 claude + 停循环线程。

        先**直接 interrupt**（不经队列）——worker 若卡在 receive_response 里根本处理不了
        ("close",)；直接 ESC 保证 CLI 回合干净收尾，**不留脏回合给下次 resume**
        （中断残留污染下一轮 receive 的实测根因，曾致 resumed 会话静默 2-3 分钟）。
        """
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._client is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._client.interrupt(),
                                                 self._loop).result(timeout=3)
            except Exception:
                pass
        if self._loop is not None and self._pending is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._pending.put(("close",)),
                                                 self._loop).result(timeout=3)
            except Exception:
                pass
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

    # ---------------- 对外接口（线程安全，非阻塞） ----------------
    def set_callbacks(self, on_result=None, on_partial=None, on_error=None):
        """换回调（controller.set_agent 用；须在 start() 前调用，避免循环线程竞态）。"""
        if on_result is not None:
            self._on_result = on_result
        if on_partial is not None:
            self._on_partial = on_partial
        if on_error is not None:
            self._on_error = on_error

    def submit(self, text, ctx=None):
        """提交一句用户文本给 agent（非阻塞）。最终结论经 on_result(ctx, text, is_error)。"""
        if self._closed or self._pending is None:
            return
        asyncio.run_coroutine_threadsafe(self._pending.put(("query", text, ctx)), self._loop)

    def abort(self):
        """中断当前回合（等价交互式 claude 的 ESC）。进程/会话存活，非阻塞。"""
        if self._closed or self._pending is None:
            return
        asyncio.run_coroutine_threadsafe(self._pending.put(("abort",)), self._loop)

    # ---------------- 循环内部：worker 串行化 ----------------
    async def _worker(self):
        """串行消费：一次只处理一件事；query 前先等上一回合收尾，abort 先收尾再续。

        队列顺序由提交方保证（controller 先 abort 再 submit）；worker 在此兜底——
        abort 到达时中断在飞 query 并 drain 干净，随后才处理下一个 query，
        避免打断后残留消息污染下一轮 receive（实测 round3 result=None 的成因）。
        """
        while True:
            try:
                item = await self._pending.get()
            except Exception:
                return
            kind = item[0]
            try:
                if kind == "close":
                    break
                elif kind == "abort":
                    await self._abort_inflight()
                elif kind == "query":
                    _, text, ctx = item
                    if self._inflight is not None:     # 防御：上一回合未收尾
                        await self._inflight
                    self._inflight = asyncio.create_task(self._do_query(text, ctx))
                    try:
                        await self._inflight
                    except Exception as e:
                        self._notify_result(ctx, None, True)
                        self._notify_error(e)
                    finally:
                        self._inflight = None
            except Exception as e:
                print("[agent] worker 异常: %s" % e, flush=True)
        # close：断开连接（进程结束）+ 停循环
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        try:
            self._loop.stop()
        except Exception:
            pass

    async def _abort_inflight(self):
        if self._inflight is None:
            return
        try:
            await self._client.interrupt()             # ESC 语义：不 kill 进程
        except Exception:
            pass
        try:
            await self._inflight                        # 旧回合收尾（error_during_execution）
        except Exception:
            pass
        self._inflight = None

    async def _do_query(self, text, ctx):
        await self._client.query(text, session_id=self._session_id)
        timeout = self._query_timeout

        async def _drain():
            """迭代本回合响应流直到 ResultMessage（流式出字 + 最终结论回调）。"""
            t0 = time.monotonic()
            first_txt = None
            async for msg in self._client.receive_response():
                if isinstance(msg, StreamEvent) and self._on_partial is not None \
                        and self._include_partial:
                    # 流式增量：include_partial_messages=True 时 CLI 发原始 Anthropic
                    # API 流事件，文本增量在 content_block_delta.text_delta（实测格式）。
                    # 逐段回调给上层做控制台流式出字；TTS 仍只取最终结论（ResultMessage）。
                    ev = msg.event
                    if (isinstance(ev, dict) and ev.get("type") == "content_block_delta"):
                        d = ev.get("delta") or {}
                        # 跳过纯空白增量（\n 等）：agent 常逐段吐换行，若不过滤，
                        # \n 也会触发一次控制台原地刷新 + 换行文本反复重写 = 刷屏
                        if (d.get("type") == "text_delta" and d.get("text")
                                and d["text"].strip()):
                            if first_txt is None:
                                first_txt = time.monotonic()
                                if self._debug:
                                    print("[agent] 首文本 %.1fs（query 发出后）" % (first_txt - t0),
                                          flush=True)
                            try:
                                self._on_partial(ctx, d["text"])
                            except Exception:
                                pass
                elif isinstance(msg, ResultMessage):
                    # 延迟探针（--debug）：区分「agent 生成慢」vs「文本已出但 ResultMessage
                    # 被 CLI 扣住」——曾见文本 2.9s 就到、ResultMessage 却晚 35s（桥接静默）。
                    if self._debug:
                        now = time.monotonic()
                        print("[agent] ResultMessage 到达：整轮 %.1fs"
                              % (now - t0), flush=True)
                        if first_txt is not None:
                            print("[agent]   首文本→ResultMessage 间隔 %.1fs（文本早到=CLI 扣结果）"
                                  % (now - first_txt), flush=True)
                    self._notify_result(ctx, msg.result, bool(msg.is_error))
                    return

        try:
            await asyncio.wait_for(_drain(), timeout=timeout)
        except asyncio.TimeoutError:
            # 看门狗：整轮超时仍无 ResultMessage → 中断当前回合并报错，绝不无限挂起
            # （曾实测 resumed 会话被中断残留污染后静默 2-3 分钟无任何事件）。
            try:
                await self._client.interrupt()      # ESC：让 CLI 干净收尾当前回合
            except Exception:
                pass
            raise TimeoutError(
                "agent 超时（%.0fs 无结果，已中断当前回合）。可查控制台 [agent-cli] "
                "最近输出诊断" % timeout)

    def _notify_result(self, ctx, text, is_error):
        if self._on_result is not None:
            try:
                self._on_result(ctx, text, is_error)
            except Exception as e:
                print("[agent] on_result 异常: %s" % e, flush=True)

    def _notify_error(self, exc):
        # dump CLI 最近输出（stderr 环形缓存）：报错/超时时带上，诊断盲区兜底
        if self._stderr_buf:
            tail = "\n".join(list(self._stderr_buf)[-15:])
            print("[agent] CLI 最近输出（诊断超时/报错）：\n%s" % tail, flush=True)
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:
                pass
