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

线程模型：本类持有**常驻 asyncio 事件循环线程**。`submit()/abort()/close()` 线程安全
（`asyncio.run_coroutine_threadsafe` 桥接；内部 worker 串行化，保证一次只有一个 query
在飞、abort 先收尾再续）。回调（on_result/on_partial）在**循环线程**执行，须快速返回、
不能阻塞循环——controller 侧只做持锁快操作。
"""
import asyncio
import os
import threading
import uuid

from claude_agent_sdk import (ClaudeSDKClient, ClaudeAgentOptions,
                              AssistantMessage, ResultMessage, StreamEvent)

# 默认 agent 工作目录（assistant 人格目录，repo 根的 assistant/）——主程序可 --agent-dir 覆盖
_DEFAULT_AGENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assistant")
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
                 on_result=None, on_partial=None, on_error=None, debug=False):
        self._cwd = cwd or _DEFAULT_AGENT_DIR
        self._persona_file = persona_file or os.path.join(self._cwd, _DEFAULT_PERSONA_FILE)
        self._session_id_file = session_id_file or _DEFAULT_SESSION_FILE
        self._resume = bool(resume)
        self._permission_mode = permission_mode
        self._allowed_tools = list(allowed_tools or [])
        self._disallowed_tools = list(disallowed_tools or [])
        self._model = model
        self._connect_timeout = connect_timeout
        self._include_partial = bool(include_partial_messages)
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

    async def _connect(self):
        persona = self._load_persona()
        opts = ClaudeAgentOptions(
            cwd=self._cwd,
            system_prompt=persona,                     # cwd 的 CLAUDE.md 不自动加载，须显式传
            session_id=None if self._resume else self._session_id,
            resume=self._session_id if self._resume else None,
            permission_mode=self._permission_mode,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            model=self._model,
            include_partial_messages=self._include_partial,  # True：控制台流式出字；TTS 仍取最终结论
            stderr=(lambda line: None),                # 静默框架噪音（tqdm/错误不回刷屏）
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
        """优雅关闭：断开 claude（进程正常结束）+ 停循环线程。"""
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._pending is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._pending.put(("close",)),
                                                 self._loop).result(timeout=5)
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
                        try:
                            self._on_partial(ctx, d["text"])
                        except Exception:
                            pass
            elif isinstance(msg, ResultMessage):
                self._notify_result(ctx, msg.result, bool(msg.is_error))
                break

    def _notify_result(self, ctx, text, is_error):
        if self._on_result is not None:
            try:
                self._on_result(ctx, text, is_error)
            except Exception as e:
                print("[agent] on_result 异常: %s" % e, flush=True)

    def _notify_error(self, exc):
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:
                pass
