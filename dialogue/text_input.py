# -*- coding: utf-8 -*-
"""文本输入源（调试用）：与麦克风语音并存的键盘/脚本输入口。

主程序 `--text-input-port` 起常驻 TCP 监听（默认关=原程序零变化），
`examples/text_input.py` 连上后 `while input()` 逐行发送文本行（UTF-8、`\n` 分隔）。
每行经 `route_text_line` 路由：

- 休眠态 → `wake.on_wake()` **静默**切 ACTIVE（返回的就绪语丢弃，不播——文本是刻意输入，
  直接对话）；
- 打断词（**整行完全等于**）→ `ctrl.hard_stop()`：立即停 LLM+TTS，该行不进历史/不送 LLM；
  无输出在途 = no-op；
- 其余（**含唤醒词/退出词——文本模式无效**，不触发唤醒/退出状态机）→ 构造 SentenceResult
  走 `ctrl.feed_asr_sentence()`，与麦克风定稿句完全同构：barge-in 代际 / post-commit barge /
  LLM/agent 启动 / 历史 / 控制台时间戳 / live2d 输出全部自动继承，无需输出侧任何改动。

线程模型：TextInputServer 每连接一线程，行到达调 `on_line`（主程序 = feed_text → route）。
controller 内部持锁 + `_gen` 代际串行化，与 ASR worker 并发提交安全。块级拦截（回声门控 /
自播门控 / 休眠 KWS 分派）都在 mic 采集回调里，文本走独立 TCP 线程**天然绕过**。

本模块纯逻辑 + socketserver，不碰硬件，headless 可测。
"""

import socketserver
import threading

# route_text_line 返回值（动作）
EMPTY = "empty"          # 空白行，忽略
INTERRUPT = "interrupt"  # 打断词整行 → 已 hard_stop（不进对话）
SENTENCE = "sentence"    # 普通句 → 已 feed_asr_sentence


def route_text_line(ctrl, wake, interrupt_words, line, make_result):
    """一行文本 → 动作（返回 EMPTY / INTERRUPT / SENTENCE）。纯逻辑，headless 可测。

    - `ctrl`：DialogueController（feed_asr_sentence / hard_stop）。
    - `wake`：WakeSession（sleeping / on_wake / note_partial）。
    - `interrupt_words`：打断词列表（可为 None）。**整行完全等于**才触发——文本是刻意
      整行输入，精确匹配避免"你停下吧"这类自然表达误打断（语音 KWS 是子串命中，两套语义）。
    - `make_result(line)`：调用方注入的 SentenceResult 工厂（文本无音频，audio_* 取当下
      会话轴时刻，与 ASR 句同轴）。
    """
    line = (line or "").strip()
    if not line:
        return EMPTY
    # 休眠态文本自动唤醒：静默切 ACTIVE（on_wake 返回的就绪语丢弃，不播）。
    if wake.sleeping:
        wake.on_wake()
    # 打断词：整行完全等于 → 立即停当前输出（不进历史/LLM；无输出在途 = no-op）。
    if interrupt_words and line in interrupt_words:
        ctrl.hard_stop()
        return INTERRUPT
    # 其余一律当普通句子送对话：唤醒词/退出词**文本模式无效**（不拦截、不进状态机）。
    wake.note_partial()          # 刷新静默计时（文本输入也算活跃，防无语音被回休眠）
    # barge_audio=True：文本是刻意完整问题，**正在输出（LLM 在途或 TTS 在播）即打断**——
    # 语音定稿句默认 False（LLM 已答完、仅音频在播时不打断、新回复排队），文本语义更强。
    ctrl.feed_asr_sentence(make_result(line), barge_audio=True)
    return SENTENCE


class _TextInputHandler(socketserver.StreamRequestHandler):
    """每个连接一个线程：读 `\n` 分隔行，非空行调 server 注入的 on_line。"""

    def handle(self):
        for raw in self.rfile:                     # StreamRequestHandler 迭代 = readline
            line = raw.decode("utf-8", errors="replace").strip()
            if line and self.server.on_line is not None:
                try:
                    self.server.on_line(line)
                except Exception:
                    pass        # 回调异常不影响继续收（不致命）


class TextInputServer:
    """常驻 127.0.0.1:PORT 文本行监听（调试输入源，与麦克风并存）。

    `start()` 起 daemon 线程 serve_forever；`close()` 关停。端口被占用 → 构造抛 OSError，
    调用方捕获后打印提示并禁用（与 live2d 测活失败禁用一致的"不致命"）。多客户端可同时
    连/断（ThreadingTCPServer），每行文本调 `on_line(line)`。
    """

    def __init__(self, host, port, on_line):
        self._srv = socketserver.ThreadingTCPServer((host, port), _TextInputHandler)
        self._srv.on_line = on_line          # handler 经 self.server 取
        self._srv.daemon_threads = True      # 客户端线程不阻塞程序退出
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        name="text-input-srv", daemon=True)
        self._thread.start()

    def close(self):
        try:
            self._srv.shutdown()
        except Exception:
            pass
        try:
            self._srv.server_close()
        except Exception:
            pass
