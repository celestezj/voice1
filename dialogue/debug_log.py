# -*- coding: utf-8 -*-
"""临时调试埋点（VOICE1_DEBUG_TTS=1 时启用，默认零开销）：把 agent 流式送 TTS 的
提交/打断/代际/气泡推进序列落盘到 sessions/debug_tts_<ts>.log，用于复现
"金价查询音频卡在'给你'、live2d 气泡冻住"这类**真实运行才出现**的冻结——所有 headless
探针都复现不出来，必须抓真实时序。定位后即删（不留死代码）。"""
import os
import threading
import time

_ON = os.environ.get("VOICE1_DEBUG_TTS") == "1"


def enable():
    """命令行开关 `--debug-tts` 用（cmd/PowerShell 设不了 `VAR=1 命令`，直接改运行时标志）。
    与 `VOICE1_DEBUG_TTS=1` 等效，两者任一开启即落盘。"""
    global _ON
    _ON = True
_F = None
_LOCK = threading.Lock()
_T0 = time.monotonic()


def _file():
    global _F
    if _F is None:
        os.makedirs("sessions", exist_ok=True)
        _F = open("sessions/debug_tts_%s.log" % time.strftime("%Y%m%d_%H%M%S"),
                  "w", encoding="utf-8")
    return _F


def dbg(*parts):
    if not _ON:
        return
    try:
        with _LOCK:
            f = _file()
            print("%.3f %s" % (time.monotonic() - _T0,
                               " | ".join(str(p) for p in parts)),
                  file=f, flush=True)
    except Exception:
        pass
