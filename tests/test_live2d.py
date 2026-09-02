# -*- coding: utf-8 -*-
"""Live2dEmitter 心态→表情发射器 + wake.on_sleep 复位联动 测试（无 live2d / 无 TTS）。

跑法：PYTHONIOENCODING=utf-8 D:/anaconda/envs/voice-asr/python.exe tests/test_live2d.py
覆盖：
  1   port=None → 不启用（enabled=False，无 worker、不尝试连接）
  2   测活成功 → enabled=True，且构造即同步收到初始「平和」（初始化归位）
  3   emit("开心") → 异步 worker 送达 {"emotion":"开心"}（短连接逐条）
  4   reset() → 发「平和」
  5   端口关闭 → 测活失败：打印告知 + 彻底禁用（enabled=False，emit 短路不抛）
  6   wake.on_sleep：go_sleep("bye") → 复位平和
  7   wake.on_sleep：静默超时（feed_decision 内部 go_sleep("timeout")）→ 复位平和
  8   close() → 同步补发「平和」归位后再停
"""
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dialogue.live2d import Live2dEmitter  # noqa: E402
from dialogue.wake import WakeSession, ACTIVE  # noqa: E402

POW = 10.0 ** (-38.0 / 10.0)     # 与 voice_dialogue SPEECH_POW 一致
QUIET = POW / 10.0               # 低于语音阈值


class FakeServer:
    """假 live2d control server：TCP 逐行收 JSON，记进 received（Live2dEmitter 每
    次短连接，服务端按连接收完整条后关闭）。"""

    def __init__(self):
        self.received = []                       # 收行原始 JSON 字符串
        self._stop = False
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        self._srv.listen(5)
        self._srv.settimeout(0.2)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                buf = b""
                while True:
                    try:
                        data = conn.recv(4096)
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line:
                            self.received.append(line.decode("utf-8"))

    def last(self):
        return json.loads(self.received[-1])["emotion"] if self.received else None

    def close(self):
        self._stop = True
        try:
            self._srv.close()
        except OSError:
            pass


# ---- 1. port=None → 不启用 ----
e = Live2dEmitter()                      # 不给端口
assert e.enabled is False, "不给端口应禁用"
print("测试1 port=None 不启用 OK: enabled=False, 无 worker")

# ---- 2. 测活成功 → enabled + 初始「平和」 ----
srv = FakeServer()
e = Live2dEmitter(port=srv.port)
assert e.enabled, "连上假 server 应启用"
time.sleep(0.1)                          # 等 accept 线程收完初始条
assert srv.last() == "平和", "构造成功应立即同步复位平和, 实际 %r" % srv.last()
print("测试2 测活成功 OK: enabled=True, 初始复位平和=%r" % srv.last())

# ---- 3. emit("开心") → 送达 ----
e.emit("开心")
time.sleep(0.2)                          # worker 异步发送
assert srv.last() == "开心", "emit 应送达, 实际 %r" % srv.last()
print("测试3 emit OK: 异步送达 %r, 共收 %d 条" % (srv.last(), len(srv.received)))

# ---- 4. reset() → 归位平和 ----
e.emit("生气")
time.sleep(0.2)
assert srv.last() == "生气"
e.reset()
time.sleep(0.2)
assert srv.last() == "平和", "reset 应归位平和, 实际 %r" % srv.last()
print("测试4 reset OK: 生气后 reset 归位平和")

# ---- 5. 测活失败 → 彻底禁用（打印 + emit 短路） ----
# 用非法 host 稳定触发 connect 失败（随机端口在 Windows 可能撞上机器已占用监听，
# "必然拒绝"不可靠；非法地址在任何环境下都必败，等价覆盖 __init__ 的 OSError 分支）
e_dead = Live2dEmitter(port=12345, host="256.256.256.256")
assert e_dead.enabled is False, "连不上应彻底禁用"
e_dead.emit("开心")                       # 必须短路不抛
e_dead.reset()
e_dead.close()
print("测试5 测活失败禁用 OK: enabled=False（上方打印为预期提醒）, emit/reset/close 短路")

# ---- 6/7. wake.on_sleep 复位联动：bye 与 timeout 两路 ----
srv2 = FakeServer()
wake = WakeSession(wake_enabled=False, inactive_timeout=60)   # 启动即对话
assert wake.state == ACTIVE
emitter = Live2dEmitter(port=srv2.port)
wake.on_sleep = lambda _reason: emitter.reset()
time.sleep(0.1)
assert srv2.last() == "平和"             # 初始归位

# bye 路（on_sentence 调 go_sleep("bye")）
emitter.emit("难过")
time.sleep(0.2)
assert srv2.last() == "难过"
wake.go_sleep("bye")
time.sleep(0.2)
assert srv2.last() == "平和", "bye 回休眠应复位平和, 实际 %r" % srv2.last()
print("测试6 go_sleep(bye) 复位 OK: 难过→回休眠→平和")

# timeout 路（feed_decision 内部 go_sleep("timeout")，外部不可感知）
wake.on_wake()                           # 回对话
emitter.emit("生气")
time.sleep(0.2)
assert srv2.last() == "生气"
wake.last_activity = 0.0
assert wake.feed_decision(1000.0, QUIET, POW, False) == "none", "应静默超时回休眠"
time.sleep(0.2)
assert srv2.last() == "平和", "超时回休眠应复位平和, 实际 %r" % srv2.last()
print("测试7 静默超时复位 OK: 生气→feed_decision 超时→平和")

# ---- 8. close() → 同步补发平和归位 ----
emitter.close()
time.sleep(0.1)
assert srv2.last() == "平和", "close 应补发平和, 实际 %r" % srv2.last()
print("测试8 close 归位 OK: 退出前补发平和")

srv.close()
srv2.close()
print("\n全部通过")
