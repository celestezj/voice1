# -*- coding: utf-8 -*-
"""SayTTS 逐句链式跟播 + 一轮播完自动复位 测试（假 TTS/Job，无 voice0 / 无 live2d）。

跑法：PYTHONIOENCODING=utf-8 D:/anaconda/envs/voice-asr/python.exe tests/test_say_tts.py
覆盖（重点回归"多句抢发"缺陷：queue 模式下音频串行，气泡文本必须逐句跟播）：
  1  单句：提交即发文本；播完 + 无在途 → idle_cb（自动复位）恰好一次
  2  三句连发（原 bug）：提交瞬间只发第 1 句，第 2 句等第 1 句播完才发，
     第 3 句等第 2 句播完才发——气泡显示"正在播的那句"
  3  打断（hard_stop 取消全部排队）：未播句文本全丢弃、不发，也不触发 idle_cb
  4  跨句停顿（active_check 仍 True）：队列排空不复位；流结束后续句播完才复位一次
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dialogue.say_tts import SayTTS  # noqa: E402


class FakeJob:
    """模仿 voice0 Job：只有 done / wait / canceled 三信号，播完由测试手动标。"""

    def __init__(self):
        self.canceled = False
        self._ev = threading.Event()

    @property
    def done(self):
        return self._ev.is_set()

    def wait(self):
        self._ev.wait()
        return []

    def finish(self, canceled=False):
        self.canceled = canceled
        self._ev.set()


class FakeTTS:
    """模仿 voice0 RealtimeTTS.submit：按调用序产出 FakeJob。"""

    def __init__(self):
        self.jobs = []

    def submit(self, text):
        j = FakeJob()
        self.jobs.append(j)
        return j

    def interrupt(self):
        pass


def _wait(cond, timeout=2.0, what=""):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("超时等待: %s" % what)


def _worker_done(p):
    """链 worker 已退出（处理完当前状态）。"""
    return p._worker is None or not p._worker.is_alive()


def make_proxy(active=None):
    tts = FakeTTS()
    said, idles = [], []
    p = SayTTS(tts, said.append, idle_cb=lambda: idles.append(1), settle=0.05)
    if active is not None:
        p.set_active_check(lambda: active[0])
    return tts, p, said, idles


# ---- 1. 单句：提交即发 → 播完无在途 → idle 恰好一次 ----
tts, p, said, idles = make_proxy()
p.submit("句一")
_wait(lambda: len(said) >= 1, what="单句提交后应立即发文本")
assert said == ["句一"], "单句应只发一次, 实际 %r" % said
assert not idles, "还没播完不应复位"
tts.jobs[0].finish()
_wait(lambda: len(idles) == 1, what="单句播完应触发一次 idle")
time.sleep(0.15)
assert idles == [1], "不应重复复位, 实际 %r" % idles
print("测试1 单句 OK: 提交即发=%r, 播完 idle 一次" % (said,))

# ---- 2. 三句连发（原 bug：气泡必须逐句跟播，不抢发）----
tts, p, said, idles = make_proxy()
p.submit("句一")
p.submit("句二")
p.submit("句三")
_wait(lambda: len(said) >= 1, what="首句应尽快发")
time.sleep(0.2)                          # 若抢发，句二/三早就冒出
assert said == ["句一"], "三句连发只能先发第 1 句, 实际 %r" % said
tts.jobs[0].finish()                     # 句一播完 → 才轮到句二
_wait(lambda: len(said) >= 2, what="句一播完应发句二")
time.sleep(0.15)
assert said == ["句一", "句二"], "句三不应抢先, 实际 %r" % said
tts.jobs[1].finish()                     # 句二播完 → 才轮到句三
_wait(lambda: len(said) >= 3, what="句二播完应发句三")
time.sleep(0.15)
assert said == ["句一", "句二", "句三"], "三句应逐句跟播, 实际 %r" % said
print("测试2 三句链式 OK: %r（逐句跟播，不抢发）" % said)

# ---- 3. 打断：未播句文本全丢，不触发 idle ----
tts, p, said, idles = make_proxy()
p.submit("句一")
p.submit("句二")
p.submit("句三")
_wait(lambda: len(said) >= 1, what="首句应先发")
assert said == ["句一"]
for j in tts.jobs:                       # hard_stop：voice0 取消所有排队 Job
    j.finish(canceled=True)
_wait(lambda: _worker_done(p), what="打断后链应退出")
assert said == ["句一"], "作废句文本不应发出, 实际 %r" % said
assert idles == [], "打断不触发一轮播完复位（那是 on_interrupt 的活）, 实际 %r" % idles
print("测试3 打断 OK: 已发=%r（句二/三作废丢弃, idle 不触发）" % said)

# ---- 4. 跨句停顿：active_check True 时不复位；流结束续句播完才复位一次 ----
active = [True]
tts, p, said, idles = make_proxy(active=active)
p.submit("句一")
_wait(lambda: len(said) >= 1, what="首句应先发")
tts.jobs[0].finish()                     # 句一播完，但 LLM 流还在（跨句停顿）
time.sleep(0.3)                          # 超过 settle；在途 → 不应复位
assert idles == [], "流仍在途不应复位, 实际 %r" % idles
_wait(lambda: _worker_done(p), what="空链应退出等下一条 submit")
p.submit("句二")                         # LLM 续句来了 → 新链
_wait(lambda: len(said) >= 2, what="续句应发")
active[0] = False                        # 流真正结束
tts.jobs[1].finish()
_wait(lambda: len(idles) == 1, what="流结束后续句播完应复位一次")
assert said == ["句一", "句二"], "两段应都发, 实际 %r" % said
time.sleep(0.15)
assert idles == [1], "不应重复复位, 实际 %r" % idles
print("测试4 跨句停顿 OK: 在途不复位, 流尽续句播完复位一次, said=%r" % said)

print("\n全部通过")
