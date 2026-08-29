# -*- coding: utf-8 -*-
"""WakeSession 状态机 + sherpa 关键词文件唯一化 测试（无麦克风 / 无真实 TTS）。

跑法：D:/anaconda/envs/voice-asr/python.exe tests/test_wake.py
覆盖：
  1   默认：有唤醒词 → 启动休眠；无唤醒词 → 启动即对话（旧行为）
  2   on_wake：休眠→对话 + 返回就绪语；重复调用幂等（None）
  3   go_sleep：bye/timeout 返回各自告别语；已休眠幂等
  4   feed_decision 自播门控：就绪语/告别语播放期 → "kws_only"；播完清理
  5   feed_decision 静默超时：非播放期 + 超时 → 回休眠 "none"；语音能量刷新计时
  6   busy（AI 播放）期不判超时；inactive_timeout=0 永不超时
  7   note_partial：对话期刷新计时；休眠期忽略
  8   sherpa 关键词文件按词集哈希唯一（打断词/唤醒词共存不互覆）
  9   静默超时告别语：feed_decision 超时回休眠后 consume_farewell 可取走播放（幂等）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dialogue.wake import (  # noqa: E402
    WakeSession, SLEEP, ACTIVE,
    READY_PHRASE, FAREWELL_BYE, FAREWELL_TIMEOUT,
)

POW = 10.0 ** (-38.0 / 10.0)     # 与 voice_dialogue SPEECH_POW 一致
LOUD = POW * 10.0                # 超过语音阈值
QUIET = POW / 10.0               # 低于语音阈值


class FakeJob:
    """假 TTS Job（镜像 voice0 Job 的 .done）。"""
    def __init__(self):
        self._done = False

    @property
    def done(self):
        return self._done

    def mark_done(self):
        self._done = True


# ---- 1. 默认状态 ----
w = WakeSession(wake_enabled=True, inactive_timeout=60)
assert w.sleeping and w.state == SLEEP, (w.state, w.sleeping)
w2 = WakeSession(wake_enabled=False, inactive_timeout=60)
assert not w2.sleeping and w2.state == ACTIVE, "无唤醒词应启动即对话（旧行为）"
print("测试1 默认状态 OK: 有唤醒词→休眠, 无唤醒词→对话")

# ---- 2. on_wake：休眠→对话 + 就绪语 + 幂等 ----
w = WakeSession(wake_enabled=True, inactive_timeout=60)
assert w.on_wake() == READY_PHRASE and w.state == ACTIVE
t0 = w.last_activity
time.sleep(0.02)
assert w.on_wake() is None, "重复唤醒应幂等（None）"
assert w.state == ACTIVE
print("测试2 on_wake OK: 就绪语=%r, 状态→对话, 重复调用幂等" % READY_PHRASE)

# ---- 3. go_sleep：bye/timeout 告别语 + 幂等 ----
w = WakeSession(wake_enabled=True, inactive_timeout=60)
w.on_wake()
assert w.go_sleep("bye") == FAREWELL_BYE and w.state == SLEEP
w.on_wake()
assert w.go_sleep("timeout") == FAREWELL_TIMEOUT and w.state == SLEEP
assert w.go_sleep("bye") is None, "已休眠再 go_sleep 应幂等（None）"
print("测试3 go_sleep OK: bye=%r\ntimeout=%r" % (FAREWELL_BYE, FAREWELL_TIMEOUT))

# ---- 4. feed_decision 自播门控 ----
w = WakeSession(wake_enabled=False, inactive_timeout=60)   # 直接对话
job = FakeJob()
w.set_self_talk(job)
now = 100.0
assert w.feed_decision(now, LOUD, POW, False) == "kws_only", "自播播放期应只喂KWS"
assert w.self_talk is job, "自播未播完不清引用"
job.mark_done()                                          # 自播播完
assert w.feed_decision(now + 1, QUIET, POW, False) == "full", "自播播完应清理并正常喂"
assert w.self_talk is None, "自播播完应清理引用"
print("测试4 自播门控 OK: 播放期→kws_only, 播完→full+清理")

# ---- 5. 静默超时 ----
w = WakeSession(wake_enabled=False, inactive_timeout=60)
now = 1000.0
# 安静块不刷新计时、正常喂
assert w.feed_decision(now, QUIET, POW, False) == "full"
# 说话块刷新计时（即使安静了 100s 也不超时）
assert w.feed_decision(now + 100.0, LOUD, POW, False) == "full"
# 之后安静 >60s → 超时回休眠
assert w.feed_decision(now + 161.0, QUIET, POW, False) == "none"
assert w.state == SLEEP, "超时应回休眠"
assert w.feed_decision(now + 162.0, QUIET, POW, False) == "none", "休眠期防御"
print("测试5 静默超时 OK: 说话刷新→不超时, 静默60s→回休眠")

# ---- 6. busy 期不判超时 / inactive_timeout=0 ----
w = WakeSession(wake_enabled=False, inactive_timeout=60)
now = 2000.0
w.last_activity = now - 1000.0                          # 早已"超时"
assert w.feed_decision(now, QUIET, POW, True) == "full", "AI播放期不应超时"
assert w.state == ACTIVE
w2 = WakeSession(wake_enabled=False, inactive_timeout=0)
w2.last_activity = now - 100000.0
assert w2.feed_decision(now, QUIET, POW, False) == "full", "inactive_timeout=0 永不超时"
print("测试6 busy/0超时 OK: AI播放期不判超时, 0=关闭自动休眠")

# ---- 7. note_partial：对话期刷新 / 休眠期忽略 ----
w = WakeSession(wake_enabled=True, inactive_timeout=60)
t0 = w.last_activity
w.note_partial()
assert w.last_activity == t0, "休眠期 note_partial 应忽略"
w.on_wake()
time.sleep(0.02)
w.note_partial()
assert w.last_activity > t0, "对话期 note_partial 应刷新计时"
print("测试7 note_partial OK: 对话期刷新, 休眠期忽略")

# ---- 8. sherpa 关键词文件按词集哈希唯一 ----
from asr.kws.sherpa import SherpaKwsDetector  # noqa: E402
d1 = SherpaKwsDetector(words=["小爱小爱"])
d2 = SherpaKwsDetector(words=["停下"])
d1._ensure_keywords_file()
f1 = os.path.basename(d1._keywords_file)
d2._ensure_keywords_file()
f2 = os.path.basename(d2._keywords_file)
assert f1 != f2, "不同词集应各占独立 keywords 文件: %r vs %r" % (f1, f2)
assert "keywords_" in f1 and f1.endswith(".txt"), f1
print("测试8 关键词文件唯一 OK: %r ≠ %r（打断/唤醒共存不互覆）" % (f1, f2))

# ---- 9. 静默超时告别语：feed_decision 超时后 consume_farewell 可取走 ----
w = WakeSession(wake_enabled=False, inactive_timeout=60)
w.last_activity = 0.0
assert w.feed_decision(100.0, QUIET, POW, False) == "none", "超时→none"
assert w.state == SLEEP
assert w.consume_farewell() == FAREWELL_TIMEOUT, "超时告别语应可取走播放"
assert w.consume_farewell() is None, "取走即清空（幂等）"
w2 = WakeSession(wake_enabled=False, inactive_timeout=60)
assert w2.consume_farewell() is None, "从未休眠→无待播告别语→None"
print("测试9 超时告别语 OK: 超时回休眠后告别语可取走播放，无待播幂等 None")

print("\n全部通过 ✔")
