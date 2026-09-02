# -*- coding: utf-8 -*-
"""休眠/对话两态会话状态机（唤醒功能）。纯逻辑、不碰硬件。

voice_dialogue.py 用注入的方式驱动它：mic 回调每块喂 `feed_decision`，
唤醒词命中调 `on_wake`、退出词调 `go_sleep("bye")`，返回的
就绪语/告别语由调用方**直连 `tts.submit` 播放**（不经 DialogueController，
所以它们不进对话历史/LLM）；播放 Job 记进 `self_talk`，mic 回调据此把
自播语音当"自播回声"门控——只喂打断词 KWS，防"在的，我在听"被识别成
用户的话又提交一轮。静默超时由 `feed_decision` **内部**调 `go_sleep("timeout")`
触发（返回值传不回调用方），告别语暂存进 `_farewell`，调用方收到
`"none"` 后用 `consume_farewell()` 取走播放——否则超时回休眠全程无声无提示。

设计决策（用户拍板）：
- 启动默认休眠（有唤醒词时），无唤醒词 → 启动即对话（旧行为）。
- 对话历史跨休眠保留：状态机不清 controller 历史，只有程序重启才清。
- 静默超时判定：对话期 `feed_decision` 里，非 AI 播放期（not busy）且
  `now - last_activity > inactive_timeout` → 回休眠；"用户语音"信号 =
  ASR 出字（`note_partial`，任意距离都算）或 mic 块能量超阈值。
"""

import time

# 状态常量
SLEEP, ACTIVE = 0, 1

# 就绪语 / 告别语（调用方直连 TTS 播放，不入 controller 历史/LLM）
READY_PHRASE = "在的，我在听。"
FAREWELL_BYE = "好的，我先退下啦，要和我说话，就唤醒我哦~"
FAREWELL_TIMEOUT = "一直不说话，我先退下啦，要和我说话，就唤醒我哦~"


class WakeSession:
    """休眠（只听唤醒词）/ 对话（全链路）两态。线程安全约定：mic 回调单线程调用
    `feed_decision`/`on_wake`；`note_partial` 来自 ASR worker 线程，只写
    `last_activity`（float 原子赋值，无锁）。`self_talk` 由调用方在 TTS 提交
    线程写、mic 回调读——同一 Job 对象，`.done` 是 threading.Event 查询。
    """

    def __init__(self, wake_enabled=True, inactive_timeout=60, on_sleep=None):
        self.wake_enabled = bool(wake_enabled)
        self.state = SLEEP if wake_enabled else ACTIVE   # 无唤醒词 → 启动即对话
        self.last_activity = time.monotonic()            # 最近用户语音时刻（静默超时用）
        self.self_talk = None                            # 就绪语/告别语 Job（自播回声门控）
        self.inactive_timeout = inactive_timeout
        self._farewell = None                            # 待播告别语（静默超时路径暂存）
        self.on_sleep = on_sleep                         # 可选：进入休眠回调(reason)，调用方注入

    @property
    def sleeping(self):
        return self.state == SLEEP

    def on_wake(self):
        """休眠 → 对话；返回要播的就绪语（调用方 tts.submit 后 set_self_talk）。
        已对话 → None（幂等）。"""
        if self.state == ACTIVE:
            return None
        self.state = ACTIVE
        self.last_activity = time.monotonic()
        return READY_PHRASE

    def go_sleep(self, reason="timeout"):
        """对话 → 休眠；返回要播的告别语（调用方 tts.submit 后 set_self_talk）。
        已休眠 → None（幂等）。reason: "bye"（用户说退出词）| "timeout"（静默超时）。
        告别语同时暂存进 `_farewell`：静默超时由 feed_decision 内部调用本方法，
        返回值传不回调用方，调用方收到 "none" 后用 `consume_farewell()` 取走播放。
        `on_sleep` 回调（若注入）在进入休眠后同步触发——bye/timeout 两条回休眠
        路径的唯一汇聚点，供上层做表情归位等收尾（回调抛异常被吞，不致命）。"""
        if self.state == SLEEP:
            return None
        self.state = SLEEP
        if self.on_sleep is not None:
            try:
                self.on_sleep(reason)
            except Exception:
                pass                # 回调失败不影响状态机（live2d 复位等属上层展示）
        self._farewell = FAREWELL_BYE if reason == "bye" else FAREWELL_TIMEOUT
        return self._farewell

    def consume_farewell(self):
        """取走暂存的待播告别语并清空（幂等：无待播 → None）。"""
        f, self._farewell = self._farewell, None
        return f

    def set_self_talk(self, job):
        self.self_talk = job

    def note_partial(self):
        """ASR 出了字 → 用户在说话（任意距离都算），刷新静默计时。休眠期无 ASR 不触发，
        防御性忽略。"""
        if self.state == ACTIVE:
            self.last_activity = time.monotonic()

    def feed_decision(self, now, rms2, speech_pow, busy):
        """对话期每 mic 块调一次；返回喂 ASR 决策。

        "none"     → 本块不喂 ASR（已回休眠：静默超时 / 休眠期防御）
        "kws_only" → 只喂打断词 KWS（自播就绪语/告别语回声门控，自播语音不进识别）
        "full"     → 正常喂（调用方走回声门控 + 滚动 grace 那套）

        副作用：自播播完清理、语音活动刷新、静默超时 → go_sleep("timeout")。
        静默超时只在非 AI 播放期（not busy）判定：AI 在播时对话仍算活跃。
        """
        if self.state == SLEEP:
            return "none"
        if self.self_talk is not None and not self.self_talk.done:
            return "kws_only"                          # 自播回声门控
        if self.self_talk is not None:
            self.self_talk = None                      # 自播播完 → 清理
        if rms2 > speech_pow:
            self.last_activity = now                   # 房间有语音能量（说话/回声）→ 刷新
        if not busy and self.inactive_timeout > 0 \
                and (now - self.last_activity) > self.inactive_timeout:
            self.go_sleep("timeout")                   # 静默超时 → 回休眠
            return "none"
        return "full"
