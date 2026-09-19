# -*- coding: utf-8 -*-
"""voice0 RealtimeTTS 的 live2d 跟播代理：文本→说话框**逐句链式** + 一轮播完自动复位。

把 voice1 送进 TTS 的每一段文本逐句显示到 live2d 桌宠说话框（say 通道）。核心难点：
voice0 `mode="queue"` **提交即入队、串行播放**——LLM 一口气吐 3 句时 3 个 Job 瞬间排进
队、音频却还在播第 1 句。若在 submit 那一刻就把文本发出去，气泡会被最后一句立刻刷新
（音频没跟上）。故本代理做**逐句链式跟播**：队首（最旧未播）的句子才发文本（≈ 它正要
开播），播完（`job.done`）才推进下一句——queue 模式下 prev-done ≈ 下句开播，气泡永远
显示"正在播的那句"、随音频逐句推进。被打断（`hard_stop` 使 job.canceled）→ 丢弃尚未播
的作废句文本。

voice0 无播放回调，Job 只暴露 done / wait / canceled 三信号；逐句跟播靠"前句播完"作
下句开播的代理。本模块纯逻辑、不碰硬件：tts / say_cb / mood_cb / idle_cb / active_check
全依赖注入，headless 可测（tests/test_say_tts.py 用假 TTS/Job）。

**心态发射也挂在本链上（2026-09-19）**：`submit(text, mood=...)` 的 mood 是 controller
`_submit_tts` 提交前从句子提取的句首心态（标记在送 TTS 前已被剥掉，须随 submit 带给本
代理）。队首句子"正在播或下一个要播"那一刻发文本的同时发 mood_cb——表情跟说话框同一
时刻切换、与听感同步。旧设计在文本到达时发心态（`_parse_mood_locked` on_mood）：LLM
~1s 吐完全文、音频要播几十秒，全部心态挤在开头 1s 发完 → 表情全程卡最后一个标签。
- say_cb(text)：发说话框（live2d.say；禁用时内部短路）。
- mood_cb(mood)：队首句子开播瞬间发心态（无标签的句子传 mood=None → 继承当前，不切）。
- idle_cb()：一轮播放真正播完 → 收框 + 表情复位（受 `--live2d-idle-reset` 控制，由
  调用方决定是否注入/注入后是否复位）。
- active_check()：返回 controller.turn_active = LLM 流在途或 TTS 队列非空。队列排空
  ≠ 一轮播完：LLM 流中句与句之间队列也会短暂排空，若在那时复位，气泡会在长回复中途被
  收掉。排空后再等结算窗、并确认 active_check 为 False 才视为一轮播完 → idle_cb。
  active_check 在 ctrl 构造完成后再注入（本代理先于 ctrl 创建）。
- 自播（就绪/告别/问候语）不入 controller，同走本链；live2d 未启用时本代理不构造
  （tts 保持原样），行为与未加 live2d 完全一致。
"""

import threading
import time

from .debug_log import _ON as _DBG_ON, dbg   # 临时：VOICE1_DEBUG_TTS=1 才落盘，定位后删


class SayTTS:
    """voice0 RealtimeTTS 的 live2d 跟播代理。包一层后 `submit` 仍是原语义（返回真 Job），
    只是文本/心态改由链式调度发送——controller / 自播调用方零改动。

    `mood_supported = True`：controller `_submit_tts` 据此判断能否随 submit 带心态参数
    （live2d 未启用时 tts 是裸 voice0 RealtimeTTS，无此属性 → controller 不传 mood）。"""

    mood_supported = True               # controller 鸭子判断：能否接收 submit(text, mood=)

    def __init__(self, tts, say_cb, idle_cb=None, settle=0.3, mood_cb=None):
        self._tts = tts
        self._say_cb = say_cb            # 发说话框文本（逐句，非阻塞入队）
        self._mood_cb = mood_cb          # 队首句子开播瞬间发心态（随文本同一时刻）
        self._idle_cb = idle_cb          # 一轮播完回调（None=不自动复位）
        self._settle = settle            # 排空后的结算窗秒数（判"真实播完"；测试可调小）
        self._active_check = None        # set_active_check 注入：返回 controller.turn_active
        self._lock = threading.Lock()
        self._pending = []               # [(job, text, mood)] 按提交序；逐句播完才轮到下一句
        self._last_mood = None           # 最近一次发射的心态（去重：无标签/同心态不重复发；
                                         # 链排空复位=新一轮首句同心态也必发）
        self._worker = None              # 链式调度线程（惰性启停）

    def set_active_check(self, fn):
        """ctrl 构造后注入：fn() 返回一轮对话是否仍在进行（LLM 流在途 / TTS 队列非空）。"""
        self._active_check = fn

    def set_mood_cb(self, fn):
        """注入心态发射回调（on_mood → live2d.emit）。主程序里 on_mood 定义晚于本代理，
        故单独注入，不用构造参数耦合。"""
        self._mood_cb = fn

    def submit(self, text, mood=None):
        """提交 voice0，文本/心态登记进跟播链（逐句播完才发，杜绝多句抢发）。返回原 Job。

        mood = controller 提交前提取的句首心态（无标记= None → 继承当前心态不切表情）。"""
        job = self._tts.submit(text)
        with self._lock:
            self._pending.append((job, text, mood))
            dbg("SAY.submit pend=%d mood=%r %r" % (len(self._pending), mood, text[:20]))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drive,
                                                name="say-tts-watch", daemon=True)
                self._worker.start()
        return job

    def _drive(self):
        """跟播链：队首（最旧未播）的句子才发文本；播完才推进到下一句。

        纯文本不插桩：LLM 流里逐句 submit 是自然顺序，链长 ≤ 本轮句数，每句播完清一个
        队首，气泡逐句跟随。**心态与文本同刻发射**：队首句子"正要开播"那一刻发文本的
        同时也发它的心态（controller 提交前提取、随 submit 带入）——表情跟听感同步切换；
        无标签（mood=None）/同心态重复 → 不切（继承当前）。被打断（job.canceled）→ 滤掉
        作废句文本**及心态**（未播句的心态不发出）；新代际的 submit 会重启 worker。
        队列排空 → 结算窗 + active_check 判"真实播完" → idle_cb。退出只在锁内、队空时
        发生，与 submit 追加无竞态；**链排空即复位 _last_mood**（本轮播放彻底结束/作废，
        新一轮首句同心态也必发——live2d 可能已被 idle/interrupt 复位成平和）。"""
        while True:
            with self._lock:
                if not self._pending:
                    self._last_mood = None   # 链排空：心态状态复位（新一轮首句必发）
                    dbg("SAY.drive exit")
                    return                # 无事可播 → 退出（下条 submit 重启新 worker）
                job, text, mood = self._pending[0]
                if job.done:
                    self._pending.pop(0)  # 队首已播完/被取消 → 清掉看下一个
                    dbg("SAY.skip done=%s canceled=%s %r" % (job.done, job.canceled, text[:20]))
                    continue
            # 队首未播 = 正在播或下一个要播 → 此刻发它文本（气泡跟上开播）+ 心态同步发射
            dbg("SAY.say mood=%r %r" % (mood, text[:20]))
            self._say_cb(text)
            if mood is not None and mood != self._last_mood:
                self._last_mood = mood    # 心态变化才切表情（无标签继承 / 同心态去重）
                if self._mood_cb:
                    self._mood_cb(mood)
            if _DBG_ON:
                # 带超时打点的 wait（仅调试）：卡死时能看到反复的 SAY.waiting 行
                _evt = threading.Event()
                threading.Thread(target=lambda: (job.wait(), _evt.set()), daemon=True).start()
                _t0 = time.monotonic()
                while not _evt.wait(2.0):
                    dbg("SAY.waiting >%ds %r" % (int(time.monotonic() - _t0), text[:20]))
            else:
                job.wait()                    # 阻塞到本句播完/被打断（voice0 永不悬挂）
            dbg("SAY.wait done=%s canceled=%s %r" % (job.done, job.canceled, text[:20]))
            with self._lock:
                if self._pending and self._pending[0][0] is job:
                    self._pending.pop(0)
            if job.canceled:
                # 被打断（hard_stop）→ 作废代际的未播句文本+心态全丢；新 submit 会重启 worker
                with self._lock:
                    self._pending = [(j, t, m) for (j, t, m) in self._pending if not j.canceled]
                    dbg("SAY.filter remain=%d" % len(self._pending))
                continue
            # 本句正常播完 → 若链已空（队列排空），做一轮"真实播完"判断
            with self._lock:
                drained = not self._pending
            if drained:
                time.sleep(self._settle)   # 结算窗：controller busy 回落是异步毫秒级
                with self._lock:
                    drained = not self._pending   # 结算窗内来了新句？
                if drained and (self._active_check is None or not self._active_check()):
                    if self._idle_cb is not None:
                        self._idle_cb()   # 一轮播放真正结束 → 复位（收框 + 表情）
            # 回到循环头：还有下一句则发它文本；链空则在锁内退出（与 submit 无竞态）

    def interrupt(self):
        return self._tts.interrupt()

    def __getattr__(self, name):
        return getattr(self._tts, name)
