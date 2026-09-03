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
下句开播的代理。本模块纯逻辑、不碰硬件：tts / say_cb / idle_cb / active_check 全依赖
注入，headless 可测（tests/test_say_tts.py 用假 TTS/Job）。

- say_cb(text)：发说话框（live2d.say；禁用时内部短路）。
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


class SayTTS:
    """voice0 RealtimeTTS 的 live2d 跟播代理。包一层后 `submit` 仍是原语义（返回真 Job），
    只是文本改由链式调度发送——controller / 自播调用方零改动。"""

    def __init__(self, tts, say_cb, idle_cb=None, settle=0.3):
        self._tts = tts
        self._say_cb = say_cb            # 发说话框文本（逐句，非阻塞入队）
        self._idle_cb = idle_cb          # 一轮播完回调（None=不自动复位）
        self._settle = settle            # 排空后的结算窗秒数（判"真实播完"；测试可调小）
        self._active_check = None        # set_active_check 注入：返回 controller.turn_active
        self._lock = threading.Lock()
        self._pending = []               # [(job, text)] 按提交序；逐句播完才轮到下一句
        self._worker = None              # 链式调度线程（惰性启停）

    def set_active_check(self, fn):
        """ctrl 构造后注入：fn() 返回一轮对话是否仍在进行（LLM 流在途 / TTS 队列非空）。"""
        self._active_check = fn

    def submit(self, text):
        """提交 voice0，文本登记进跟播链（逐句播完才发，杜绝多句抢发）。返回原 Job。"""
        job = self._tts.submit(text)
        with self._lock:
            self._pending.append((job, text))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drive,
                                                name="say-tts-watch", daemon=True)
                self._worker.start()
        return job

    def _drive(self):
        """跟播链：队首（最旧未播）的句子才发文本；播完才推进到下一句。

        纯文本不插桩：LLM 流里逐句 submit 是自然顺序，链长 ≤ 本轮句数，每句播完清一个
        队首，气泡逐句跟随。被打断（job.canceled）→ 滤掉作废句文本；新代际的 submit
        会重启 worker。队列排空 → 结算窗 + active_check 判"真实播完" → idle_cb。
        退出只在锁内、队空时发生，与 submit 追加无竞态。"""
        while True:
            with self._lock:
                if not self._pending:
                    return                # 无事可播 → 退出（下条 submit 重启新 worker）
                job, text = self._pending[0]
                if job.done:
                    self._pending.pop(0)  # 队首已播完/被取消 → 清掉看下一个
                    continue
            # 队首未播 = 正在播或下一个要播 → 此刻发它文本（气泡跟上开播）
            self._say_cb(text)
            job.wait()                    # 阻塞到本句播完/被打断（voice0 永不悬挂）
            with self._lock:
                if self._pending and self._pending[0][0] is job:
                    self._pending.pop(0)
            if job.canceled:
                # 被打断（hard_stop）→ 作废代际的未播句文本全丢；新 submit 会重启 worker
                with self._lock:
                    self._pending = [(j, t) for (j, t) in self._pending if not j.canceled]
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
