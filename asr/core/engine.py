# -*- coding: utf-8 -*-
"""RealtimeASR：后端无关的实时语音识别引擎（单例 · 常驻线程 · 完整生命周期）。

设计沿用 voice0 的 RealtimeTTS 骨架，适配 ASR 的"连续音频流"语义：
- **单例**：模型只加载一次；backend/device/vad_tail/interrupt_words 变更 → 销毁重建。
- **常驻处理线程**：`_worker` 消费 `_audio_q` 音频块 → VAD 断句 → 持 `_recog_lock` 识别 → 回调。
- **有界背压**：`_audio_q` maxsize=8 —— 识别慢于喂入时阻塞喂入方，防延迟膨胀。
- **会话代际 `_gen`**：`interrupt()` 计数 +1，旧会话音频块作废；VAD/后端状态同步重置。
- **打断词旁路（T12）**：`interrupt_words` 非空时启用轻量 KWS。主路径是 **ingest
  流式 `feed()`**——每个音频块先过 KWS，命中打断词（如"停下"）→ `interrupt()`
  即时作废排队任务、触发块丢弃不识别（不等 VAD 断句，避免"打断词排队尾"悖论）；
  正在识别的任务无法中止（整句前向原子），完成后判 `stale=True` 不进普通回调。
  另保留 VAD 断句后整句 `detect()` 兜底（流式 miss 的第二道防线）。
- **`wait()` 语义**：实时场景无 Job；`ingest_file()` 阻塞同步返回逐句结果，bench 兼容。
- **插桩**：`profile`（逐句时序）/ `debug` 守卫，热路径零埋点。
"""
import queue
import threading
import time

import numpy as np

from .audio import EnergyVAD, read_wav, resample_to
from .backend import get_backend
from .jobs import SentenceResult
from ..kws.interrupt import get_interrupt_detector


class RealtimeASR:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, backend="paraformer", device="auto", sample_rate=16000,
                vad_silence_tail_ms=250, profile=False, debug=False,
                interrupt_words=None):
        # 单例：backend/device/vad_tail/interrupt_words 任一变更 → 销毁重建；否则同一实例
        inst = cls._instance
        if inst is not None and (inst._backend_name != backend or inst._device != device
                                 or inst._vad_tail_ms != vad_silence_tail_ms
                                 or inst._interrupt_words != list(interrupt_words or [])):
            inst.close()
            inst = None
        if inst is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend="paraformer", device="auto", sample_rate=16000,
                 vad_silence_tail_ms=250, profile=False, debug=False,
                 interrupt_words=None):
        if getattr(self, "_inited", False):
            return
        self._backend_name = backend or "paraformer"
        self._device = device or "auto"
        self._sr = int(sample_rate)
        self._vad_tail_ms = vad_silence_tail_ms
        self._profile = profile
        self._debug = debug
        self._interrupt_words = list(interrupt_words) if interrupt_words else []

        # 惰性加载后端 + 模型（失败抛 BackendNotInstalledError 带提示）
        self._backend = get_backend(self._backend_name, device=device)
        if self._debug:
            print("[RealtimeASR] 加载后端 %s 模型…" % self._backend_name, flush=True)
        self._backend.load()
        if self._backend.sr != self._sr:
            self._sr = self._backend.sr

        # 打断词旁路（T12）：interrupt_words 非空 → 加载轻量 KWS；失败仅告警降级为"无打断"
        self._interrupt_detector = None
        if self._interrupt_words:
            try:
                det = get_interrupt_detector("sherpa", words=self._interrupt_words)
                det.load()
                self._interrupt_detector = det
                if self._debug:
                    print("[RealtimeASR] 打断词旁路就绪: %s" % self._interrupt_words, flush=True)
            except Exception as e:
                print("[RealtimeASR] 打断词检测加载失败（旁路关闭，不影响识别）: %s" % e, flush=True)

        self._vad = EnergyVAD(sample_rate=self._sr, silence_tail_ms=vad_silence_tail_ms)
        self._audio_q = queue.Queue(maxsize=8)     # 有界背压
        self._recog_lock = threading.Lock()        # 模型非线程安全
        self._state_lock = threading.RLock()       # VAD/句子索引等状态（RLock：worker 持锁时 interrupt 内部可重入）
        self._gen = 0
        self._shutdown = False
        self._cb = None
        self._sentences = []                       # profile 时收集
        self._next_idx = 1
        self._t0 = time.monotonic()                # 会话起点（时序基准）

        self._worker = threading.Thread(target=self._worker_loop, name="asr-worker", daemon=True)
        self._worker.start()
        self._inited = True
        if self._debug:
            print("[RealtimeASR] 就绪（backend=%s device=%s sr=%d, VAD 尾长 %dms）"
                  % (self._backend_name, self._device, self._sr, self._vad_tail_ms), flush=True)

    # ------------------------------------------------------------------ 生命周期

    def on_sentence(self, callback):
        """设置句子完成回调 `cb(result: SentenceResult)`。返回旧回调。"""
        old = self._cb
        self._cb = callback
        return old

    def ingest(self, audio, source_ts=None):
        """喂入音频块（16kHz float32，非阻塞入队；识别慢时阻塞 = 背压）。

        source_ts：块第一采样的 monotonic 时刻（None 自动反推）。

        **T12 流式打断旁路**：块先经轻量 KWS `feed()`——命中打断词（如"停下"）
        → `interrupt()` 即时作废全部排队任务，**该触发块丢弃不入队**（"停下"
        本身不进入识别管线）。必须在 ingest 旁路做：打断词若也走队列，它排在
        队尾，等它被处理时前面任务早已完成——打断悖论（ADR 有完整论证）。
        """
        self._check_alive()
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return
        if source_ts is None:
            source_ts = time.monotonic() - len(audio) / self._sr

        # T12 流式旁路：命中 → 作废排队任务 + 丢弃触发块（不识别"停下"）
        det = self._interrupt_detector
        if det is not None:
            try:
                if det.feed(audio):
                    self.interrupt()
                    if self._debug:
                        print("[asr] 打断词命中（流式旁路）→ 排队任务已作废", flush=True)
                    return
            except Exception:
                if self._debug:
                    import traceback
                    traceback.print_exc()

        self._audio_q.put((self._gen, audio, source_ts))

    def ingest_file(self, path, chunk_ms=100):
        """同步识别整个音频文件（重采样 16k + VAD 断句 + 逐句识别）。

        阻塞；返回 `[SentenceResult, ...]`。bench 主用。
        """
        self._check_alive()
        sr, data = read_wav(path)
        data = resample_to(data, sr, self._sr)
        # 文件 t=0 对齐会话起点 _t0 → 结果时间戳 = 文件内相对秒（正数，bench 可比）
        base_ts = self._t0
        chunk = max(int(self._sr * chunk_ms / 1000), 1)
        results = []
        with self._state_lock:
            for i in range(0, len(data), chunk):
                seg = data[i:i + chunk]
                for sent, s_ts, e_ts in self._vad.add(seg, base_ts + i / self._sr):
                    results.append(self._process_sentence_locked(sent, s_ts, e_ts, recog_axis="audio"))
            for sent, s_ts, e_ts in self._vad.flush(base_ts + len(data) / self._sr):
                results.append(self._process_sentence_locked(sent, s_ts, e_ts, recog_axis="audio"))
        return results

    def interrupt(self):
        """打断当前会话：即时作废排队任务 + 清 VAD/后端状态（打断词/上层策略触发）。

        两步设计（T12，ADR 有完整论证）：
        ① `_gen += 1` **即时生效**（GIL 原子，无需锁）——作废语义立即传达：
           排队中的块 gen 不匹配被 worker 丢弃；正在识别的任务完成后判
           `stale=True` 不进普通回调。
        ② 状态清理（VAD/后端 reset + 清队列）持 `_state_lock`——若 worker 正
           占锁（正在识别整句前向），清理会等到该前向结束才执行；但作废语义
           已在①传达，等待只是物理收尾。
        """
        self._gen += 1
        with self._state_lock:
            self._vad.reset()
            self._backend.reset()
            det = self._interrupt_detector
            if det is not None:
                try:
                    det.reset()            # 清 KWS 流式状态，防旧词残留误触发
                except Exception:
                    pass
            while True:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break

    def close(self):
        """销毁（幂等）：停线程 → 关后端 → 清单例槽位。`with`/`__del__`/atexit 兜底。"""
        with self._init_lock:
            if not getattr(self, "_inited", False) or getattr(self, "_closed", False):
                return
            self._closed = True
            self._shutdown = True
            try:
                self._audio_q.put(None)
                self._worker.join(timeout=10)
            except Exception:
                pass
            try:
                self._backend.close()
            except Exception:
                pass
            if self._interrupt_detector is not None:
                try:
                    self._interrupt_detector.close()
                except Exception:
                    pass
            if RealtimeASR._instance is self:
                RealtimeASR._instance = None
            self._inited = False

    def __enter__(self):
        self._check_alive()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ 内部

    def _check_alive(self):
        if not getattr(self, "_inited", False):
            raise RuntimeError("RealtimeASR 已 close()，需重新构造")

    def _worker_loop(self):
        while True:
            item = self._audio_q.get()
            if item is None or self._shutdown:
                break
            gen, audio, ts = item
            if gen != self._gen:
                continue                       # 旧会话块作废
            with self._state_lock:
                for sent, s_ts, e_ts in self._vad.add(audio, ts):
                    if self._interrupt_on_detect(sent):
                        break                   # 打断词命中：该句不识别，本块剩余句子一并作废
                    # task_gen 用「出队时的块 gen」而非当前 _gen：interrupt() 的
                    # _gen+=1 可能发生在 worker 处理本块中途（VAD 已含触发词音频），
                    # 此时该块产出的句子仍属旧会话 → stale 丢弃（T12c 实测修复）。
                    self._process_sentence_locked(sent, s_ts, e_ts, task_gen=gen)

    def _process_sentence_locked(self, audio, s_ts, e_ts, recog_axis="wall", task_gen=None):
        """已持 _state_lock。分配 idx → 识别（持 _recog_lock）→ 构造结果 → 回调。

        recog_axis：时间轴基准。
          - "wall"（实时流）：recog 时刻 = monotonic 相对 _t0；ttfb 天然含 VAD 尾长 + 识别延迟。
          - "audio"（文件同步）：加速喂入，识别时刻映射到音频轴（断句即识别），
            ttfb = 纯识别耗时，与 feed 加速无关。

        T12 stale：识别完成后若 `task_gen != self._gen`（识别期间被 interrupt() 打断），
        该结果标记 stale=True，**不进普通回调**。实时流（worker）传**出队块的 gen**
        作 task_gen（interrupt 的 _gen+=1 可能发生在处理本块中途）；文件同步路径
        默认取当前 _gen（无并发打断语义）。
        """
        if task_gen is None:
            task_gen = self._gen              # 文件同步路径：当前代际
        idx = self._next_idx
        self._next_idx += 1
        t1 = SentenceResult.now()
        with self._recog_lock:
            text = self._backend.recognize(audio)
        t2 = SentenceResult.now()
        stale = (task_gen != self._gen)       # 识别期间被打断？
        a_start, a_end = s_ts - self._t0, e_ts - self._t0
        if recog_axis == "audio":
            recog_start, recog_end = a_end, a_end + (t2 - t1)
        else:
            recog_start, recog_end = t1 - self._t0, t2 - self._t0
        result = SentenceResult(idx, text, a_start, a_end, recog_start, recog_end, stale=stale)
        if self._profile:
            self._sentences.append(result)    # profile 仍收集（含 stale，供诊断）
        if not stale and self._cb:
            try:
                self._cb(result)
            except Exception:
                if self._debug:
                    import traceback
                    traceback.print_exc()
        if self._debug:
            print("[asr #%d] %s（%.2fs）%s" % (idx, text, result.ttfb if result.ttfb else 0,
                                                "[stale]" if stale else ""), flush=True)
        return result

    def _interrupt_on_detect(self, sent):
        """VAD 断句后旁路预检：命中打断词 → interrupt() 作废排队任务，返回 True。

        已持 `_state_lock`（RLock 可重入 interrupt 内部的状态清理）。
        检测器缺失（interrupt_words 未设 / 加载失败降级）→ 恒 False，零开销。
        """
        det = self._interrupt_detector
        if det is None:
            return False
        try:
            if det.detect(sent):
                self.interrupt()
                if self._debug:
                    print("[asr] 打断词命中 → 排队任务已作废", flush=True)
                return True
        except Exception:
            if self._debug:
                import traceback
                traceback.print_exc()
        return False
