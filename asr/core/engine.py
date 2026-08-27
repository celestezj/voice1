# -*- coding: utf-8 -*-
"""RealtimeASR：后端无关的实时语音识别引擎（单例 · 常驻线程 · 完整生命周期）。

设计沿用 voice0 的 RealtimeTTS 骨架，适配 ASR 的"连续音频流"语义：
- **单例**：模型只加载一次；backend/device 变更 → 销毁重建，其余返回同一实例。
- **常驻处理线程**：`_worker` 消费 `_audio_q` 音频块 → VAD 断句 → 持 `_recog_lock` 识别 → 回调。
- **有界背压**：`_audio_q` maxsize=8 —— 识别慢于喂入时阻塞喂入方，防延迟膨胀。
- **会话代际 `_gen`**：`interrupt()` 计数 +1，旧会话音频块作废；VAD/后端状态同步重置。
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


class RealtimeASR:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, backend="paraformer", device="auto", sample_rate=16000,
                vad_silence_tail_ms=250, profile=False, debug=False):
        # 单例：backend/device/vad_silence_tail_ms 任一变更 → 销毁重建；否则返回同一实例（模型不二次加载）
        inst = cls._instance
        if inst is not None and (inst._backend_name != backend or inst._device != device
                                 or inst._vad_tail_ms != vad_silence_tail_ms):
            inst.close()
            inst = None
        if inst is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend="paraformer", device="auto", sample_rate=16000,
                 vad_silence_tail_ms=250, profile=False, debug=False):
        if getattr(self, "_inited", False):
            return
        self._backend_name = backend or "paraformer"
        self._device = device or "auto"
        self._sr = int(sample_rate)
        self._vad_tail_ms = vad_silence_tail_ms
        self._profile = profile
        self._debug = debug

        # 惰性加载后端 + 模型（失败抛 BackendNotInstalledError 带提示）
        self._backend = get_backend(self._backend_name, device=device)
        if self._debug:
            print("[RealtimeASR] 加载后端 %s 模型…" % self._backend_name, flush=True)
        self._backend.load()
        if self._backend.sr != self._sr:
            self._sr = self._backend.sr

        self._vad = EnergyVAD(sample_rate=self._sr, silence_tail_ms=vad_silence_tail_ms)
        self._audio_q = queue.Queue(maxsize=8)     # 有界背压
        self._recog_lock = threading.Lock()        # 模型非线程安全
        self._state_lock = threading.Lock()        # VAD/句子索引等状态
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
        """
        self._check_alive()
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return
        if source_ts is None:
            source_ts = time.monotonic() - len(audio) / self._sr
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
        """打断当前会话：旧块作废，VAD/后端状态重置（新说话人/新会话用）。"""
        with self._state_lock:
            self._gen += 1
            self._vad.reset()
            self._backend.reset()
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
                    self._process_sentence_locked(sent, s_ts, e_ts)

    def _process_sentence_locked(self, audio, s_ts, e_ts, recog_axis="wall"):
        """已持 _state_lock。分配 idx → 识别（持 _recog_lock）→ 构造结果 → 回调。

        recog_axis：时间轴基准。
          - "wall"（实时流）：recog 时刻 = monotonic 相对 _t0；ttfb 天然含 VAD 尾长 + 识别延迟。
          - "audio"（文件同步）：加速喂入，识别时刻映射到音频轴（断句即识别），
            ttfb = 纯识别耗时，与 feed 加速无关。
        """
        idx = self._next_idx
        self._next_idx += 1
        t1 = SentenceResult.now()
        with self._recog_lock:
            text = self._backend.recognize(audio)
        t2 = SentenceResult.now()
        a_start, a_end = s_ts - self._t0, e_ts - self._t0
        if recog_axis == "audio":
            recog_start, recog_end = a_end, a_end + (t2 - t1)
        else:
            recog_start, recog_end = t1 - self._t0, t2 - self._t0
        result = SentenceResult(idx, text, a_start, a_end, recog_start, recog_end)
        if self._profile:
            self._sentences.append(result)
        if self._cb:
            try:
                self._cb(result)
            except Exception:
                if self._debug:
                    import traceback
                    traceback.print_exc()
        if self._debug:
            print("[asr #%d] %s（%.2fs）" % (idx, text, result.ttfb if result.ttfb else 0), flush=True)
        return result
