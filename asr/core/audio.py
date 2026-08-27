# -*- coding: utf-8 -*-
"""音频工具：读 WAV、重采样、能量 VAD 断句。

core 只依赖 numpy + scipy（重采样），不 import 任何推理框架。
VAD 是 ASR 的"分句器"——静音尾长（silence_tail_ms）是延迟-准确率的旋钮：
太小→断句过碎；太大→尾字延迟变长。参数化并实测标定（T10）。
"""
import math
import time
import wave

import numpy as np
from scipy import signal


def read_wav(path):
    """读取 WAV → (sr, 1D float32 mono)。"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:                      # 取均值转单声道
        data = data.reshape(-1, nch).mean(axis=1)
    return sr, data


def resample_to(audio, src_sr, dst_sr):
    """重采样到目标采样率（scipy polyphase；core 级轻依赖）。"""
    if src_sr == dst_sr:
        return audio
    gcd = math.gcd(src_sr, dst_sr)
    return signal.resample_poly(np.asarray(audio, dtype=np.float32),
                                dst_sr // gcd, src_sr // gcd).astype(np.float32)


class EnergyVAD:
    """能量 VAD 状态机：silence ⇄ speech，静音尾长判定句末。

    add(audio) 逐帧推进状态；检测到断句时返回该句音频（含尾部少量静音保边）。
    """

    def __init__(self, sample_rate=16000, frame_ms=20, silence_tail_ms=600,
                 threshold_db=-35.0, min_speech_ms=250):
        self.sr = sample_rate
        self.frame_len = max(int(sample_rate * frame_ms / 1000), 1)
        self.silence_tail_frames = max(int(silence_tail_ms / frame_ms), 1)
        self.min_speech_frames = max(int(min_speech_ms / frame_ms), 1)
        self.threshold_db = threshold_db
        self.reset()

    def _frame_db(self, frame):
        rms = np.sqrt(np.mean(np.square(frame))) + 1e-12
        return 20.0 * math.log10(rms)

    def reset(self):
        """清状态（会话打断 / 新开始）。"""
        self._speech = False
        self._silence = 0
        self._speech_frames = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._sentence_start = 0       # 当前句子在 _buf 的起始采样
        self._scan = 0                 # 已扫描采样偏移
        self._ts_cur = None            # _buf 第一个采样的时刻（monotonic 秒）

    def add(self, audio, ts=None):
        """喂音频块（float32），返回断句完成的句子 `[(audio, start_ts, end_ts)]`（可空）。

        ts：本块第一个采样的时刻（monotonic 秒）；None 时按调用时刻反推。
        已扫描但未断句的语音帧保留（属当前句）；只修剪句前静音，防缓冲无限增长。
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return []
        if ts is None:
            ts = time.monotonic() - len(audio) / self.sr
        if len(self._buf) == 0:
            self._ts_cur = ts
        self._buf = np.concatenate([self._buf, audio])
        sentences = []
        i = self._scan
        n = len(self._buf)
        while i + self.frame_len <= n:
            db = self._frame_db(self._buf[i:i + self.frame_len])
            if db >= self.threshold_db:
                if not self._speech:
                    self._sentence_start = i
                self._speech = True
                self._silence = 0
                self._speech_frames += 1
                i += self.frame_len
            elif not self._speech:
                i += self.frame_len
            else:
                self._silence += 1
                if self._silence >= self.silence_tail_frames:
                    # 断句：句子 = [_sentence_start, cut)，保留 tail 帧静音保边
                    end = i + self.frame_len
                    keep = self.silence_tail_frames * self.frame_len
                    cut = max(end - keep, 0)
                    if self._speech_frames >= self.min_speech_frames and cut > self._sentence_start:
                        s_ts = self._ts_cur + self._sentence_start / self.sr
                        e_ts = self._ts_cur + cut / self.sr
                        sentences.append((self._buf[self._sentence_start:cut].copy(), s_ts, e_ts))
                    drop = cut
                    self._buf = self._buf[drop:]
                    self._ts_cur = self._ts_cur + drop / self.sr
                    self._speech = False
                    self._silence = 0
                    self._speech_frames = 0
                    self._sentence_start = 0
                    i = 0
                    n = len(self._buf)
                else:
                    i += self.frame_len
        # 循环后：修剪句前静音（_sentence_start 之前），重新对齐扫描位置
        self._scan = i
        if self._sentence_start > 0:
            drop = self._sentence_start
            self._buf = self._buf[drop:]
            self._ts_cur = self._ts_cur + drop / self.sr
            self._scan = max(i - drop, 0)
            self._sentence_start = 0
        if len(self._buf) == 0:
            self._ts_cur = None
            self._scan = 0
        return sentences

    def flush(self, ts=None):
        """收尾：把残留缓冲作为最后一句吐出（若够长）。"""
        if ts is None:
            ts = time.monotonic()
        start = self._sentence_start
        sent = self._buf[start:]
        self._buf = np.zeros(0, dtype=np.float32)
        self._ts_cur = None
        self._scan = 0
        self._sentence_start = 0
        self._speech = False
        self._silence = 0
        self._speech_frames = 0
        if len(sent) >= self.min_speech_frames * self.frame_len:
            return [(sent.copy(), ts - len(sent) / self.sr, ts)]
        return []
