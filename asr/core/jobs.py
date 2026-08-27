# -*- coding: utf-8 -*-
"""一次句子识别的结果对象（连续流识别，按句产出）。

timing schema（**bench/bench_asr.py 强依赖，改动必须同步 bench**）：
- idx          句子序号（从 1 起）
- text         识别文本
- audio_start  该句音频起点（会话 start 起的 monotonic 秒）
- audio_end    该句音频终点（VAD 断句时刻）
- recog_start / recog_end  识别线程实际推理起止
- ttfb         尾字延迟 = recog_end - audio_end（说话结束 → 文本回调）
- stale        识别期间被 interrupt() 打断 → True。**该结果不进普通回调**
               （T12 打断词旁路；上层如需"全量语音记录"可经 profile 观测）。
"""
import time


class SentenceResult:
    __slots__ = ("idx", "text", "audio_start", "audio_end",
                 "recog_start", "recog_end", "ttfb", "stale")

    def __init__(self, idx, text, audio_start, audio_end,
                 recog_start, recog_end, stale=False):
        self.idx = idx
        self.text = text
        self.audio_start = audio_start
        self.audio_end = audio_end
        self.recog_start = recog_start
        self.recog_end = recog_end
        self.ttfb = (recog_end - audio_end) if recog_end is not None and audio_end is not None else None
        self.stale = bool(stale)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @staticmethod
    def now():
        return time.monotonic()


class PartialResult:
    """流式逐块出字回调的结果对象（T13，说话期间"边说边出字"）。

    - text        累计部分文本（后端内部拼 delta，非增量）——供实时展示
    - audio_start 本块起点（会话 _t0 起的秒，wall 轴）
    - wall_ts     本回调发生的墙钟（_t0 起），首字延迟 = 首个非空 partial 的 wall_ts
    """

    __slots__ = ("text", "audio_start", "wall_ts")

    def __init__(self, text, audio_start, wall_ts):
        self.text = text
        self.audio_start = audio_start
        self.wall_ts = wall_ts
