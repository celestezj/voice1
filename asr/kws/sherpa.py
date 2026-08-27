# -*- coding: utf-8 -*-
"""SherpaKwsDetector：sherpa-onnx KeywordSpotter 打断词检测（T12 默认实现）。

极轻量独立模型（zipformer wenetspeech 3.3M int8），专司关键词命中，
不走主 ASR 队列——"停下"一说出口，VAD 断句后毫秒级命中 → 引擎 interrupt()。
模型首次联网下载（ghfast.top 代理），缓存落 `.cache/kws_models/`。

关键词文件：KWS 建模单元是"拼音（声母+韵母）"，所以 keywords.txt 不能直接写
汉字，需经 pypinyin 把汉字转成空格分隔的带声调音节串，如 `停下 → t íng x ià @停下`。
注意 **不能用 sherpa_onnx.utils.text2token(ppinyin)**——它会把组合声母
`zh/sh/ch` 拆成 `z h`、把带调韵母 `uò` 拆成 `u ò`，与 tokens.txt 建模单元不符
（实测 T12c；本实现自建转换与官方 test_keywords.txt 逐词一致）。
词集在 load 时按构造传入的 words 生成，可增删后重建（替换引擎配置即可）。
"""


def _word_to_tokens(word):
    """汉字 → ppinyin token 列表（声母 + 带调韵母，组合声母不拆开）。

    与 KWS 模型 tokens.txt 建模单元一致（实测对照官方 8 词全对）。
    依赖 pypinyin（`pip install pypinyin`）。
    """
    from pypinyin import pinyin
    from pypinyin.contrib.tone_convert import to_initials, to_finals_tone
    res = []
    for syl in (x[0] for x in pinyin(word)):
        ini = to_initials(syl, strict=False)
        fin = to_finals_tone(syl, strict=False)
        if ini:
            res.append(ini)
        if fin:
            res.append(fin)
    return res
import os
import tarfile
import threading
import urllib.request

import numpy as np

from ..core.backend import BackendNotInstalledError
from .interrupt import InterruptDetector

_MODEL_URL = ("https://ghfast.top/https://github.com/k2-fsa/sherpa-onnx/releases/download/"
              "kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2")
_MODEL_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
_MODEL_SUBDIR = "kws_models"
_CFG = dict(num_threads=2, sample_rate=16000, feature_dim=80)

_DEFAULT_WORDS = ["停下"]


def _pick_model_file(model_dir, pat, *prefers):
    """按偏好选模型文件：`*int8.onnx` 优先（官方推荐组合），其次 epoch-12，最后任意。"""
    import glob
    cands = glob.glob(os.path.join(model_dir, pat))
    for prefer in prefers:
        for c in cands:
            if prefer in c:
                return c
    return cands[0]


class SherpaKwsDetector(InterruptDetector):
    name = "sherpa"
    sr = 16000

    def __init__(self, words=None, **cfg):
        self._words = list(words) if words else list(_DEFAULT_WORDS)
        self._cfg = dict(_CFG, **cfg)
        self._model_url = _MODEL_URL
        self._model_dir = None
        self._kws = None
        self._keywords_file = None
        # 模型非线程安全：ingest 旁路（主线程 feed）与 worker 兜底（detect）
        # 可能并发调用 KeywordSpotter——同一把锁串行化全部 KWS 调用。
        self._lock = threading.Lock()

    # -- 模型与关键词文件路径（缓存于项目内 .cache/kws_models）-----------
    @staticmethod
    def _cache_dir():
        proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(proj, ".cache", _MODEL_SUBDIR)

    def _ensure_model(self):
        model_dir = os.path.join(self._cache_dir(), _MODEL_NAME)
        if os.path.exists(model_dir):
            self._model_dir = model_dir
            return
        os.makedirs(self._cache_dir(), exist_ok=True)
        tarball = os.path.join(self._cache_dir(), "model.tar.bz2")
        print("[SherpaKwsDetector] 首次联网下载打断词模型…", flush=True)
        urllib.request.urlretrieve(self._model_url, tarball)
        with tarfile.open(tarball) as tf:
            tf.extractall(self._cache_dir())
        os.remove(tarball)
        self._model_dir = model_dir

    def _ensure_keywords_file(self):
        """汉字词集 → 拼音音节串（KWS 建模单元），每次 load 重写（词集可随配置变化）。

        转换用 `_word_to_tokens`（pypinyin 拆声母+韵母，组合声母不拆开，
        与 tokens.txt 建模单元一致）。依赖 pypinyin（缺失在 load 抛错→引擎降级）。
        """
        os.makedirs(self._cache_dir(), exist_ok=True)
        path = os.path.join(self._cache_dir(), "keywords.txt")
        with open(path, "w", encoding="utf-8") as f:
            for w in self._words:
                flat = " ".join(_word_to_tokens(w))
                f.write("%s @%s\n" % (flat, w))   # `t íng x ià @停下`
        self._keywords_file = path

    # -- InterruptDetector 协议 ------------------------------------------
    def load(self):
        try:
            import sherpa_onnx
        except ImportError:
            raise BackendNotInstalledError(
                "sherpa-onnx 未安装，无法启用打断词检测。请 `pip install sherpa-onnx` "
                "后重试，或不传 interrupt_words 关闭打断旁路。")
        self._ensure_model()
        self._ensure_keywords_file()
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(self._model_dir, "tokens.txt"),
            encoder=_pick_model_file(self._model_dir, "encoder-*.onnx", "int8", "epoch-12"),
            decoder=_pick_model_file(self._model_dir, "decoder-*.onnx", "int8", "epoch-12"),
            joiner=_pick_model_file(self._model_dir, "joiner-*.onnx", "int8", "epoch-12"),
            keywords_file=self._keywords_file,
            **self._cfg)
        self._stream = self._kws.create_stream()   # 流式旁路：常驻一个检测流
        return self

    def feed(self, chunk):
        """流式旁路检测（主路径）：喂一个音频块，命中打断词 → True（已 reset）。

        常驻 stream 持续监听；命中即 reset，等待下一次"停下"。引擎在 ingest
        路径调用——用户一说"停下"（无需等 VAD 断句）即可触发打断，作废
        队列中（含打断词之前）的所有任务。
        """
        if self._kws is None or self._stream is None:
            return False
        a = np.ascontiguousarray(chunk, dtype=np.float32)
        if len(a) == 0:
            return False
        with self._lock:                    # 与 detect/reset 串行（模型非线程安全）
            self._stream.accept_waveform(self.sr, a)
            hit = False
            while self._kws.is_ready(self._stream):
                self._kws.decode_stream(self._stream)
                r = self._kws.get_result(self._stream)
                if r != "":
                    hit = True
                    self._kws.reset_stream(self._stream)
            return hit

    def detect(self, audio):
        """整段检测（兜底）：audio 是 VAD 断句出的一个句子（16k float32）。

        补 0.66s 尾静音再 input_finished——streaming KWS 需要尾音把解码收尾
        （官方 keyword-spotter.py 同款，否则短句/关键词可能漏检）。
        每次新建 stream（独立于 feed 的常驻流），无跨调用状态。
        """
        if self._kws is None:
            return False
        a = np.ascontiguousarray(audio, dtype=np.float32)
        tail = np.zeros(int(0.66 * self.sr), dtype=np.float32)
        with self._lock:                    # 与 feed/reset 串行（模型非线程安全）
            stream = self._kws.create_stream()
            stream.accept_waveform(self.sr, a)
            stream.accept_waveform(self.sr, tail)
            stream.input_finished()
            hit = False
            while self._kws.is_ready(stream):
                self._kws.decode_stream(stream)
                r = self._kws.get_result(stream)
                if r != "":
                    hit = True
                    self._kws.reset_stream(stream)
            return hit

    def reset(self):
        """清流式检测状态（会话打断/切换时）。"""
        if self._kws is not None and self._stream is not None:
            with self._lock:
                self._kws.reset_stream(self._stream)

    def close(self):
        self._kws = None
        self._stream = None
