# -*- coding: utf-8 -*-
"""SherpaBackend：sherpa-onnx 流式 zipformer 基线后端。

定位：CPU 轻量对照基线（RTF≈0.04，加载 1.2s），类比 voice0 的 `sapi/`。
模型首次联网下载（ghfast.top 代理），缓存落 `.cache/sherpa_models/`。
"""
import os
import tarfile
import urllib.request

import numpy as np

from ..core.backend import ASRBackend, BackendNotInstalledError

_MODEL_URL = ("https://ghfast.top/https://github.com/k2-fsa/sherpa-onnx/releases/download/"
              "asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2")
_MODEL_NAME = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
_MODEL_SUBDIR = "sherpa_models"

_CFG = dict(
    num_threads=4, sample_rate=16000, feature_dim=80,
    enable_endpoint_detection=False,
)


class SherpaBackend(ASRBackend):
    name = "sherpa"
    sr = 16000
    supports_streaming = True

    def __init__(self, device="auto", model_url=None, debug=False):
        self._device = device
        self._model_url = model_url or _MODEL_URL
        self._model_dir = None
        self._rec = None
        self._stream = None

    # -- 模型路径（缓存于项目内 .cache/sherpa_models）----------------------
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
        print("[SherpaBackend] 首次联网下载模型…", flush=True)
        urllib.request.urlretrieve(self._model_url, tarball)
        with tarfile.open(tarball) as tf:
            tf.extractall(self._cache_dir())
        os.remove(tarball)
        self._model_dir = model_dir

    # -- ASRBackend 协议 ------------------------------------------------
    def load(self):
        try:
            import sherpa_onnx
        except ImportError:
            raise BackendNotInstalledError(
                "sherpa-onnx 未安装。请 `pip install sherpa-onnx` 后重试。")
        self._ensure_model()
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(self._model_dir, "tokens.txt"),
            encoder=os.path.join(self._model_dir, "encoder-epoch-99-avg-1.onnx"),
            decoder=os.path.join(self._model_dir, "decoder-epoch-99-avg-1.onnx"),
            joiner=os.path.join(self._model_dir, "joiner-epoch-99-avg-1.onnx"),
            **_CFG)
        self._stream = self._rec.create_stream()

    def recognize(self, audio):
        """整段（句子级）识别。"""
        self._stream = self._rec.create_stream()
        a = np.ascontiguousarray(audio, dtype=np.float32)
        self._stream.accept_waveform(self.sr, a)
        self._stream.input_finished()
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        return self._rec.get_result(self._stream).strip()

    def recognize_stream(self, chunk, is_final=False):
        """在线流式增量（sherpa 原生支持，T13）：返回**累计**部分/完整文本。

        `is_final=True` 收尾：`input_finished()` 结束输入 → 解码出最终文本，
        并重建 stream（input_finished 后不可再喂；引擎随后也会 reset）。
        """
        a = np.ascontiguousarray(chunk, dtype=np.float32)
        self._stream.accept_waveform(self.sr, a)
        if is_final:
            self._stream.input_finished()
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        text = self._rec.get_result(self._stream).strip()
        if is_final:
            self._stream = self._rec.create_stream() if self._rec else None
        return text

    def reset(self):
        """新句子 / 会话打断：清流式状态。"""
        self._stream = self._rec.create_stream() if self._rec else None

    def close(self):
        self._rec = None
        self._stream = None
