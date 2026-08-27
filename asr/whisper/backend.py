# -*- coding: utf-8 -*-
"""WhisperBackend：faster-whisper medium 离线可选项。

定位：GPU 高精度可选项（CER 下限最优），类比 voice0 的备用音源。
离线非流式：句子级识别质量上限高，但无逐块增量——`recognize_stream` 抛 NotImplementedError，
引擎按 ABC 契约回退"积累块 + 整句 recognize"（"边说边出字"不可用，选型时已知权衡）。
首次联网下载（hf-mirror 镜像），缓存落 `.cache/hf/`。
"""
import os

import numpy as np

from ..core.backend import ASRBackend, BackendNotInstalledError

# huggingface 直连被墙 → hf-mirror 镜像；缓存落项目内 .cache/hf（须在 import faster_whisper 前）
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(_PROJ, ".cache", "hf"))

_MODEL_ID = "medium"
_REPO_ID = "Systran/faster-whisper-medium"


class WhisperBackend(ASRBackend):
    name = "whisper"
    sr = 16000
    supports_streaming = False     # 离线非流式：recognize_stream 抛 NotImplementedError

    def __init__(self, device="auto", model_id=None, language="zh", beam_size=5):
        self._device = device
        self._model_id = model_id or _MODEL_ID
        self._language = language
        self._beam_size = beam_size
        self._model = None

    # -- 本地缓存路径（离线零网络优先）---------------------------------
    @staticmethod
    def _local_model_dir():
        hub = os.path.join(_PROJ, ".cache", "hf", "hub",
                           "models--" + _REPO_ID.replace("/", "--"), "snapshots")
        if os.path.isdir(hub):
            for d in sorted(os.listdir(hub)):
                cand = os.path.join(hub, d)
                if os.path.isfile(os.path.join(cand, "model.bin")):
                    return cand
        return None

    # -- ASRBackend 协议 ------------------------------------------------
    def load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise BackendNotInstalledError(
                "faster-whisper 未安装。请 `pip install faster-whisper` 后重试。")
        device = self._device
        if device in ("auto", None):
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        # 本地缓存优先（hf-mirror 联网不可靠 + 满足"缓存后零网络"）；缺失才走 hub 下载
        model = self._local_model_dir() or self._model_id
        compute_type = "float16" if device.startswith("cuda") else "int8"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def recognize(self, audio):
        """整段（句子级）识别。"""
        a = np.ascontiguousarray(audio, dtype=np.float32)
        segments, _ = self._model.transcribe(a, language=self._language,
                                             beam_size=self._beam_size)
        return "".join(seg.text.strip() for seg in segments).strip()

    def recognize_stream(self, chunk, is_final=False):
        raise NotImplementedError(
            "faster-whisper 离线模型不支持流式增量；引擎回退积累块 + 整句 recognize")

    def reset(self):
        """离线模型无跨调用状态。"""
        pass

    def close(self):
        self._model = None
