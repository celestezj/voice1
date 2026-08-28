# -*- coding: utf-8 -*-
"""WhisperBackend：faster-whisper 离线可选项（medium 默认 / large-v3-turbo 高精度）。

定位：GPU 高精度可选项（CER 下限最优），类比 voice0 的备用音源。
离线非流式：句子级识别质量上限高，但无逐块增量——`recognize_stream` 抛 NotImplementedError，
引擎按 ABC 契约回退"积累块 + 整句 recognize"（"边说边出字"不可用，选型时已知权衡）。
首次联网下载（hf-mirror 镜像），缓存落 `.cache/hf/`。

模型档位（`model_id` 参数 / get_backend 注册名 whisper=medium、whisper-large=large-v3-turbo）：
- `medium`（默认）：现有基线（语料严格 CER 0.120，GPU 可用）。
- `large-v3-turbo`（高精度）：自回归强 LM，对同音字/罕见文学词纠错最强
  （large 明显强于 medium），8GB 显存可跑（float16 ~1.6GB）。
"""
import os

import numpy as np

from ..core.backend import ASRBackend, BackendNotInstalledError

# huggingface 直连被墙 → hf-mirror 镜像；缓存落项目内 .cache/hf（须在 import faster_whisper 前）
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(_PROJ, ".cache", "hf"))

# 档位 → CTranslate2 模型仓库（faster-whisper 直接吃 repo id）。
# 注意：Systran 官方**没有** large-v3-turbo 转换（只有 large-v3），turbo 用社区
# mobiuslabsgmbh 转换（镜像上 404 验证过 Systran 版本不存在）。
_MODELS = {
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


class WhisperBackend(ASRBackend):
    name = "whisper"
    sr = 16000
    supports_streaming = False     # 离线非流式：recognize_stream 抛 NotImplementedError

    def __init__(self, device="auto", model_id=None, language="zh", beam_size=5):
        mid = model_id or "medium"
        if mid not in _MODELS:
            raise ValueError("未知 whisper 模型: %r（可选: %s）"
                             % (mid, "/".join(sorted(_MODELS))))
        self._device = device
        self._model_id = mid
        self._repo_id = _MODELS[mid]
        self._language = language
        self._beam_size = beam_size
        self._model = None

    # -- 本地缓存路径（离线零网络优先）---------------------------------
    def _local_model_dir(self):
        hub = os.path.join(_PROJ, ".cache", "hf", "hub",
                           "models--" + self._repo_id.replace("/", "--"), "snapshots")
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
