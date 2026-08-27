# -*- coding: utf-8 -*-
"""ParaformerBackend：FunASR paraformer-zh-streaming 主力后端。

定位：GPU 主力（T4 实测 GPU RTF 低、CER 最优），类比 voice0 的 `melo/`。
首次联网下载模型（modelscope），缓存落 `.cache/modelscope/`。
- `recognize`：句子级识别（引擎 VAD 断句后调用）。`is_final=True` 关键——数组输入默认
  `is_final=False`，末 chunk 会被丢弃（实测整句截尾缺"技术系"）。
- `recognize_stream`：FunASR 官方 cache 流式（60ms 粒度逐块增量，跨调用保持状态）。
  引擎暂以"VAD 断句 + 整句 recognize"为主，此接口留给 T10 帧级实时增强；`reset()` 清 cache。
"""
import os

import numpy as np

from ..core.backend import ASRBackend, BackendNotInstalledError

# FunASR 模型缓存指到项目内 .cache/modelscope（必须在 import funasr 前设置环境变量）
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(_PROJ, ".cache", "modelscope"))
os.environ.setdefault("MODELSCOPE_HUB_CACHE", os.path.join(_PROJ, ".cache", "modelscope"))

_MODEL_ID = "paraformer-zh-streaming"

# 流式 chunk 参数：encoder 5帧(50ms) / decoder 10帧(100ms)，look_back 带历史上下文
_CFG = dict(chunk_size=[5, 10, 5], encoder_chunk_look_back=4, decoder_chunk_look_back=1)


class ParaformerBackend(ASRBackend):
    name = "paraformer"
    sr = 16000

    def __init__(self, device="auto", model_id=None):
        self._device = device
        self._model_id = model_id or _MODEL_ID
        self._model = None
        self._cache = None

    # -- ASRBackend 协议 ------------------------------------------------
    def load(self):
        try:
            from funasr import AutoModel
        except ImportError:
            raise BackendNotInstalledError(
                "FunASR 未安装。请 `pip install funasr modelscope torchaudio` 后重试。")
        device = self._device
        if device in ("auto", None):
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._model = AutoModel(model=self._model_id, device=device, disable_update=True)

    def recognize(self, audio):
        """整段（句子级）识别。is_final=True：末 chunk 不截断。"""
        a = np.ascontiguousarray(audio, dtype=np.float32)
        res = self._model.generate(input=a, is_final=True, **_CFG)
        return (res[0]["text"] if isinstance(res, list)
                else res.get("text", str(res))).strip()

    def recognize_stream(self, chunk):
        """流式增量（FunASR cache 模式，60ms 粒度逐块输出）。

        跨调用保持 cache 状态；`reset()` 在句子边界 / 打断时清状态。
        句子级识别请走 `recognize`（is_final=True，不截断）。
        """
        if self._cache is None:
            self._cache = {}
        a = np.ascontiguousarray(chunk, dtype=np.float32)
        res = self._model.generate(input=a, cache=self._cache, is_final=False, **_CFG)
        return (res[0]["text"] if isinstance(res, list)
                else res.get("text", str(res))).strip()

    def reset(self):
        """清流式状态（新句子 / interrupt）。"""
        self._cache = None

    def close(self):
        self._model = None
        self._cache = None
