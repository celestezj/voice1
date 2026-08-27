# -*- coding: utf-8 -*-
"""ParaformerBackend：FunASR paraformer-zh-streaming 主力后端。

定位：GPU 主力（T4 实测 GPU RTF 低、CER 最优），类比 voice0 的 `melo/`。
首次联网下载模型（modelscope），缓存落 `.cache/modelscope/`。
- `recognize`：句子级识别（引擎 VAD 断句后调用）。`is_final=True` 关键——数组输入默认
  `is_final=False`，末 chunk 会被丢弃（实测整句截尾缺"技术系"）。
- `recognize_stream(chunk, is_final=False)`：FunASR 官方 cache 流式（T13 接入引擎）。
  **注意**：FunASR `generate` 流式返回的是**增量 delta**（实测"明天早"→"上八点"→"开会"），
  不是累计假设——本类内部把 delta 拼进 `_partial_buf`，对外统一返回**累计**文本；
  `is_final=True` 句末收尾返回该句完整文本并清状态。`reset()` 清 cache + partial_buf。
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
    supports_streaming = True

    def __init__(self, device="auto", model_id=None):
        self._device = device
        self._model_id = model_id or _MODEL_ID
        self._model = None
        self._cache = None
        self._partial_buf = ""       # 流式 delta 累积（FunASR 返回增量，须拼接成累计文本）

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

    def recognize_stream(self, chunk, is_final=False):
        """流式增量（T13）：逐块出字 + 句末 flush 定稿。

        FunASR `generate` 流式返回**增量 delta**（非累计），内部拼进 `_partial_buf`，
        对外统一返回**累计**文本（部分/完整）。`is_final=True` 收尾返回完整句文本并清状态。
        """
        if self._cache is None:
            self._cache = {}
        a = np.ascontiguousarray(chunk, dtype=np.float32)
        res = self._model.generate(input=a, cache=self._cache, is_final=is_final, **_CFG)
        delta = (res[0]["text"] if isinstance(res, list)
                 else res.get("text", str(res))).strip()
        self._partial_buf += delta
        if is_final:
            final = self._partial_buf
            self._partial_buf = ""       # 流结束：清累积 + cache（下句从零开始）
            self._cache = None
            return final
        return self._partial_buf

    def reset(self):
        """清流式状态（新句子 / interrupt）。"""
        self._cache = None
        self._partial_buf = ""

    def close(self):
        self._model = None
        self._cache = None
        self._partial_buf = ""
