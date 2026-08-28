# -*- coding: utf-8 -*-
"""ParaformerBackend：FunASR paraformer-large（online 流式 / offline 整句）后端。

定位：GPU 主力（T4 实测 GPU RTF 低、CER 最优），类比 voice0 的 `melo/`。
首次联网下载模型（modelscope）缓存落 `.cache/modelscope/`；**此后加载走本地路径零网络**
（`_local_model_dir`，避开每次启动的 hub 文件清单请求）。

两种变体（`variant` 参数，get_backend 注册名 paraformer / paraformer-offline）：
- `online`（默认）：`paraformer-zh-streaming`，**流式**（T13 接入引擎），100ms chunk 实时出字。
  为延迟牺牲语言上下文 → 罕见文学词同音字（神妙/神庙）易选错。
- `offline`：`speech_paraformer-large...-vocab8404-pytorch`，**整句非流式**，全上下文无 chunk 约束，
  同族离线版准确率更高（文件转写高精度选项，实时流式仍用 online）。

`recognize`：句子级识别（引擎 VAD 断句后调用）。online 变体 `is_final=True` 关键——数组输入
默认 `is_final=False`，末 chunk 会被丢弃（实测整句截尾缺"技术系"）；offline 变体用原生 generate。
`recognize_stream(chunk, is_final=False)`：FunASR 官方 cache 流式（T13 接入引擎，仅 online）。
  **注意**：FunASR `generate` 流式返回的是**增量 delta**（实测"明天早"→"上八点"→"开会"），
  不是累计假设——本类内部把 delta 拼进 `_partial_buf`，对外统一返回**累计**文本；
  `is_final=True` 句末收尾返回该句完整文本并清状态。`reset()` 清 cache + partial_buf。
  offline 变体抛 NotImplementedError（引擎整句路径兜底）。

热词纠错（`hotword_file` 参数，T16）：FunASR 1.4.4 内置**文本级 postprocess hotwords**——
rapidfuzz + pypinyin **拼音级模糊匹配**（神庙/神妙 拼音同为 shenmiao → 自动替换）。
实测 paraformer-offline + 热词文件 {神妙 心天 四时 季候 减少于} 锚定句同音字 100% 修复。
**实现要点**：纠错不在 generate 时透传 postprocess_hotword_file（流式 delta 是增量片段，
跨块单词"神/庙"分两次返回，片段内匹配不到目标词），而在**本类统一对返回文本/累计
buf 调 `_correct()`**——online 流式对 `_partial_buf` 累计文本纠（跨块词可命中），
offline 整句对完整文本纠。对已纠正文本重复应用是幂等的（fuzzy 中 segment==target 跳过）。
"""
import os

import numpy as np

from ..core.backend import ASRBackend, BackendNotInstalledError

# FunASR 模型缓存指到项目内 .cache/modelscope（必须在 import funasr 前设置环境变量）
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(_PROJ, ".cache", "modelscope"))
os.environ.setdefault("MODELSCOPE_HUB_CACHE", os.path.join(_PROJ, ".cache", "modelscope"))

_VARIANTS = {
    # FunASR 别名（映射到 hub id）；offline 为同族整句离线版（准确率更高）
    "online":  dict(hub="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                    streaming=True),
    "offline": dict(hub="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    streaming=False),
}

# 流式 chunk 参数（仅 online）：encoder 5帧(50ms) / decoder 10帧(100ms)，look_back 带历史上下文
_CFG = dict(chunk_size=[5, 10, 5], encoder_chunk_look_back=4, decoder_chunk_look_back=1)


class ParaformerBackend(ASRBackend):
    name = "paraformer"
    sr = 16000
    supports_streaming = True          # 类默认；offline 变体在 __init__ 覆写为 False

    def __init__(self, device="auto", model_id=None, variant="online", hotword_file=None):
        if variant not in _VARIANTS:
            raise ValueError("未知 paraformer 变体: %r（可选: %s）"
                             % (variant, "/".join(sorted(_VARIANTS))))
        self._device = device
        self._variant = variant
        self._hub_id = model_id or _VARIANTS[variant]["hub"]
        self.supports_streaming = _VARIANTS[variant]["streaming"]   # 实例级：offline=False
        self._hotword_file = hotword_file
        self._hotword_matcher = None     # T16：拼音级热词纠错（load() 时编译）
        self._model = None
        self._cache = None
        self._partial_buf = ""       # 流式 delta 累积（FunASR 返回增量，须拼接成累计文本）

    def _local_model_dir(self):
        """优先本地缓存路径（离线零网络）；缺失返回 None 走 hub 下载。

        modelscope 缓存布局：`{MODELSCOPE_CACHE}/models/iic--<name>/snapshots/<rev>/`。
        直接喂 AutoModel 本地路径可**完全绕开 hub 文件清单请求**（实测把端点设成
        不可达地址仍加载成功；用 model_id 则每次启动都查 `/repo/files`）。
        **按当前变体的 hub 尾段精确匹配目录**（online/offline 两个变体目录共存时
        不能靠"paraformer in d"扫第一个——之前在线/离线只有一版不冲突，加 offline 后
        必须区分 `...vocab8404-online` vs `...vocab8404-pytorch`）。
        """
        tag = self._hub_id.rsplit("/", 1)[-1]   # 目录名以 iic--<tag> 结尾
        root = os.path.join(_PROJ, ".cache", "modelscope", "models")
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                if not d.endswith(tag):
                    continue                     # 只认本变体缓存，避免误扫另一变体
                snap = os.path.join(root, d, "snapshots")
                if os.path.isdir(snap):
                    for rev in sorted(os.listdir(snap)):
                        cand = os.path.join(snap, rev)
                        if os.path.isfile(os.path.join(cand, "model.pt")):
                            return cand
        return None

    def _correct(self, text):
        """T16 热词后纠错：拼音级模糊匹配，同音字（神庙→神妙）自动修正。

        空文本/未配热词 → 原样返回。对已纠正文本重复应用幂等（fuzzy 匹配
        segment==target 时跳过，显式替换 wrong→right 后 wrong 不再出现）。
        """
        if self._hotword_matcher is None or not text:
            return text
        updated, _ = self._hotword_matcher.apply_text(text)
        return updated

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
        # 本地缓存优先（离线零网络，满足硬指标"权重缓存后零网络请求"）；缺失才走 hub
        model = self._local_model_dir() or self._hub_id
        self._model = AutoModel(model=model, device=device, disable_update=True)
        # T16：编译热词纠错器（拼音级模糊匹配；依赖缺失/文件错误 → 告警降级，不影响识别）
        self._hotword_matcher = None
        if self._hotword_file:
            try:
                from funasr.utils.postprocess_hotwords import build_postprocess_hotword_matcher
                self._hotword_matcher = build_postprocess_hotword_matcher(
                    postprocess_hotword_file=self._hotword_file)
            except Exception as e:
                print("[paraformer] 热词文件加载失败（热词关闭，识别不受影响）: %s" % e, flush=True)
                self._hotword_matcher = None

    def recognize(self, audio):
        """整段（句子级）识别。online：is_final=True 末 chunk 不截断；offline：原生 generate。"""
        a = np.ascontiguousarray(audio, dtype=np.float32)
        if self._variant == "offline":
            res = self._model.generate(input=a)          # 离线整句：无 chunk CFG
        else:
            res = self._model.generate(input=a, is_final=True, **_CFG)
        text = (res[0]["text"] if isinstance(res, list)
                else res.get("text", str(res))).strip()
        return self._correct(text)      # T16 热词纠错（整句完整文本，同音字可命中）

    def recognize_stream(self, chunk, is_final=False):
        """流式增量（T13）：逐块出字 + 句末 flush 定稿。仅 online 变体。

        FunASR `generate` 流式返回**增量 delta**（非累计），内部拼进 `_partial_buf`，
        对外统一返回**累计**文本（部分/完整）。`is_final=True` 收尾返回完整句文本并清状态。
        offline 变体抛 NotImplementedError → 引擎 `supports_streaming=False` 自动走整句。
        """
        if self._variant == "offline":
            raise NotImplementedError("paraformer-offline 非流式（文件整句高精度专用）")
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
            return self._correct(final)  # T16：对完整累计文本纠错（跨块单词也能命中）
        # 部分文本也纠（对外统一出"已纠错"累计文本；幂等，无副作用）
        return self._correct(self._partial_buf)

    def reset(self):
        """清流式状态（新句子 / interrupt）。"""
        self._cache = None
        self._partial_buf = ""

    def close(self):
        self._model = None
        self._cache = None
        self._partial_buf = ""
