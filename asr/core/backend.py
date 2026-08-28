# -*- coding: utf-8 -*-
"""后端抽象：ASRBackend 协议 + 惰性加载。

选择性安装的核心：`asr/core` 只在需要时 import 具体后端模块（paraformer/whisper/sherpa），
后端模块顶层不 import 任何推理依赖（numpy 等轻量依赖除外），模型与框架级 import 全部延迟到 `load()` 内
——缺失依赖时抛 `BackendNotInstalledError`（带安装提示），而不是在 import 期炸掉整个引擎。
"""
import importlib
from abc import ABC, abstractmethod

# 后端模块与类名约定：asr/<name>/backend.py 的 <Name>Backend。
# 元组第三项（可选）是传给构造器的额外 cfg（如 paraformer-offline 指定 variant）。
_BACKEND_MODULES = {
    "paraformer": ("asr.paraformer.backend", "ParaformerBackend"),
    # paraformer-large 离线版（文件整句高精度；非流式，实时流式仍用 paraformer=online）
    "paraformer-offline": ("asr.paraformer.backend", "ParaformerBackend", {"variant": "offline"}),
    "whisper": ("asr.whisper.backend", "WhisperBackend"),
    # faster-whisper large-v3-turbo（文件转写高精度，同音字/文学词纠错最强；GPU-only 非流式）
    "whisper-large": ("asr.whisper.backend", "WhisperBackend", {"model_id": "large-v3-turbo"}),
    "sherpa": ("asr.sherpa.backend", "SherpaBackend"),
}


class BackendNotInstalledError(RuntimeError):
    """对应后端依赖未安装 / 加载失败。携带安装提示。"""


class ASRBackend(ABC):
    """引擎消费的统一接口（抽象基类）。引擎只认这几个能力，其余差异全在后端内部。

    各后端继承本类，须实现 @abstractmethod（name/sr 为类属性）。
    未实现完的类无法实例化（ABC 提前报错），从"文档型协议"变为编译期契约。
    """

    name = ""
    sr = 16000                     # 采样率：ASR 链路统一 16kHz（麦克风/模型）
    supports_streaming = False     # 是否支持流式增量（recognize_stream）；whisper=False

    @abstractmethod
    def load(self):
        """惰性 import 模型并初始化。失败抛 BackendNotInstalledError。"""
        raise NotImplementedError

    def recognize(self, audio):
        """整段（句子级）识别，返回文本字符串。

        audio: 1D float32 numpy，16kHz。引擎按 VAD 断句后调用。
        """
        raise NotImplementedError

    def recognize_stream(self, chunk, is_final=False):
        """流式增量识别（T13 接入引擎，逐块出字 + 句末 flush 定稿）。

        - `is_final=False`（默认）：喂一块增量音频，返回**累计**部分文本（非增量 delta）。
        - `is_final=True`：结束当前流，返回该句**最终完整文本**；流状态随即失效，
          调用方随后 `reset()`（paraformer 清 cache、sherpa 重建 stream）。
        - 非流式后端（whisper）抛 NotImplementedError，引擎回退为"积累块 + 整句 recognize"。
        """
        raise NotImplementedError

    def reset(self):
        """清流式内部状态（新句子开始 / 会话被打断时调用）。默认无操作。"""
        pass

    @abstractmethod
    def close(self):
        """释放模型/会话。引擎 close() 时调用。"""
        raise NotImplementedError


def get_backend(name="paraformer", device="auto", **cfg):
    """按名字惰性加载后端类并构造实例（不 load——模型在 load() 才真正建）。

    - 未知名字 → ValueError（代码笔误，不是运行时问题）
    - 依赖缺失 / 加载失败 → BackendNotInstalledError（附安装提示）
    """
    n = (name or "paraformer").lower()
    if n not in _BACKEND_MODULES:
        raise ValueError("未知后端: %r（可选: %s）"
                         % (name, "/".join(sorted(_BACKEND_MODULES))))
    mod_name, cls_name, *extra = _BACKEND_MODULES[n]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:          # 后端模块顶层错误：统一转成带提示的异常
        raise BackendNotInstalledError(
            "后端 %r 模块加载失败（%s）。请按对应 README 安装依赖后重试。"
            % (name, e)) from e
    cls = getattr(mod, cls_name)
    return cls(device=device, **{**(extra[0] if extra else {}), **cfg})
