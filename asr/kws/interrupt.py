# -*- coding: utf-8 -*-
"""打断词检测器抽象（InterruptDetector）+ 惰性加载。

与主后端同构（参照 asr/core/backend.py 的 ASRBackend/get_backend）：
- ABC 定义统一能力（load/detect/reset/close），引擎只认这几个。
- `get_interrupt_detector(name, words)` 惰性加载，依赖缺失抛
  `BackendNotInstalledError`（引擎捕获后**降级为"无打断"**，不阻塞主识别）。
- 模块顶层不 import 任何推理依赖（仅 numpy 级轻量依赖除外），
  框架级 import 全部延迟到 `load()` 内。

设计背景（T12，详见 ADR）："停止/停下"这类打断指令**不能**走普通识别队列
（否则它排在队尾，等它被识别时前面的任务早已完成，打断无意义——悖论）。
正确做法是旁路 KWS：VAD 断句出的句子先经轻量关键词检测（毫秒级），命中
打断词 → 引擎 `interrupt()` 即时作废全部排队任务，该句丢弃不完整识别。
"""
import importlib
from abc import ABC, abstractmethod

from ..core.backend import BackendNotInstalledError

# 检测器模块与类名约定：asr/kws/<name>.py 的 <Name>Detector
_DETECTOR_MODULES = {
    "sherpa": ("asr.kws.sherpa", "SherpaKwsDetector"),
}


class InterruptDetector(ABC):
    """打断词检测器：VAD 断句后的一个句子 → 是否命中打断关键词（如"停下"）。"""

    name = ""
    sr = 16000                     # 采样率：与 ASR 链路统一 16kHz

    @abstractmethod
    def load(self):
        """惰性 import 模型并初始化。失败抛 BackendNotInstalledError。"""
        raise NotImplementedError

    def feed(self, chunk):
        """**流式旁路检测（主路径）**：喂入一个音频块（16k float32）。

        命中任一打断关键词返回 True（内部自动 reset 流状态）。
        未实现的后端返回 False——引擎回退为 VAD 断句后整句 `detect`（兜底，
        但那样无法提前作废"打断词之前"的排队任务）。

        为什么必须在 ingest 旁路流式检测：打断词若也走 worker 队列，它排在
        队尾，等它被处理时前面的任务早已完成——打断悖论（T12 论证）。
        """
        return False

    def detect(self, audio):
        """整段检测（兜底）：audio 是 VAD 断句出的一个句子（16k float32）。

        返回 True = 命中任一打断关键词。引擎据此**丢弃该句（不识别）**并 interrupt()。
        """
        raise NotImplementedError

    def reset(self):
        """清检测状态（会话打断/切换时）。默认无操作。"""
        pass

    @abstractmethod
    def close(self):
        """释放模型。引擎 close() 时调用。"""
        raise NotImplementedError


def get_interrupt_detector(name="sherpa", words=None, **cfg):
    """按名字惰性加载检测器类并构造实例（不 load——模型在 load() 才建）。

    - 未知名字 → ValueError（代码笔误）
    - 依赖缺失 / 加载失败 → BackendNotInstalledError（引擎降级为无打断）
    """
    n = (name or "sherpa").lower()
    if n not in _DETECTOR_MODULES:
        raise ValueError("未知打断词检测器: %r（可选: %s）"
                         % (name, "/".join(sorted(_DETECTOR_MODULES))))
    mod_name, cls_name = _DETECTOR_MODULES[n]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        raise BackendNotInstalledError(
            "打断词检测器 %r 模块加载失败（%s）。" % (name, e)) from e
    cls = getattr(mod, cls_name)
    return cls(words=words, **cfg)
