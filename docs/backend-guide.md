# voice1 后端接入指南（新增流式/非流式 ASR 后端）

> 本文教你把一个新的 ASR 模型接进 voice1 引擎（例如 SenseVoice、Fun-ASR-Nano、Qwen3-ASR、
> whisper-large 已如此接入）。读完后你只需**写一个文件**（`asr/<name>/backend.py`）+
> **改 3 处注册/预下载**，引擎的 VAD 断句、时序、打断词、热词纠错会自动生效——后端一概不用管。
>
> 前置阅读：`docs/engine-guide.md` §8（后端矩阵与契约语义）、§9（热词，后端无需支持）；
> 硬指标口径见 `docs/asr-architecture-decision.md`。

## 1. 架构总览：后端在整个链路里的位置

```
麦克风/文件 ──audio块(16k float32)──▶ 引擎 ingest()
                                        │  KWS 打断词旁路（引擎层，与后端无关）
                                        ▼
                                  有界队列 → worker 线程
                                        │  持 _state_lock + _recog_lock
                                        ▼
                              EnergyVAD 断句（引擎层，后端无关）
                                        │
                    ┌───────────────────┴───────────────────────┐
        streaming=True 且后端支持           否则
                    │                                          │
    recognize_stream(chunk, is_final)             recognize(整句audio)
    （逐块出字 + 句末 flush 定稿）                    （整句识别）
                    └───────────────────┬───────────────────────┘
                                        ▼
                              文本 ──▶ 引擎 _correct()（热词纠错）
                                        ▼
                             on_partial / on_sentence 回调
```

- **引擎只消费后端两个能力**：句子级 `recognize(audio)` 和流式 `recognize_stream(chunk, is_final)`。
- **VAD 断句、时序、打断、热词全是引擎的活**，后端不感知；后端只做"音频 → 文本"。
- 采样率约定：ASR 链路统一 **16kHz float32 1D numpy**（`sr` 类属性）。麦克风/文件在引擎里已重采样，
  后端拿到的永远是这个格式。

## 2. 后端契约（ASRBackend ABC，`asr/core/backend.py`）

| 成员 | 必须? | 语义 |
|---|---|---|
| `name = ""`（类属性） | 必须 | 注册名，须与 `_BACKEND_MODULES` 的键一致 |
| `sr = 16000`（类属性） | 必须 | 采样率；引擎会改用 `backend.sr`（若不同） |
| `supports_streaming = False`（类属性） | 必须 | 是否支持流式增量 |
| `load()` | **必须** | 惰性 import 推理依赖 + 建模型。失败抛 `BackendNotInstalledError`（带安装提示） |
| `recognize(audio) → str` | **必须** | **句子级**整段识别（VAD 已断好的完整句音频） |
| `recognize_stream(chunk, is_final=False) → str` | 流式才要 | 逐块增量，返回**累计**文本（见 §4 坑1） |
| `reset()` | 流式建议 | 清流式状态（新句子 / `interrupt()` 时引擎调用） |
| `close()` | **必须** | 释放模型/会话 |

**两类后端的差别只有一处**：

- **非流式**（如 whisper）：只实现 `recognize` + `close`；`recognize_stream` 保持抛
  `NotImplementedError`（引擎按 `supports_streaming=False` 自动走"积累块 + 整句 recognize"）。
- **流式**（如 paraformer/sherpa）：实现 `recognize_stream` + `reset`；`is_final=True` 收尾
  返回该句**最终完整文本**，之后流状态失效——**引擎随即会调 `reset()`**。

## 3. 三步接入清单

### 第 1 步：写 `asr/<name>/backend.py`

**骨架（非流式，直接可复制）**：

```python
# -*- coding: utf-8 -*-
"""MyBackend：一句话定位（GPU/CPU 高精度可选项，类比…）。"""
import os

import numpy as np

from ..core.backend import ASRBackend, BackendNotInstalledError

# 缓存落项目内 .cache/<name>（必须在 import 推理框架前 setdefault）
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("<FRAMEWORK_CACHE_ENV>", os.path.join(_PROJ, ".cache", "<name>"))


class MyBackend(ASRBackend):
    name = "mybackend"          # 注册名（须与 §3 第 2 步的键一致）
    sr = 16000                  # ASR 链路统一 16kHz
    supports_streaming = False  # 非流式

    def __init__(self, device="auto", **cfg):
        self._device = device
        self._model = None

    # -- 本地缓存优先（离线零网络，满足"权重缓存后零网络"）--------------
    def _local_model_dir(self):
        """命中返回缓存目录/路径；缺失返回 None → load() 走首次下载。"""
        root = os.path.join(_PROJ, ".cache", "<name>")
        ...  # 扫磁盘，找到已下载的模型返回其路径
        return None

    # -- ASRBackend 协议 ------------------------------------------------
    def load(self):
        try:
            from <framework> import <Model>       # 惰性 import：缺失才报错（不炸引擎）
        except ImportError:
            raise BackendNotInstalledError(
                "<framework> 未安装。请 `pip install ...` 后重试。")
        device = self._device
        if device in ("auto", None):
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._model = <Model>(model=self._local_model_dir() or "<repo_id>",
                              device=device)

    def recognize(self, audio):
        a = np.ascontiguousarray(audio, dtype=np.float32)
        ...  # 整段识别
        return text.strip()

    def recognize_stream(self, chunk, is_final=False):
        raise NotImplementedError(
            "<framework> 离线模型不支持流式；引擎回退积累块 + 整句 recognize")

    def reset(self):
        """离线模型无跨调用状态，可留空。"""
        pass

    def close(self):
        self._model = None
```

**流式版**：把 `supports_streaming = True`，并替换 `recognize_stream`/`reset` 为：

```python
    def __init__(self, device="auto", **cfg):
        ...
        self._partial_buf = ""      # 若框架返回 delta，须拼成累计文本（见坑1）
        self._state = None          # 框架流式状态（cache / stream）

    def recognize_stream(self, chunk, is_final=False):
        a = np.ascontiguousarray(chunk, dtype=np.float32)
        delta = self._model.<feed>(a, state=self._state, final=is_final)  # 框架 API
        self._partial_buf += delta       # 关键：对外必须返回累计文本，不是 delta
        if is_final:
            final = self._partial_buf
            self._partial_buf = ""       # 流结束：清状态（下句从零开始）
            self._state = None
            return final
        return self._partial_buf

    def reset(self):
        self._state = None               # 新句子 / interrupt 清状态
        self._partial_buf = ""
```

**纪律**（三个"必须"）：

1. **模块顶层 light import**：顶层只 import `numpy` 等轻量依赖；推理框架（funasr /
   faster_whisper / sherpa_onnx / torch）必须在 `load()` 内 import。这样依赖缺失只在
   `load()` 抛 `BackendNotInstalledError`，不会在 import 期炸掉整个引擎。
2. **离线缓存优先**：模块顶层 `setdefault` 环境变量指向项目内 `.cache/<name>`（必须在
   import 框架**之前**）；`_local_model_dir()` 本地优先，缺失才走网络——这是"权重缓存后
   零网络"硬指标的直接保证。
3. **`auto` 设备解析**：`device="auto"` → 用 `torch.cuda.is_available()` 定 cuda/cpu。

### 第 2 步：注册到 `asr/core/backend.py` 的 `_BACKEND_MODULES`

```python
_BACKEND_MODULES = {
    ...
    "mybackend": ("asr.mybackend.backend", "MyBackend"),
    # 带构造参数的注册（第三项是传给构造器的额外 cfg）：
    "mybackend-large": ("asr.mybackend.backend", "MyBackend", {"model_id": "large"}),
}
```

- 元组：`(模块路径, 类名, {额外cfg}?)`；`cfg` 会与 `device` 一起透传给构造器。
- `get_backend()` 惰性 `importlib` 模块；模块顶层任何异常都会统一转成
  `BackendNotInstalledError`（带安装提示）。未知名字 → `ValueError`（代码笔误）。
- 例：`paraformer-offline` 用 `{"variant": "offline"}`、`whisper-large` 用
  `{"model_id": "large-v3-turbo"}`——同一类不同档位，一行注册。

### 第 3 步：加入 `preload_asr.py` 的 `_MODELS` 列表

```python
_MODELS = ["paraformer", "paraformer-offline", "whisper", "whisper-large", "sherpa", "mybackend"]
```

`python preload_asr.py` 会逐个 `get_backend(...).load().close()` 把权重落到 `.cache/`，
保证运行期零网络。

## 4. 引擎会怎么用你的后端（读透，避免踩坑）

- **整句路径**（`streaming=False`，或后端 `supports_streaming=False`）：VAD 断好句 →
  `recognize(整句 audio)` → 结果 → `on_sentence` 回调。
- **流式路径**（`streaming=True` 且 `supports_streaming=True`）：
  - 每个未断句块 → `recognize_stream(chunk, is_final=False)` → 非空则 `on_partial` 出累计部分文本；
  - VAD 断句边界块 → `recognize_stream(chunk, is_final=True)` **flush 定稿** → 引擎**立即
    `reset()`** → 该文本作为最终句（`preset_text`，不再整句重识别）。
  - 这是项目"尾字延迟与句长无关"的关键路径（T13）。
- **异常降级**：流式调用抛异常 → 引擎置 `_stream_capable=False`，本会话剩余块自动走整句，
  不中断。别把异常吞掉不抛——抛出来让引擎降级是正确行为。
- **热词纠错自动生效**：引擎对每句最终文本/每个 partial 文本调 `_correct()`（拼音级，
  T16）——后端**不需要**任何热词支持，也**别**在框架的 generate 参数里透传热词表
  （流式 delta 片段内匹配不到目标词，且引擎层已做）。
- **锁纪律**：所有识别调用由引擎持 `_recog_lock` 串行化（模型非线程安全）——后端
  **不要**自建锁、不要假设单线程调用顺序；只需保证 `reset()` 后能继续用。
- **`interrupt()`**：引擎会调 `reset()` 清流式状态——`reset()` 必须真正清干净（cache /
  累积 buf / stream），否则下句会串上旧句残留。
- **文件同步**：`ingest_file()` 内部镜像 worker 的流式/整句路径，同一套契约。

## 5. 验收纪律（新增后端必须过）

1. **bench 全量**：`python bench/bench_asr.py --backend mybackend --device cuda --tag my`
   → CER（严格/规范）/ RTF / 平均 ttfb；CPU 再跑一趟。对照 `docs/engine-guide.md` §8 矩阵。
2. **内容级复核**：看 `reports/bench_my_*.txt` 的逐句 hyp vs ref——**别只看数字**，逐句过
   一遍（sherpa 短句空文本、whisper 数字/繁体形态这类问题只有逐句才看得出来）。
3. **CER 超标定位**：CER >5% 就文档标注"可选项/基线"（whisper 0.120、sherpa 0.190 都是这么
   处理的），如实写，不硬吹。
4. **流式后端额外**：`bench/bench_streaming.py`（首字/尾字延迟 + 流式定稿 CER）；
   用 `demonstrate_streaming.py` 冒烟"边说边出字"。
5. **文档同步**：README 后端矩阵、engine-guide §8、CLAUDE.md 关键坑（若有新坑）、preload 注释。
6. **回归**：确认既有后端不受影响（`get_backend` 惰性加载，新后端互不干扰）。

## 6. 已实现后端 = 现成模板

| 后端 | 模板角色 | 值得抄的地方 |
|---|---|---|
| `asr/paraformer/backend.py` | **流式模板** | FunASR cache 流式、**delta→累计**的坑（`_partial_buf`）、variant 档位机制、`is_final=True` 清 cache |
| `asr/sherpa/backend.py` | **流式模板** | sherpa-onnx `input_finished()` 收尾 + 结束后重建 stream |
| `asr/whisper/backend.py` | **非流式模板** | 本地 hf 缓存路径、`recognize_stream` 抛 NotImplementedError、float16/int8 compute_type |

新增后端优先以同名文件为起点复制改造，改完跑 §5 验收。

## 7. 常见坑速查

- **坑1 · delta ≠ 累计**：FunASR 流式 `generate` 返回**增量**（实测"明天早"→"上八点"→"开会"），
  不是累计假设。后端必须内部拼接，对外统一返回**累计**文本——否则引擎 `on_partial` 逐块出
  的是半截、跨块单词（"神/庙"分两次）热词也命中不到（引擎对累计文本匹配）。
- **坑2 · `is_final=True` 后流状态失效**：paraformer 清 cache、sherpa 重建 stream；引擎随后
  `reset()`，双保险。别假设 `is_final=True` 后还能继续喂。
- **坑3 · 模型非线程安全**：别在后端里开自己的线程/并行前向；引擎 `_recog_lock` 已串行。
- **坑4 · 本地缓存优先**：`AutoModel(model=model_id)` 即使权重已缓存也会发 hub 文件清单请求
  （离线失败）。必须 `_local_model_dir()` 命中本地路径直接喂（paraformer 后端有完整先例）。
- **坑5 · 环境变量在 import 框架前 setdefault**：缓存路径、镜像端点（HF_ENDPOINT /
  MODELSCOPE_CACHE）放模块顶层、import 框架之前。
- **坑6 · 别在 generate 透传热词**：热词是引擎层文本后处理（T16），后端无需、也不该支持。
- **坑7 · 抛 NotImplementedError 是对的**：非流式后端 `recognize_stream` 抛它 = 让引擎走
  整句路径，是契约的一部分，不是错误。
