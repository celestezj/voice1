# voice1 — 离线实时中文语音识别（ASR）

后端无关的实时中文「音频 → 文字」引擎：麦克风/音频流喂入，VAD 断句 + 后端识别 + 逐句回调。
设计沿用 voice0（实时中文 TTS）的骨架方法论：后端抽象 + 可插拔实现 + 硬指标验收。

**硬指标（ADR 立项定案）**：RTF <0.3(GPU) / <1(CPU) · 尾字延迟 <0.5s · CER <5% · 离线（权重缓存后零网络）。

## 快速上手（从零复现）

> **依赖前置**：voice1 的语音对话程序（`examples/voice_dialogue.py`）在同一进程里同时跑
> voice1 的 ASR 和 **voice0** 的 TTS。voice0 约定放在 voice1 的**同级目录**（安装脚本会自动
> 克隆到那里）——环境装的是 **voice0/voice1 共用的唯一环境 `voice-asr`**（TTS 底座复用
> voice0 的 `voice-tts` 克隆而来）。

```bash
# 0) 前置：安装 Miniconda/Anaconda + Git for Windows（git 需在 PATH）

# 1) 一键安装（自动检测显卡→建 voice-tts 底座→克隆出共享环境 voice-asr→
#    装 ASR 依赖→预下载权重→端到端验证；重复执行自动跳过已完成步骤）
python setup_env.py

# 2) 用共享环境（voice0 / voice1 都用它）
conda activate voice-asr

# 3) 跑起来（中文输出加 PYTHONIOENCODING=utf-8）
python examples/transcribe_file.py 音频.wav --device cuda [--streaming]   # 文件转写（wav/mp3/flac/ogg；--streaming 流式逐块出字）
python examples/record_mic.py --device cuda                  # 麦克风实时识别（无麦克风自动检测提示；--input-device 可指定设备；--streaming 流式逐块出字）
python bench/bench_asr.py --backend paraformer --device cuda --tag my     # 验收（CER/RTF/尾字延迟）
```

- 完整环境版本与镜像源：见 [`docs/environment-voice1.md`](docs/environment-voice1.md)。
- 全部设计决策与坑：见 [`docs/asr-architecture-decision.md`](docs/asr-architecture-decision.md)。
- **语音对话快速开始**（ASR + DeepSeek LLM + voice0 TTS 全链路，GPU）：
  `PYTHONIOENCODING=utf-8 python examples/voice_dialogue.py --asr-device cuda --tts-device cuda --vad-tail 300`
- 语音对话：快速开始 + 参数白话解释 + 架构时序图：见 [`docs/voice-dialogue.md`](docs/voice-dialogue.md)。

## 引擎设计（RealtimeASR）

> **使用与原理详解**（线程模型 / 每个 API 的参数与返回值 / SentenceResult 字段 /
> wall 与 audio 时间轴 / VAD 原理 / 后端对比）：见 [`docs/engine-guide.md`](docs/engine-guide.md)。
> 下面代码案例是"一分钟上手"，逐行含义见上面的指南。

- **单例 + 常驻 worker 线程**：模型只加载一次；音频块经有界队列（maxsize=8，识别慢则背压）交给 worker。
- **VAD 断句**：能量 VAD（`EnergyVAD`）状态机，静音尾长 `vad_silence_tail_ms` 判定句末（**默认 250ms**，T10 标定）。
- **后端抽象**：`ASRBackend` ABC（`load / recognize / recognize_stream / reset / close`）+ `get_backend(name)` 惰性加载；
  依赖缺失抛 `BackendNotInstalledError`，互不影响。
- **会话代际**：`interrupt()` 令旧音频块作废、重置 VAD/后端状态（换说话人/新会话）。
- **打断词旁路（T12）**：`interrupt_words=["停下"]` 启用轻量 KWS（sherpa-onnx zipformer 3.3M int8，
  独立于主 ASR，毫秒级）。每个音频块**流式 `feed()`** 检测——用户说「停下」立即命中 → `interrupt()`
  即时作废全部排队任务，触发块丢弃不识别（见下方代码 case）。**不支持自动抢占**（连说多句会正常
  排队识别）；打断词若走普通队列会排在队尾，等它被识别前面任务早完成——打断悖论，故必须旁路。
- **时序双轴**（`recog_axis`）：实时流用 wall 轴（ttfb 含 VAD 尾长 + 识别）；文件同步用 audio 轴（ttfb = 纯识别耗时，与 feed 加速无关）。

```python
from asr import RealtimeASR
import sounddevice as sd, time

# 1) 创建引擎（单例：模型只加载一次；backend/device/vad_tail/interrupt_words 变更才重建）
#    paraformer(默认)/whisper/sherpa；device auto|cpu|cuda；interrupt_words 非空启用打断词旁路
asr = RealtimeASR(backend="paraformer", device="cuda",
                  vad_silence_tail_ms=250,          # VAD 静音尾长：>它判定"这句话说完了"
                  interrupt_words=["停下"])         # 可选：说「停下」→ 作废全部排队任务

# 2) 结果回调：worker 每断出一句、识别完就调用一次（异步，句子粒度）
def cb(r):                                          # r 是 SentenceResult
    print("#%d [%.2f~%.2fs] %s  尾字延迟=%.3fs  stale=%s"
          % (r.idx, r.audio_start, r.audio_end, r.text, r.ttfb, r.stale))
asr.on_sentence(cb)

# 3) 实时流：每批麦克风采样 = 一个 chunk（非阻塞入队；识别慢时阻塞=背压）
def mic_cb(indata, frames, t, status):
    asr.ingest(indata[:, 0], source_ts=time.monotonic())
with sd.InputStream(samplerate=16000, channels=1, callback=mic_cb):
    time.sleep(30)                                   # 说 30 秒

# 4) 文件：同步阻塞，内部切成 100ms 子块 → VAD 断句 → 每句识别，返回全部结果
results = asr.ingest_file("会议录音.wav")

asr.interrupt()                                      # 手动打断（作废排队任务）
asr.close()                                          # 幂等；with / __del__ 兜底
```

**流式逐帧出字**（T13，`streaming=True`）：边说边出字 + 句末 flush 定稿，首字延迟与句长无关。

```python
# 1) 创建流式引擎（streaming 变更会销毁重建单例）
asr = RealtimeASR(backend="paraformer", device="cuda", streaming=True)
asr.on_partial(lambda p: print("边说边出:", p.text))    # 每块回调累计部分文本（首字 ≈0.9s）
asr.on_sentence(lambda r: print("句末定稿:", r.text))   # VAD 断句后 flush 完整句（尾字 ≈0.35s）
#   后端不支持流式（whisper）→ 告警自动降级整句，不中断
#   完整可运行对比（流式 vs 整句首字/尾字延迟）：
#   PYTHONIOENCODING=utf-8 python examples/demonstrate_streaming.py 句1.wav 句2.wav ...
```

**热词纠错**（T16，`hotword_file`，**所有后端统一生效**）：引擎级对识别文本做**拼音级纠错**，
修同音字（神庙→神妙、心天→新天、四十→四时——纯音频歧义，换更大模型也解不了，实测
whisper-large 反而更差）。显式映射 `神庙=>神妙` 零误伤；模糊目标行兜底未知变体。
流式/整句、paraformer/whisper/sherpa 全部生效（**无需后端支持热词**——它是引擎层文本后处理）。

```python
# 创建引擎时给热词文件；每行一个纠错项（见 assets/hotwords/xiaoshuang.txt 注释头）
asr = RealtimeASR(backend="paraformer", device="cuda",
                  hotword_file="assets/hotwords/xiaoshuang.txt")
# 命令行：transcribe_file.py 音频.mp3 --hotword-file assets/hotwords/xiaoshuang.txt
#        record_mic.py --hotword-file assets/hotwords/xiaoshuang.txt
```

**`on_partial` vs `on_sentence`（一句话内两个回调）**：

| | `on_partial`（流式，`streaming=True`） | `on_sentence`（通用） |
|---|---|---|
| 次数 | 每句多次（每个非空增量块一次） | 每句一次 |
| 内容 | 累计**部分**文本（边说边出字） | **完整句**文本（定稿） |
| 时机 | 说话期间逐块，首字延迟 ≈0.9s | 句末 VAD 断句 + flush/识别完成后 |
| 可靠性 | 会随上下文修正，只供展示 | 权威最终结果（stale 除外） |
| 回调参数 | `PartialResult`（text/audio_start/wall_ts） | `SentenceResult`（…/ttfb/stale） |

- 一句话内先有多次 `on_partial`（实时滚屏），句末以一次 `on_sentence` 收尾（落库）——**最终文本以 `on_sentence` 为准**。
- `on_sentence` 永远触发（整句/流式/降级都会）；`on_partial` 仅流式且后端支持时触发。
- 打断（`interrupt()`/「停下」）：`on_sentence` 判 `stale=True` 不触发；`on_partial` 被 `_gen` 守卫作废，回调里 `audio_start` 会跳变——**上层别缓存 partial 累计文本跨句用**。

> **chunk ≠ 识别任务**：喂入的块先由 VAD 累积，能量走到"句末"才断成一句，**一句**才触发
> 一次识别/一次回调。停顿超过 `vad_silence_tail_ms` 即断句。
>
> **打断语义**：正在识别的句子无法中止（整句前向原子），完成后判 `stale=True` **不进普通
> 回调**（`SentenceResult.stale`，`profile=True` 时 `asr._sentences` 仍收集）；已排队未处理的
> 句子全部作废；打断词本身不被识别。完整设计论证与 KWS 坑见 ADR T12；可运行示例见
> `examples/demonstrate_interrupt.py`。

## 后端矩阵（T9 验收，24 句语料 · VAD tail=250）

| 后端 | 设备 | 严格CER | RTF | 平均ttfb | 结论 |
|---|---|---|---|---|---|
| **paraformer** | **cuda** | **0.059** | **0.154** | **0.226s** | **默认主力**（实时尾字延迟 0.48s 达标） |
| paraformer | cpu | 0.059 | 0.26 | — | CER/RTF 达标，CPU 尾字延迟略超 |
| whisper (medium) | cuda | 0.120 | 0.139 | 0.370s | 高精度可选项（规范 CER 0.058；离线非流式） |
| whisper (medium) | cpu | 0.132 | 2.66 | — | CPU 不可用 |
| whisper-large (v3-turbo) | cuda | 0.141 | 0.112 | 0.266s | **不建议**（T16 实测：同音字不修、CER 最差，见已知限制） |
| sherpa (14M) | cpu | 0.190 | 0.033 | 0.084s | 轻量基线（RTF 极优，质量不足） |

> CER 口径：严格 = 去标点；规范 = 去标点 + 繁简统一 + 中文/阿拉伯数字归一（whisper 的数字/繁体形态差异归因用）。
> VAD tail 权衡：250ms（实时延迟达标、CER 0.059）vs 600ms（CER 0.047 达标、延迟超标，离线转录用）。无单一值同时达标，详见 ADR T10。

**流式 vs 整句出字延迟**（T13，`streaming=True`，实时节奏喂入同一语料）：

| 设备 | 首字（流式） | 首字（整句） | 尾字（流式） | 尾字（整句） | 流式CER |
|---|---|---|---|---|---|
| cuda | **0.932s** | 3.110s | **0.348s**（max 0.486 达标） | 0.626s（max 1.022 破线） | **0.017** |
| cpu | **0.984s** | 3.405s | **0.416s**（max 0.580） | 0.905s（max 1.842） | **0.017** |

流式逐块出字（`on_partial`）+ 句末 flush 定稿，首字延迟与句长无关；整句首字=整句话说完才出字。

## 目录结构

```
asr/
├── core/      后端无关骨架
│   ├── engine.py   RealtimeASR（单例/worker/队列/VAD/代际/lifecycle/打断词旁路）
│   ├── jobs.py     SentenceResult（__slots__ 时序字段，含 stale）
│   ├── backend.py  ASRBackend ABC + get_backend 惰性加载
│   └── audio.py    read_audio（wav/mp3/flac/ogg…）/ resample_to / EnergyVAD
├── kws/        打断词旁路（T12）
│   ├── interrupt.py   InterruptDetector ABC + get_interrupt_detector 惰性加载
│   └── sherpa.py      SherpaKwsDetector（zipformer 3.3M int8，拼音建模单元）
├── paraformer/  ParaformerBackend（默认主力，FunASR 流式 + cache 增量）
├── whisper/     WhisperBackend（可选，离线，本地缓存路径加载）
└── sherpa/      SherpaBackend（CPU 基线，sherpa-onnx zipformer）
bench/           bench_asr.py（整句 CER/RTF）+ bench_streaming.py（流式 vs 整句出字延迟）
examples/        transcribe_file / record_mic / demonstrate_interrupt / demonstrate_streaming（代码 case）
preload_asr.py   权重预下载（一次性联网）
docs/            ADR（选型/标定）/ engine-guide（引擎使用与原理）/ backend-guide（新增后端接入）/ 环境版本锁定
assets/corpus/   CER 验收语料（24 句 + manifest.json）
assets/hotwords/ 热词文件示例（T16：xiaoshuang.txt 同音字纠错）
```

## 已知限制

- **尾字延迟 vs CER 权衡**（T10）：无单一 VAD tail 同时达标；实时 250ms、离线 600ms。
- **paraformer 短句**：sherpa 基线对极短句有空文本/半句缺陷；paraformer/whisper 正常。
- **whisper**：离线非流式，不支持"边说边出字"（`streaming=True` 会告警并降级为整句识别）。
- **whisper-large（v3-turbo）不建议用**（T16 实测）：语料严格 CER 0.141 是全部后端最差（比 medium 0.120 还差），且同音字（神庙/新天/四十）**不修复**——同音字是纯音频歧义，非模型大小问题。已接入但**默认不推荐**，仅作扩展接口存在。同音字正确解法见下条热词。
- **同音字正确解法是热词纠错（T16）**：`--hotword-file` 对识别文本做**引擎级**拼音级纠错（神庙→神妙），**所有后端**（paraformer/whisper/sherpa）流式/整句统一生效，实测锚定句 100% 修复。**注意**：热词文件里的"模糊目标行"（单独一个词）对 2 字词可能吞相邻同音字，精度要求高时用"显式映射 `错误词=>正确词`"。详见 engine-guide §9。
- **流式已实现**（T13）：`streaming=True` 逐块出字 + 句末 flush 定稿（见上表）；缺省 `streaming=False` 仍是整句识别。流式定稿 CER 0.017 优于整句 0.053，无 CER 代价。**文件同步也走流式**（`ingest_file` 随引擎模式；命令行 `transcribe_file.py --streaming`）。
- **安静短句在流式路径可能漏断句**：VAD 能量门限 -35dB/最短句 250ms，过静音短句（如 corpus s01 原始电平）会被当噪声丢弃 → 与下一句合并（整句路径靠文件末 flush 兜底，流式无文件边界）。demo/bench 已做"只放大安静文件到 peak 0.10"的响度归一；真实麦克风电平通常达标。
- **打断词命中需尾随音频**：流式 KWS 要 ~0.2-0.4s 尾随音频才能收尾解码（真实麦克风持续采样天然满足）；若音频流在「停下」后立即结束，命中延迟到后续音频到达。
- **KWS 对喂入响度敏感**：过静音的音频（peak~0.04、低 SNR）若被放大到 peak≥0.15，噪声底抬高会导致「停下」漏检；VAD 又会把过静音句子当噪声丢弃。引擎不自动归一（麦克风电平通常达标）；demo 归一至 peak 0.10 为双检公共区间（见 ADR T12）。
- **打断词依赖 sherpa-onnx + pypinyin**：`interrupt_words` 非空但两者缺失时，打断旁路降级为"无打断"（仅告警，不影响主识别）。
