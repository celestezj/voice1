# voice1 — 离线实时中文语音识别（ASR）

后端无关的实时中文「音频 → 文字」引擎：麦克风/音频流喂入，VAD 断句 + 后端识别 + 逐句回调。
设计沿用 voice0（实时中文 TTS）的骨架方法论：后端抽象 + 可插拔实现 + 硬指标验收。

**硬指标（ADR 立项定案）**：RTF <0.3(GPU) / <1(CPU) · 尾字延迟 <0.5s · CER <5% · 离线（权重缓存后零网络）。

## 快速上手（从零复现）

```bash
# 1) 环境：克隆 voice0 的 voice-tts（含 torch），独立名 voice-asr
conda create -n voice-asr --clone voice-tts
conda activate voice-asr

# 2) 权重预下载（仅首次联网；之后运行期零网络）
python preload_asr.py

# 3) 跑验收（CER/RTF/尾字延迟，CPU/GPU 双测）
python bench/bench_asr.py --backend paraformer --device cuda --tag my

# 4) 跑起来（中文输出加 PYTHONIOENCODING=utf-8）
python examples/transcribe_file.py 音频.wav --device cuda   # 文件转写
python examples/record_mic.py --device cuda                  # 麦克风实时识别（sounddevice）
```

- 完整环境版本与镜像源：见 [`docs/environment-voice1.md`](docs/environment-voice1.md)。
- 全部设计决策与坑：见 [`docs/asr-architecture-decision.md`](docs/asr-architecture-decision.md)。

## 引擎设计（RealtimeASR）

- **单例 + 常驻 worker 线程**：模型只加载一次；音频块经有界队列（maxsize=8，识别慢则背压）交给 worker。
- **VAD 断句**：能量 VAD（`EnergyVAD`）状态机，静音尾长 `vad_silence_tail_ms` 判定句末（**默认 250ms**，T10 标定）。
- **后端抽象**：`ASRBackend` ABC（`load / recognize / recognize_stream / reset / close`）+ `get_backend(name)` 惰性加载；
  依赖缺失抛 `BackendNotInstalledError`，互不影响。
- **会话代际**：`interrupt()` 令旧音频块作废、重置 VAD/后端状态（换说话人/新会话）。
- **时序双轴**（`recog_axis`）：实时流用 wall 轴（ttfb 含 VAD 尾长 + 识别）；文件同步用 audio 轴（ttfb = 纯识别耗时，与 feed 加速无关）。

```python
asr = RealtimeASR(backend="paraformer", device="cuda", vad_silence_tail_ms=250)
asr.on_sentence(cb)             # cb(SentenceResult)：idx/text/audio_start/audio_end/recog_start/recog_end/ttfb
asr.ingest(chunk, ts)           # 实时流：非阻塞入队（背压时阻塞）
asr.ingest_file("x.wav")        # 文件：同步阻塞，返回 [SentenceResult, ...]
asr.interrupt()                 # 打断当前会话
asr.close()                     # 幂等；with / __del__ 兜底
```

## 后端矩阵（T9 验收，24 句语料 · VAD tail=250）

| 后端 | 设备 | 严格CER | RTF | 平均ttfb | 结论 |
|---|---|---|---|---|---|
| **paraformer** | **cuda** | **0.059** | **0.154** | **0.226s** | **默认主力**（实时尾字延迟 0.48s 达标） |
| paraformer | cpu | 0.059 | 0.26 | — | CER/RTF 达标，CPU 尾字延迟略超 |
| whisper (medium) | cuda | 0.120 | 0.139 | 0.370s | 高精度可选项（规范 CER 0.058；离线非流式） |
| whisper (medium) | cpu | 0.132 | 2.66 | — | CPU 不可用 |
| sherpa (14M) | cpu | 0.190 | 0.033 | 0.084s | 轻量基线（RTF 极优，质量不足） |

> CER 口径：严格 = 去标点；规范 = 去标点 + 繁简统一 + 中文/阿拉伯数字归一（whisper 的数字/繁体形态差异归因用）。
> VAD tail 权衡：250ms（实时延迟达标、CER 0.059）vs 600ms（CER 0.047 达标、延迟超标，离线转录用）。无单一值同时达标，详见 ADR T10。

## 目录结构

```
asr/
├── core/      后端无关骨架
│   ├── engine.py   RealtimeASR（单例/worker/队列/VAD/代际/lifecycle）
│   ├── jobs.py     SentenceResult（__slots__ 时序字段）
│   ├── backend.py  ASRBackend ABC + get_backend 惰性加载
│   └── audio.py    read_wav / resample_to / EnergyVAD
├── paraformer/  ParaformerBackend（默认主力，FunASR 流式 + cache 增量）
├── whisper/     WhisperBackend（可选，离线，本地缓存路径加载）
└── sherpa/      SherpaBackend（CPU 基线，sherpa-onnx zipformer）
bench/           bench_asr.py（--backend/--device/--tail/--tag）
preload_asr.py   权重预下载（一次性联网）
docs/            ADR / 环境版本锁定
assets/corpus/   CER 验收语料（24 句 + manifest.json）
```

## 已知限制

- **尾字延迟 vs CER 权衡**（T10）：无单一 VAD tail 同时达标；实时 250ms、离线 600ms。
- **paraformer 短句**：sherpa 基线对极短句有空文本/半句缺陷；paraformer/whisper 正常。
- **whisper**：离线非流式，不支持"边说边出字"（引擎回退积累块 + 整句识别）。
- **流式增量**：paraformer cache 模式已具备（60ms 粒度），引擎当前走"VAD 断句 + 整句识别"；逐帧实时输出是后续增强点。
