# CLAUDE.md — voice1 项目指南

离线实时中文语音识别（ASR，音频转文字）系统。核心指标：**RTF <0.3(GPU) / <1(CPU)**、
**尾字延迟 <0.5s**、**CER <5%**、**离线运行**（权重缓存后零网络请求）。

两个后端（**选择性安装**，互不影响）：
- **paraformer**（默认，实时主力）：FunASR / paraformer-zh-streaming，chunk 流式，边说边出字。
- **whisper**（可选项，高精度离线）：faster-whisper，通用中文精度上限，滑动窗口模拟流式作后备。
- 对照基线（非神经）：sherpa-onnx tiny / Vosk（探索后定，类比 voice0 的 `sapi/`）。

## 快速上手（安装 → 使用）

1. **环境**：先 `conda activate voice-asr`（从 `voice-tts` 克隆，含 torch 2.11+cu126）再跑 `python`；
   克隆后如遇版本冲突在 voice-asr 内单独重装，**不碰 voice0 的 voice-tts 环境**。
2. **权重预下载**（仅首次联网，之后运行期零网络）：`python preload_asr.py`。
3. **跑起来**（中文输出加 `PYTHONIOENCODING=utf-8`）：
   `python examples/transcribe_file.py 音频.wav` —— 文件转写；`python examples/record_mic.py` —— 麦克风实时识别。
4. **代码里用**：
   ```python
   from asr import RealtimeASR
   asr = RealtimeASR(backend="paraformer", device="cuda")  # 常驻识别，音频块喂入
   asr.on_sentence(lambda r: print(r.text, r.ttfb))        # 逐句结果回调
   asr.ingest(audio_chunk)                                 # 非阻塞入队，内部 VAD 断句 + 识别
   asr.ingest_file("x.wav")                                # 文件同步识别，返回 [SentenceResult]
   asr.close()                                             # 常驻单例，必须显式关
   ```
   > 详细 API 见 README「引擎设计」（T11 完善）。

## 环境硬性约束（新 Claude Code 接手必须先知道）

- **Python 必须用 `voice-asr` conda 环境**：先 `conda activate voice-asr` 再跑 `python`；base Python 3.12 无 torch。
- **跑带中文输出的命令加 `PYTHONIOENCODING=utf-8`**：Windows 默认 GBK 会直接崩。
- **权重/缓存重定向到项目内 `.cache/`**（`HF_HOME`/`HF_ENDPOINT`/`MODELSCOPE_CACHE` 在模块里已设好）；首次下载走 `HF_ENDPOINT=https://hf-mirror.com` 镜像（huggingface.co 直连被墙）。
- git 仓库根即 `E:\temp\voice1`（独立仓库，**不含 voice0 内容**；`third_party/` 是 gitignored 的上游 clone，别在里面提交）。

## 关键坑（非显而易见的，先看再动）

> 探索/实施阶段逐个补齐（voice0 教训前置：AEC 回声、VAD 参数、模型非线程安全、采样率 16k）。

- **模型非线程安全**：所有识别调用必须持 `_recog_lock`（写新后端/新调用路径时别绕过）。
- **VAD 是断句旋钮**：静音尾长 `vad_silence_tail_ms` 决定"这句说完"判定，是延迟-准确率权衡。**实测标定（T10）：默认 250ms**——实时尾字延迟达标、CER 0.059 逼近 5%；离线高精度用 600ms（CER 0.047 达标但延迟超标）。tail 小→句尾拖音幻听（"啊/嗯"等尾字）。无单一值同时达标，按场景选。
- **麦克风 16kHz / 模型 16kHz**：采样率与 voice0 TTS（44.1kHz）不同，两条链路各管各的。
- **回声/双讲（AEC）**：若与 voice0 组合成语音对话，麦克风会收到喇叭声音，需回声消除。

## 代码结构

```
asr/
├── core/      后端无关骨架
│   ├── engine.py   RealtimeASR（单例/常驻 worker 线程+有界队列/VAD 断句/lifecycle）
│   ├── jobs.py     SentenceResult（idx/text/audio_start/audio_end/recog_start/recog_end/ttfb）
│   ├── backend.py  ASRBackend（ABC：load/recognize/recognize_stream/reset/close）+ get_backend(name) 惰性加载
│   └── audio.py    音频工具（read_wav/resample_to/EnergyVAD 断句状态机）
├── paraformer/  ParaformerBackend（默认主力，FunASR 流式，cache 模式可增量）
├── whisper/     WhisperBackend（可选高精度，离线非流式，本地缓存路径加载）
└── sherpa/      SherpaBackend（CPU 轻量基线，sherpa-onnx zipformer）
bench/           bench_asr.py（CER/RTF/延迟，--tail 可调 VAD 尾长）
docs/            asr-architecture-decision.md（ADR，选型/标定/环境决策）
assets/          验收语料（CER 裁判集）
reports/         bench 报告（gitignored）
```

## 权威文档（动手前先读对应章节）

- `docs/asr-architecture-decision.md` = **选型结论与硬指标**（ADR，从立项第一天写起）。
- `README.md`「引擎设计」（T6 后落地）= RealtimeASR 完整设计。
- `docs/ai-project-methodology.md`（在 voice0 仓库） = 本项目沿用并沉淀的 **AI 项目全流程方法论**，可复用。

## 协作习惯（本项目）

- 与用户用**中文**交流。
- 里程碑式改动后用户会说「commit吧」——按其节奏提交，commit message 用中文。
- **验收纪律**：每个后端/改动必须跑 CER 内容级复核，不要只报 RTF/延迟数字。
