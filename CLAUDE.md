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
   asr.on_sentence(lambda r: print(r.text, r.ttfb))        # 逐句结果回调（r.stale 可判打断残留）
   asr.ingest(audio_chunk)                                 # 非阻塞入队，内部 VAD 断句 + 识别
   asr.ingest_file("x.wav")                                # 文件同步识别，返回 [SentenceResult]
   asr.close()                                             # 常驻单例，必须显式关
   # 打断词（T12，可选）：interrupt_words=["停下"] → 用户说「停下」即作废全部排队任务
   asr = RealtimeASR(backend="paraformer", device="cuda", interrupt_words=["停下"])
   ```
   > 详细 API 见 README「引擎设计」（T11/T12 完善）；打断词完整设计见 ADR T12、代码 case 见 `examples/demonstrate_interrupt.py`。

## 环境硬性约束（新 Claude Code 接手必须先知道）

- **Python 必须用 `voice-asr` conda 环境**：先 `conda activate voice-asr` 再跑 `python`；base Python 3.12 无 torch。
- **跑带中文输出的命令加 `PYTHONIOENCODING=utf-8`**：Windows 默认 GBK 会直接崩。
- **权重/缓存重定向到项目内 `.cache/`**（`HF_HOME`/`HF_ENDPOINT`/`MODELSCOPE_CACHE` 在模块里已设好）；首次下载走 `HF_ENDPOINT=https://hf-mirror.com` 镜像（huggingface.co 直连被墙）。
- git 仓库根即 `E:\temp\voice1`（独立仓库，**不含 voice0 内容**；`third_party/` 是 gitignored 的上游 clone，别在里面提交）。
- **`voice-asr` 也是 voice0(TTS) 的运行环境（两项目共享，实测 2026-08-28）**：voice-asr 当初从 voice-tts 克隆，是 voice-tts 的**严格超集**——voice0 的 melo 后端（`from melo.api import TTS`，导入名是 `melo` 不是 `MeloTTS`）在 voice-asr 可直接跑（合成冒烟通过）。同进程组合示例见 `examples/use_with_voice0_tts.py`（TTS→ASR→TTS 闭环）。**组合时 HF_HOME 是唯一可能冲突的环境变量**：melo 用 voice0/.cache/hf，voice1 仅 whisper 后端才用 HF_HOME（默认 paraformer 走 MODELSCOPE_CACHE，无冲突）——显式播种见该示例。**cosy 后端不能与 melo/ASR 同进程**（voice0 设计约束：cosy 注入 transformers 4.51.3，与主环境 4.57.6 冲突，须独立子进程）。

## 关键坑（非显而易见的，先看再动）

> 探索/实施阶段逐个补齐（voice0 教训前置：AEC 回声、VAD 参数、模型非线程安全、采样率 16k）。

- **模型非线程安全**：所有识别调用必须持 `_recog_lock`（写新后端/新调用路径时别绕过）。KWS 检测器同理：ingest 旁路 `feed()`（主线程）与 worker 兜底 `detect()`（worker 线程）可能并发，`SherpaKwsDetector` 内部有 `_lock` 串行。
- **打断词旁路必须在 ingest 流式 `feed()`，不能走 worker 队列**（T12）：打断词若排进普通队列，它在队尾，等 worker 处理时前面任务早完成——打断悖论。`interrupt()` 拆两步：`_gen += 1` 即时（GIL 原子）+ `_state_lock` 内清 VAD/队列。
- **stale 的 task_gen 必须用「出队块的 gen」**（T12c 实测竞态）：`_process_sentence_locked(..., task_gen=出队gen)`。若用处理时的当前 `_gen`，interrupt 的 `_gen+=1` 恰在 worker 处理该块中途发生时，块内 VAD 已含的打断词音频会以新代际漏入管线（paraformer 把「停下」误识别为"影响下"）。
- **KWS 建模单元是拼音**：keywords 文件写 `t íng x ià @停下`，汉字→音节串用 pypinyin `to_initials/to_finals_tone(strict=False)` 自建转换（组合声母/带调韵母不拆）；**不能用 `text2token`**（会拆 `sh`→`s h`）。命中需 ~0.2-0.4s 尾随音频收尾解码（麦克风天然满足）。
- **KWS 对喂入响度敏感（T12d）**：静音文件放大到 peak≥0.15 会漏检「停下」（噪声底抬高）；VAD（门限 -35dB）又会丢过静音句。引擎不归一；demo 归一至 peak 0.10 为双检公共区间。
- **FunASR 每次启动查 hub 文件清单，须本地路径加载（T15）**：`AutoModel(model=model_id)`
  即使权重已缓存也发 `/api/v1/models/.../repo/files` 核对清单（日志见 "Downloading 11 files"），
  离线时重试失败——违反"权重缓存后零网络"。paraformer 后端 `_local_model_dir()` 扫
  `.cache/modelscope/models/*/snapshots/*/model.pt`，命中直接喂**本地路径**（实测端点
  设成不可达地址仍加载成功）。写新后端/改加载路径时别丢"本地优先"。
- **FunASR 流式 cache 返回 DELTA 非累计（T13）**：`recognize_stream(chunk, is_final=False)` 返回的是本块**新增**片段（"明天早"→"上八点"→"开会"），不是累计文本。后端必须内部累加（`_partial_buf`），`is_final=True` 返回累加结果并清 cache。sherpa `get_result()` 本身累计。写新后端时别把 delta 当累计回调给上层。
- **流式 flush 持锁边界（T13）**：worker 流式路径在 `_state_lock` 内、持 `_recog_lock` 调 `recognize_stream(is_final=True)` 完成定稿，随后 `_process_sentence_locked(..., preset_text=text)` **跳过整句 recognize**——避免 `_recog_lock` 重入死锁。改这段时别让 flush 与整句识别抢锁。
- **preset 路径 ttfb 须补 flush 耗时（T15）**：`preset_text` 路径下 t2-t1 只剩微秒（识别已提前在 `_stream_finalize` 完成），audio 轴 ttfb 会虚报 0——`_stream_finalize` 实测 flush 耗时经 `preset_dur` 传回，audio 轴 `ttfb=(t2-t1)+preset_dur`（wall 轴不动）。
- **流式文件末残句别重喂 `sent`（T15）**：`vad.flush()` 返回的残句音频**早已逐块喂过 partial**（cache 已含整句），收尾时**不能再喂 `sent`**——会 double-feed：文本重复/静音幻听（实测 mp3 复现「内心反映…内心反映…」）。修法：喂 100ms 静音块触发 `is_final=True` 取回累计文本（等价实时流句末边界块）；cache 为空（残句只是纯尾静音）返回 '' → 跳过。
- **安静文件流式漏断句 + 归一化只放大不缩小（T13）**：VAD 门限 -35dB/最短句 250ms，过静音短句（corpus s01/s04 原始 RMS<-38dB，仅 2~5 帧过阈）被当噪声丢弃 → 流式路径（无文件末 flush 兜底）句子永不闭合、与下一句合并。demo/bench 对语料**只放大** peak<0.10 的文件到 0.10（响亮文件保持原电平）——**别统一压到 0.10**，否则 RMS 在 -34dB 附近的文件（s03/s06/s07）跌破门限同样漏断句，单文件卡 ~19s 拖垮整趟 bench（整句 CER 0.059→0.108）。
- **bench 延迟趟必须连续喂入（T13）**：paced 趟别"喂一文件等一文件"——等 final 的 wait 期间 VAD 时间线（`_ts_cur`）不推进、与真实墙钟脱节 → `audio_end` 被低估 → ttfb 虚高且逐文件累积（实测 24 文件后虚高到 1.8s）。连续喂入（文件间靠尾静音停顿）+ 末尾统一 `_wait_idle` 排空，ttfb 才真实。
- **同音字别靠换模型，用热词纠错（T16）**：神庙/神妙 等拼音相同，纯声学不可分——whisper-large 实测语料 CER 0.141 最差且不修复。正解是 FunASR `postprocess_hotword_file`（拼音级模糊匹配）。
- **热词纠错放在引擎层，不在各后端（T16）**：文本级后处理与后端无关——`RealtimeASR._correct()` 对每句最终文本/流式 partial 统一应用（后端不支持也无需支持，全后端生效）。流式跨块单词（"神/庙"分两次）因后端返回**累计**文本同样可命中；别在 generate 时透传 postprocess_hotword_file（delta 片段内匹配不到目标词）。
- **热词文件优先显式映射（T16）**：模糊目标行（单独一个词）对 2 字词会吞相邻同音字——"的神妙"（相似度 0.94）整窗替换删掉"的"、"必减少于"被"减少于"命中删掉"必"。精度要求高用 `错误词=>正确词` 显式映射（确定性零误伤），模糊目标仅兜底未知变体。
- **FunASR generate 默认打 tqdm 进度条刷屏（rtf_avg: ...，T17d）**：`AutoModel.generate` 默认 `disable_pbar=False`，流式逐块刷屏（实时运行最烦人）。paraformer 后端全部 generate 调用传 `disable_pbar=not debug`，引擎把 `debug` 透传后端构造器——默认静默，`--debug` 才显示。**写新后端默认静默框架输出**，别让 tqdm/INFO 日志刷屏。
- **VAD 是断句旋钮**：静音尾长 `vad_silence_tail_ms` 决定"这句说完"判定，是延迟-准确率权衡。**实测标定（T10）：默认 250ms**——实时尾字延迟达标、CER 0.059 逼近 5%；离线高精度用 600ms（CER 0.047 达标但延迟超标）。tail 小→句尾拖音幻听（"啊/嗯"等尾字）。无单一值同时达标，按场景选。
- **麦克风电平够不着 VAD 门限 → "说话没反应"（T17c 实测）**：VAD 断句门限 -35dB，但不少麦克风说话 RMS 只有 -36~-46dB（本机 HD Audio 麦实测 6s 仅 24/300 帧过阈）——**录音正常、识别全无**。`check_mic_signal` 只拦"全哑（<-80dB）"拦不住这个。解法：record_mic 用 `MicAGC` 采集层自适应放大（目标 peak 0.3、只放大不压小、上限 8x；底噪放满仍 ~-50dB 不会误断句）。**引擎层故意不归一（T12d），mic 层负责**。排查 mic 无反应先跑 `tmp/probe_mic.py` 看电平与过阈帧数。
- **麦克风 16kHz / 模型 16kHz**：采样率与 voice0 TTS（44.1kHz）不同，两条链路各管各的。
- **回声/双讲（AEC）**：若与 voice0 组合成语音对话，麦克风会收到喇叭声音，需回声消除。
- **MeloTTS-Chinese 被切成 Xet 存储 → huggingface_hub 绕开缓存重下 208M（2026-08-29 实测）**：仓库启用 Xet 后，新版 huggingface_hub 把 xet 仓库当"未缓存"，即使权重完整躺在 voice0/.cache/hf 也重新下载 config.json+checkpoint.pth（hf-mirror ~70kB/s，卡 46 分钟）。修复：组合程序 import 前设 `HF_HOME=voice0/.cache/hf` **且** `HF_HUB_DISABLE_XET=1`（实测 0.55s 命中缓存零下载）。voice_dialogue/use_with_voice0_tts/test_e2e 已内置；写新的 voice0-melo 组合程序时别忘了这两行。

## 语音对话子程序（voice1 ASR + DeepSeek LLM + voice0 TTS）

单进程非阻塞编排：`dialogue/` 包（LLM 客户端 + 对话控制器 + 麦克风基建）＋
`examples/voice_dialogue.py` 主程序。**只读引用 voice0**（TTS 组件在 voice0 仓库，
不在这里改；本程序把 voice0 路径塞进 `sys.path` 导入 `from tts import RealtimeTTS`）。

- **跑法**（中文输出必须 `PYTHONIOENCODING=utf-8`）：
  `python examples/voice_dialogue.py --asr-device cuda --tts-device cuda`
- **参数含义白话版**（vad-tail / post-commit-window / echo-guard / merge-window 的直觉 +
  时间线 + 校准）：见 [`docs/voice-dialogue.md`](docs/voice-dialogue.md)。用户强调这些
  参数很难懂，解释时先讲直觉（"你停多久算说完""AI 答完但音频没播的空档""回声防护"），
  别只念数值。
- **机密**：DeepSeek API key 只放 `dialogue/config.local.json`（`.gitignore` 已排除，
  **绝不提交/绝不外传**）；读取优先级 显式参数 > 配置文件 > 环境变量 `DEEPSEEK_API_KEY`。
- **打断（barge-in）**：LLM 在途时来新 ASR 句 → `gen` 代际 +1 弃流（生成器 close 关连接），
  **重发本轮累计**（句1+句2…）；被作废的回复不 commit 历史。
- **停用词"停下"**：`--interrupt-words`（默认"停下"）。KWS 旁路命中 → `interrupt()` →
  `on_interrupt` 回调 → 控制器 `hard_stop()`：立即终止 LLM 流与 TTS 输出；**被打断的问题
  保留进历史**，"停下"本身经 KWS 旁路吞掉、绝不进历史/LLM 输入。
- **回声半双工门控（v1）**：TTS 播放期（`ctrl.tts_busy`）mic 只喂 `asr.ingest_kws_only()`
  （只听"停下"，回声不进识别 → 无反馈自答）；`--no-echo-gate` 关（耳机近场可用）。
  忙碌跟踪靠 voice0 `Job.done` + 守护 watcher 线程（voice0 无播放回调且不可改）。
  **门控开启有滚动 grace**（`--echo-guard` 默认 1200ms）：回声还没到（首句仍在合成）的
  窗口内，mic 块有语音能量（> -38dB）就顺延"仍喂正常识别"到 现在+静音尾+0.2s，让"AI 开答
  瞬间用户还没说完的尾巴"走完 VAD 静音尾定稿（否则被切去 KWS-only、悬成 partial 被吞）；
  回声一到由 `--echo-guard` 硬上限兜住不自答。voice0 无播放回调，音频"是否已开播"无精确
  信号，post-commit 窗口以首句提交时刻作代理。
- **首句 hold-off**：`--reply-hold`（默认 0.35s）——每轮回复首句先锁外延迟，给用户续句
  打断窗口；hold 内 barge → 弃句绝不播。代价是每轮首包音频 +0.35s。
- **VAD 静音尾长 `--vad-tail`**（默认 600ms）：判句末的停顿阈值，只管"识别何时收句"。
  别指望调大它根治句中停顿拆句——组织语言的停顿实测常超 1s（`--vad-tail 1000` 仍切），
  任何固定尾长都拆不干净；拆句/残句被吞由 **post-commit barge** 兜底（见下，零延迟）。
- **post-commit barge（拆句根治，零固定延迟）** `--post-commit-window`（默认 1500ms）：
  残句定稿**立即发 LLM**；AI 已答完但音频还没开播（本轮首句提交至今 < 窗口，≈melo 首句
  合成延迟）时用户补句 → `_rollback_last_turn_locked()` 撤下刚 commit 的 (残句→答复)、
  残句+新句连同历史重发。只在真补了句尾巴才重答，无每轮延迟。音频已开播后的续句 = 新轮。
  窗口锚点是 `_turn_first_submit_ts`（本轮首句提交时刻），`_launch_llm` 重置。
- **句末合并窗口 `--merge-window`**（默认 0=关）：断句后等窗口内补句才发 LLM（每轮固定
  延迟，用户已否决，留作可选）。`_merge_wait` 守护线程（0.1s 分片睡、新句重置 deadline）
  + `_launch_llm` 统一入口；`hard_stop`/`close` 作废挂起窗口。
- **历史压缩**：`--max-context-tokens`（默认 40000）。DeepSeek `include_usage` 精确
  `prompt_tokens` 计量；超阈值（预留 `headroom` 4000）且 LLM 空闲 → **一次性后台线程**
  调 LLM 压缩旧历史为摘要，最近 `recent_keep`（6）条原样保留，摘要拼进 system
  （【此前对话摘要】）。事件驱动，**无常驻监控线程**。压缩任务有快照竞态防护：换入前校验
  `_history` 未变，变了放弃本轮下轮再压。
- **线程纪律**：`feed_asr_sentence` 在 ASR worker 线程只做快操作（累加/gen/起 LLM 线程），
  绝不阻塞识别；锁序固定 `controller._lock(RLock) → tts._submit_lock`（RLock 因 finally
  在锁内 `_submit_tts` 会重入）。TTS 默认 `mode="queue"` 非打断。
- **控制台诊断标记**（区分「没提交 / LLM 卡住 / LLM 出错」）：`_Console` 三态行——
  `… `前缀=ASR 流式出字**未定稿**（不会提交）；`[ts-ts]`=定稿句已提交给 LLM；
  `→ LLM 请求中…`=LLM 请求已发出等首 token（controller `on_llm_start`，首 delta 原地覆盖）；
  `× LLM 出错`=流抛异常（`on_llm_error`）；`[门控]`=回声门控转换提示
  （AI 播放期 mic 只听"停下"，此刻说话不被识别——离远/音量低时 VAD 不闭句，句子
  "悬在流式 cache"永远不定稿，正是`… `行无后续的成因）。
- headless 测试：`tmp/test_dialogue.py`（gitignored）——fake LLM/TTS 覆盖切句/barge-in/
  hold-off/hard_stop/busy/压缩/引擎钩子（`RealtimeASR.__new__` 绕过模型加载）。

## 代码结构

```
asr/
├── core/      后端无关骨架
│   ├── engine.py   RealtimeASR（单例/常驻 worker 线程+有界队列/VAD 断句/lifecycle/打断词旁路）
│   ├── jobs.py     SentenceResult（…/ttfb/stale）+ PartialResult（T13，on_partial 流式出字）
│   ├── backend.py  ASRBackend（ABC：load/recognize/recognize_stream/reset/close）+ get_backend(name) 惰性加载
│   └── audio.py    音频工具（read_audio 通用解码 wav/mp3…、read_wav、resample_to、EnergyVAD 断句状态机）
├── kws/        打断词旁路（T12）
│   ├── interrupt.py   InterruptDetector ABC（load/feed/detect/reset/close）+ get_interrupt_detector
│   └── sherpa.py      SherpaKwsDetector（zipformer 3.3M int8；feed 流式主路径 / detect 兜底）
├── paraformer/  ParaformerBackend（默认主力，FunASR 流式，cache 模式可增量）
├── whisper/     WhisperBackend（可选高精度，离线非流式，本地缓存路径加载）
└── sherpa/      SherpaBackend（CPU 轻量基线，sherpa-onnx zipformer）
dialogue/        语音对话子程序：llm.py（OpenAI 兼容 SSE 客户端 + compress）/
                 controller.py（DialogueController：barge-in/hard_stop/tts_busy/hold-off/历史压缩）/
                 mic.py（MicAGC/check_mic_signal/pick_input_device）
                 config.local.json（机密 API key，gitignored，绝不提交）
bench/           bench_asr.py（整句 CER/RTF/延迟）+ bench_streaming.py（流式 vs 整句出字延迟）
examples/        transcribe_file / record_mic / demonstrate_interrupt（T12）/ demonstrate_streaming（T13）/
                 use_with_voice0_tts（共享 voice-asr 环境组合 demo）/ voice_dialogue（语音对话主程序）
docs/            asr-architecture-decision.md（ADR，选型/标定/环境决策）
assets/          验收语料（CER 裁判集）
reports/         bench 报告（gitignored）
```

## 权威文档（动手前先读对应章节）

- `docs/asr-architecture-decision.md` = **选型结论与硬指标**（ADR，从立项第一天写起）。
- `docs/engine-guide.md` = **引擎使用与工作原理指南**（线程模型/API 逐参/SentenceResult 字段/wall 与 audio 轴/VAD 原理/后端对比/**§9 热词纠错（同音字）**）。
- `docs/voice-dialogue.md` = **语音对话参数白话解释**（vad-tail/post-commit-window/echo-guard/merge-window 的直觉、时间线、为什么 post-commit 是时间窗、校准表）。
- `docs/backend-guide.md` = **新增后端接入指南**（流式/非流式后端契约、三步接入清单、引擎消费语义、验收纪律，接 SenseVoice 等新模型时先读）。
- `README.md`「引擎设计」（T6 后落地）= RealtimeASR 完整设计（一分钟上手）。
- `docs/ai-project-methodology.md`（在 voice0 仓库） = 本项目沿用并沉淀的 **AI 项目全流程方法论**，可复用。

## 协作习惯（本项目）

- 与用户用**中文**交流。
- 里程碑式改动后用户会说「commit吧」——按其节奏提交，commit message 用中文。
- **验收纪律**：每个后端/改动必须跑 CER 内容级复核，不要只报 RTF/延迟数字。
