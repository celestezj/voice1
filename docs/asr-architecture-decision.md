# 实时 ASR 方案选型（2026-08 立项草稿）

> 本文件是项目的**架构决策记录（ADR）**：为什么选 FunASR/paraformer-streaming 为主力、
> faster-whisper 为可选项、硬件与环境的来龙去脉、验收约定。
>
> **状态：立项草稿**。探索阶段（T3-T5）的实测数字将回填本表；最终定案后本文件成为权威依据，
> 实现细节见 README（引擎设计、环境与复现）。本文从第一天写起——决策记录是"当时的思考"，不是事后的补丁。

## 目标与核心指标（口径写死）

- **目标**：实时音频→文本，流式"边说边出字"，**离线运行**（最终方案不联网），中文准确率高。
- **硬指标**（可测、有阈值、可判定）：
  - **RTF（实时率）** = 识别耗时 / 音频时长，一次完整识别计。GPU < **0.3**，CPU < **1**（识别不比说话慢）。
  - **尾字延迟** = 说话停止时刻 → 该句完整文本回调时刻（VAD 判定停止 + 计时戳）。GPU < **0.5s**。
  - **CER（字错误率）** = 编辑距离 / 总字数，固定测试集（自建 20-30 句中文已知文本）对照。**< 5%**。
  - **离线**：权重缓存后运行期零网络请求。
- **软目标**（排序但不设及格线）：流式首段延迟 <0.6s、标点恢复、数字归一化、热词纠偏。
- 口径不清会两个人测出两个数——以上定义写死在 `bench/bench_asr.py` 与 README 验收区。

## 硬件与开发环境

- GPU：**RTX 2070 SUPER 8GB**（Turing sm_75），与 voice0 同机；总显存 8.59GB。
- conda 环境 **`voice-asr`**：从 `voice-tts` **克隆**（含 torch 2.11.0+cu126、已修 jax/setuptools 坑），
  省去 torch ~2.5GB 重下载；克隆后遇版本不对**在 voice-asr 内单独重装**，不干扰 voice0 的 voice-tts 环境。
  **（2026-08-27 克隆验证通过：torch 2.11.0+cu126 / CUDA=True / RTX 2070 SUPER / numpy 2.2.6 / transformers 4.57.6）**
- 本机网络：huggingface.co / raw.githubusercontent.com 直连被墙或极慢；**hf-mirror.com 与 ghfast.top 代理可用**（voice0 已验证，仅首次下载权重用）。
- 采样率：ASR 链路统一 **16kHz**（麦克风/模型）；voice0 TTS 是 44.1kHz，两条链路互不干扰。

## 候选方案对比（2026 版开源 ASR 横向；**状态：待实测回填**）

| 方案 | 中文准确率 | 原生流式 | CPU 实时 | GPU | 离线 | 依赖 | 许可 | 状态 |
|---|---|---|---|---|---|---|---|---|
| **FunASR / paraformer-zh-streaming** | 第一梯队 | ✅ chunk 流式 | ✅ 0.24~0.26 | ~0.87GB | ✅ | 中（pip/modelscope） | 阿里 | **主力 ✅ 已实测** |
| **faster-whisper**（large-v3/medium） | 高（通用） | ❌ 滑动窗口模拟 | ❌ CPU 2.23 | GPU ✅ 0.20 | ✅ | 低（pip/ctranslate2） | MIT | **可选项（GPU 离线）** |
| sherpa-onnx（zipformer-zh-14M） | 好 | ✅ 原生 | ✅ 0.037 | 可 | ✅ | 低（pip/onnx） | Apache | **基线 ✅ 已实测** |
| Vosk | 一般 | ✅ | ✅ | 极小 | ✅ | 低 | Apache | 待测（或作基线） |
| SenseVoice-small | 好、低延迟 | 一般 | ✅ | 小 | ✅ | 低 | 阿里 | 待测 |
| ~~云端（讯飞/阿里云/火山/OpenAI）~~ | 高 | ✅ | — | — | ❌ 需联网 | 零 | 闭源 | **排除**（离线剪掉） |
| ~~NeMo~~ | 高 | ✅ | — | 重 | ✅ | 重 | Apache | **排除**（依赖过重） |
| ~~纯端到端音频大模型（GPT-4o audio 类）~~ | — | — | — | — | — | — | — | **排除**（尾字延迟 >1s） |

## 选型定案（2026-08-27，T4 实测后）

- **主力**：**FunASR / paraformer-zh-streaming** ✅ 已实测（GPU RTF 0.139 / CPU 0.262 / 显存 870MB）。
- **可选项**：**faster-whisper medium** ✅ 已实测（GPU RTF 0.204；CPU RTF 2.229 不实时 → 定位 **GPU 离线专用**，自带标点）。
- **对照基线**：**sherpa-onnx zipformer-zh-14M** ✅ 已实测（CPU RTF 0.037，加载 1.2s）。
- **排除项**：云端全家桶（离线剪掉）；NeMo（依赖重收益不值）；纯端到端音频大模型（延迟 >1s）；faster-whisper 的 **CPU 实时**场景（RTF 2.229）。
- **双轨原则**：主路线（实时流式）+ 可选项（高精度）分离；**CPU 实时链路由 FunASR 独担**。

## T4 实测记录（2026-08-27，探针 `tmp/probe_funasr.py`）

**FunASR / paraformer-zh-streaming（`speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`）**：

| 项 | GPU (2070S) | CPU |
|---|---|---|
| 模型下载 | 11 文件 ~41s（modelscope 直连，缓存落 `.cache/modelscope`） | 同 |
| 加载耗时 | 首次 47s / 二次 6s | 5.8s |
| 一次性 RTF（3.32s 音频） | **0.183** | **0.241** |
| 流式 chunk RTF（chunk=50/100ms） | **0.139** | **0.262** |
| 峰值显存 | **~870MB** | — |
| 内容级验证 | ✅「今天天气真不错我们一起去公园散散步吧」 | ✅ 同 |

- 硬指标判定：GPU RTF<0.3 ✅、CPU RTF<1 ✅（3.32s 短音频样本；长音频 RTF 随固定开销摊薄待 bench 复核）。
- ⚠️ 观察到**流式 chunk 会丢重复字**：一次性「散散步」vs 流式「散步」——chunk 边界信息不足所致，CER 语料阶段量化该代价（对应方法论"区分固有代价与可优化点"）。

**faster-whisper（medium，可选项）**：

| 项 | GPU (2070S, float16) | CPU (int8) |
|---|---|---|
| 模型下载 | hf-mirror 镜像（缓存落 `.cache/hf`） | 同 |
| 加载耗时 | 首次 71.7s | 3.4s |
| RTF | **0.204** ✅ | **2.229** ❌（CPU 无法实时） |
| 内容级验证 | ✅ 自动带标点：「今天天气真不错**,**我们一起去公园散步吧」 | 同 |
| 显存 | ctranslate2 独立分配，torch 测不到（后续 nvidia-smi 复核） | — |

- **定位修正**：faster-whisper medium **CPU 不实时**（RTF 2.23）→ 定位为 **GPU 高精度离线转写可选项**；CPU 实时链路完全由 FunASR 承担。**自动标点**是加分项（软目标「标点恢复」）。

**sherpa-onnx（zipformer-zh-14M，对照基线）**：

| 项 | CPU |
|---|---|
| 模型下载 | ghfast.top 代理（GitHub releases，缓存落 `.cache/sherpa_models`） |
| 加载耗时 | **1.2s** |
| RTF（一次性/100ms块/50ms块） | **0.036 / 0.037 / 0.037** |
| 内容级验证 | ✅ 与 FunASR 同（同样丢「散」重复字） |

- API 坑（1.13 版）：循环 `recognizer.is_ready(stream)` → `decode_stream`；收尾 `stream.input_finished()`；波形须 `np.ascontiguousarray(float32)`。
- 定位：**CPU 轻量实时基线**（RTF 极低 + 加载 1.2s），类比 voice0 的 `sapi/` 对照基线。

## 实施顺序（沿 voice0 方法论）

1. **立项（T1-T2）**：骨架 + 本 ADR + CLAUDE.md（本文档）。
2. **探索（T3-T5）**：克隆环境 → 候选后端最小验证（RTF/显存/权重可达性/流式）→ 建立 CER 验收语料。
3. **实施（T6-T8）**：core 骨架（ASRBackend 抽象 + RealtimeASR 引擎 + VAD）→ 接主力 paraformer → 接可选项 faster-whisper。
4. **测试（T9）**：bench_asr.py + 内容级验收（CER<5%）。
5. **迭代（T10）**：VAD/chunk 参数标定 + 瓶颈量化。
6. **文档+复现（T11）**：README/CLAUDE.md 完善 + 版本锁定 + preload 脚本 + 从零复现步骤。

## 验收约定（本项目强约定）

- `bench/bench_asr.py` 产出 **CER / RTF / 尾字延迟**，**CPU 与 GPU 各跑一轮**，结果自动回写 README 验收区间。
- **内容级验收**：CER<5% 达标判定——绝不止看 RTF/延迟（voice0 血泪：数字漂亮≠输出正确）。
- **探针文化**：一次性诊断脚本隔离「采集 / VAD / 识别」三段，用后即弃但**留存输出**。
- 运行时零网络请求（仅首次下载权重可联网）；权重经 `preload_asr.py` 预下载到项目内 `.cache/`。

## 实施记录（T6 已完成 · 2026-08-27）

**core 骨架全链路冒烟通过**（sherpa 后端 · CPU）：5 段语料全部正确识别、回调触发 5 次、ttfb 0.016–0.125s（≪0.5s 硬指标）。`tmp/smoke_t6_result.txt` 留存。

- **VAD 断句 bug（已修）**：初版 `EnergyVAD.add()` 用 `ptr` 扫描后 `_buf = _buf[ptr:]`，把**尚未断句的语音帧也裁剪丢弃**——melo 单句无 600ms 连续静音，句子永远留在语音态，flush 时缓冲已空 → 0 句。重写状态机：语音帧保留在 `_buf`（属当前句），`_sentence_start` 记录句首采样，仅修剪句前静音；断句时切出 `[_sentence_start, cut)`（保留 tail 静音保边）。分块 50ms 流式喂入复验正确。
- **时序基准设计（决策）**：实时流与文件同步是两条时间轴。`recog_axis`：
  - `"wall"`（实时）：recog 时刻 = monotonic 相对 `_t0`；ttfb 天然含 VAD 尾长 + 识别延迟。
  - `"audio"`（文件同步，`ingest_file`）：加速喂入，识别时刻映射到音频轴（断句即识别），**ttfb = 纯识别耗时**，与 feed 加速无关。
  - 文件 t=0 对齐会话起点 `_t0` → `audio_start/end` 恒为文件内相对秒（正数，bench 可比）。
- 初版 ttfb 为负、后为巨值：均为"两条时间轴未统一"的同一问题（wall 轴减 `_t0` 与 audio 轴直接用 `e_ts`），已用上表根治。

## 实施记录（T7 已完成 · 2026-08-27）

**ParaformerBackend 接入全链路通过**：24 句语料全部出结果，长句正确断句（s22/s23 各断 2 句、每段识别正确）。`tmp/smoke_t7_result.txt` 留存。

- **`is_final=True` 关键坑**：FunASR 对 numpy 数组输入默认 `is_final=False`，末 chunk 被丢弃（实测整句截尾缺"技术系"）。句子级 `recognize` 必须显式传 `is_final=True`。
- **`recognize_stream` 增量形态**：FunASR 官方 cache 流式——同一个 `cache={}` dict 跨调用复用 + `is_final` 标记，60ms 粒度逐块输出（实测 JOIN 完全正确：'他毕业于'→…→'技术系'）。模型 `inference` 对空 cache 自动 `init_cache`；**不能**传 `{"cache": {}}` 嵌套结构（len≠0 不触发 init → `KeyError: 'prev_samples'`）。
- **VAD 长句断句正确**：逗号处停顿 >600ms → 断成 2 句，每段识别正确（"晚上八点的高铁"、"准备去杭州旅游"均在第二句）。
- **裸 CER 0.175 构成（非最终口径）**：① smoke 只显示 `res[0]`（第二句未拼入，属脚本显示问题）；② ref 含标点而 hyp 无标点，CER 按全字符计算被抬高；③ 个别错字（谈→滩、周末→中国、开到→看到）。正式 CER 在 T9 以「去标点 + 整句拼接」口径验收。
- 模型加载 ~14s（CPU 首次含下载）。

## 实施记录（T8 已完成 · 2026-08-27）

**WhisperBackend 接入全链路通过**（faster-whisper medium · GPU）：24 句全出结果、长句拼接完整（s22/s23 一次过）、8 句严格 CER=0。`tmp/smoke_t8_result.txt` 留存。

- 加载 ~6s（模型已缓存 `.cache/hf/`），GPU 每句 250–770ms（不含句长趋势明显）。
- **CER(去标点)=0.120 构成（归因）**：
  - **数字形态**（非听错）：八点→8点、百分之二十→20%、一三八零零零…→1380012345、六点五→6.5%。
  - **繁体**：中國→中国、我們→我们、電影→电影（faster-whisper 默认输出简体概率高但偶发繁/异体）。
  - **真实错字**：请把→起码、这道→知道、这款→去款、三点整…→…胎时…、音箱→音响（同音）。
- **对比数据点**：paraformer 保持中文数字、无繁体 → 中文数字/繁体类指标 paraformer 更优；whisper 在"它→她"等人称/同音错误上更弱于 CER 计数。正式硬指标用**严格 CER**（仅去标点），数字/繁体作为形态差异在 bench 报告中单独归因。
- **recognize_stream = NotImplementedError**：离线非流式，引擎按 ABC 契约回退"积累块 + 整句 recognize"——"边说边出字"对 whisper 不可用，选型时已知权衡。

## 实施记录（T9 已完成 · 2026-08-27）

**`bench/bench_asr.py` 内容级验收**：严格 CER（仅去标点）为主指标，规范 CER（+繁简统一+中文/阿拉伯数字归一）作形态差异归因；RTF、平均 ttfb、硬指标自动判定；报告落 `reports/bench_{tag}_{backend}_{device}.txt`。全 24 句、CPU/GPU 双测。

| 组合 | 严格CER | 规范CER | RTF | 平均ttfb | 硬指标(cer/rtf/ttfb) |
|---|---|---|---|---|---|
| **paraformer/cuda** | **0.047** | 0.047 | **0.150** | **0.400s** | **✓✓✓ 全达标** |
| paraformer/cpu | 0.047 | 0.047 | 0.257 | 0.688s | ✓✓ ✗（ttfb 超 0.5） |
| whisper/cuda | 0.120 | 0.058 | 0.139 | 0.370s | ✓✓ ✗（CER 超 5%） |
| whisper/cpu | 0.132 | — | 2.659 | 7.164s | 全 ✗（CPU 不可用） |
| sherpa/cpu | 0.190 | 0.190 | 0.033 | 0.084s | ✓✓ ✗（CER 超 5%） |

- **主力定案成立**：paraformer/cuda 三项全达标（24 句 17 句严格 CER=0）。CPU 场景 CER/RTF 达标但 ttfb 0.69s 超标——CPU 推理慢是根本，留 T10 验证 VAD 优化空间。
- **whisper 严格 CER 的形态归因**：数字（八→8、百分之→%、手机号转阿拉伯）+ 繁体（中國/我們）占大头，规范 CER=0.058 接近门槛；剩余为同音/人称错字。**离线非流式不可"边说边出字"**。
- **sherpa/cpu RTF 0.033 极优但 CER 0.19 质量不足**（轻量 14M 模型）；s03 空文本、s04 半句——**短句识别缺陷**（记 T11 复现时标注）。
- **whisper 本地缓存加载修复**：`.cache/hf/` 快照文件完整但 `blobs/` 空 → huggingface_hub 联网校验 revision 失败（`LocalEntryNotFoundError`）。改为**检测本地快照路径直接加载**（`_local_model_dir()`），绕过 hub、满足"缓存后零网络"硬指标。
- bench 脚本两个坑：① 繁简映射 `str.maketrans` 括号提前闭合（IndentationError）；② `re.sub` 的 repl 函数收到 `re.Match` 而非字符串（`_zh_num` 需 `m.group(0)`）。

## 实施记录（T10 已完成 · 2026-08-27）

**VAD silence_tail_ms 标定（paraformer/cuda，24 句）**：`tmp/calib_vad_result.txt` 留存。

| tail | 严格CER | 断句数 | 平均识别耗时 | 实时尾字延迟估算 | 达标 |
|---|---|---|---|---|---|
| 150ms | 0.096 | 52 | 0.196s | 0.346s | ✓（CER 恶化） |
| **250ms** | **0.059** | 47 | 0.235s | **0.485s** | **✓（默认）** |
| 400ms | 0.050 | 27 | 0.376s | 0.776s | ✗ |
| 600ms | 0.047 | 26 | 0.406s | 1.006s | ✗ |

- **无单一 tail 同时达标**：CER 随 tail 单调改善（句尾越干净），实时延迟随 tail 单调恶化。tail=250 为实时最优折中（延迟达标、CER 0.059 逼近 5%）；tail=600 为"精准模式"（CER 0.047 达标、延迟超标，适合离线转录）。
- **CER 恶化根因 = 句尾语气词幻听**，非断句切错：tail 小 → 断句早 → melo 句尾拖音入句 → paraformer 幻听出尾字（"稍等一下啊"、"六点五六嗯"、"亿亿次"、"很高好"）。tail=600 时句尾干净、CER 最优。
- **瓶颈量化**：识别耗时与句长线性（RTF≈0.15 GPU 恒定）；尾字延迟 = VAD tail + 整句识别耗时，两者同量级（tail=250 时 0.25 vs 0.226s）。进一步降延迟需**流式增量识别**（边说边出字，paraformer cache 模式已备）——T11 后记。
- **引擎修复**：`RealtimeASR.__new__` 单例原先只比较 backend/device，改 `vad_silence_tail_ms` 不重建 → 无法标定。已把 tail 纳入重建条件。
- **bench 扩展**：`--tail` 参数，报告标注 VAD tail。默认 tail 改 250ms。

## 实施记录（T11 已完成 · 2026-08-27）

**文档 + 复现闭环**：

- **README.md**：硬指标 / 从零复现（可整段复制）/ 引擎设计（RealtimeASR 单例·worker·VAD·时序双轴）/ 后端矩阵验收表 / 目录 / 已知限制。全程去绝对路径。
- **docs/environment-voice1.md**：环境版本锁定（torch 2.11.0+cu126、funasr 1.4.4、faster-whisper 1.2.1、sherpa-onnx 1.13.6 等 11 项）+ 镜像源 + 从零复现。
- **preload_asr.py**：预下载三后端权重（paraformer→modelscope、whisper→hf-mirror、sherpa→ghfast.top），验证全 OK（缓存下秒级）。
- **examples/**：`transcribe_file.py`（文件转写，已验证）、`record_mic.py`（sounddevice 麦克风实时，wall 轴 ttfb 含 VAD 尾长）。
- **CLAUDE.md**：VAD 标定结论、代码结构修正（SentenceResult/sherpa）、API 示例更新。

**从零复现验证**：环境克隆 → preload → bench → examples 全链路命令在 README 可整段复制，已逐一跑通（本机缓存态）。

## 实施记录（T12 已完成 · 2026-08-27）

**打断词旁路（interrupt word bypass）**：用户说固定触发词（默认「停下」）→ 即时作废全部排队识别任务。
`tmp/test_t12c_result.txt` 留存（A 白盒 stale / B 积压打断 / C 恢复，全 PASS）。

### 需求缘起（打断悖论，用户质疑后澄清）

连续流场景用户连说多句 → 任务排队串行识别。此时说「停止」**不能**走普通识别队列：它是队尾一个
普通任务，等它被识别时前面的任务早已完成——打断悖论。因此打断指令必须**旁路**主队列，用极轻量
的独立模型毫秒级识别，命中即作废队列。

### 定案

- **不支持自动抢占**（用户说一长串，中途停顿>250ms 会被当作句末断句，若自动打断则只有最后一句被
  提交，前几句全丢——破坏连续语速说话）。**只支持旁路主动打断**：用户主动说「停下」。
- **KWS 旁路**：sherpa-onnx `KeywordSpotter`（zipformer wenetspeech **3.3M int8**，~毫秒级），与主 ASR
  并行运行。`interrupt_words=["停下"]` 时启用；加载失败仅告警降级为"无打断"，不阻塞主识别。
- **触发路径（关键）**：KWS 检测必须在 **ingest 流式 `feed()`**（音频块到达即喂，不等 VAD 断句）。
  若放在 worker 里等 VAD 断句成句再 detect，打断词音频在队尾，等 worker 处理到时前面任务已完成——
  同样悖论。VAD 断句后整句 `detect()` 仅作兜底（流式 miss 的第二道防线）。
- **触发块丢弃**：`feed()` 命中的音频块不入队（该"停下"音频不进识别管线），同时 `interrupt()` 作废
  全部排队任务、清 VAD/后端状态。
- **in-progress 任务 stale**：正在识别的那句无法中止（整句前向原子），完成后判 `stale=True`，
  **不进普通回调**（语义上属旧会话；profile 仍收集供诊断）。
- **task_gen 用出队块代际（T12c 实测发现的竞态修复）**：`_process_sentence_locked` 的 `task_gen`
  必须传**出队块的 gen** 而非处理时的当前 `_gen`。否则 interrupt() 的 `_gen+=1` 发生在 worker
  处理该块中途时，该块 VAD 里已含的触发词音频会以"新代际"被识别 → 触发词漏进管线（实测 paraformer
  把 melo「停下」误识别为"影响下"）。用块 gen 后该句判 stale 被丢弃。

### KWS 实现要点（坑）

- **建模单元是拼音（声母+韵母）非汉字**：keywords 文件 `t íng x ià @停下`。汉字→音节串用 pypinyin
  的 `to_initials/to_finals_tone(strict=False)` 自建转换（组合声母 `zh/sh/ch` **不拆**、带调韵母
  `uò` 不拆）。**不能**用 `sherpa_onnx.utils.text2token(ppinyin)`——它把 `sh` 拆成 `s h`、`uò` 拆成
  `u ò`，与 tokens.txt 建模单元不符（实测对照官方 test_keywords.txt 逐词一致才定案）。
- **命中需尾随音频收尾解码**：流式 KWS 要 ~0.2-0.4s 尾随音频才能 finalize 关键词（detect 整句用
  0.66s 尾静音同理）。真实麦克风持续采样天然满足；若音频流在「停下」后立即结束，命中延迟到后续
  音频到达（`num_trailing_blanks` 调 0 无效，实测）。引擎侧实测触发延迟 ~0.66-1.4s（含背压等待）。
- **每词 boost 文件格式不支持**（`2.0 @停下` 会被当 token）：用全局 `keywords_score`（默认 1.0 最稳；
  4.0/8.0 反而漏检，实测反直觉）。
- **模型非线程安全**：ingest 旁路（主线程 feed）与 worker 兜底（detect）可能并发调用 KWS → 检测器
  内部 `threading.Lock()` 串行全部 KWS 调用。
- **打断词音频质量**：melo TTS 短词「停下」sherpa ASR 主链路识别为空（TTS 短词质量），但 KWS 能命中
  ——不阻塞（打断词本就不走主链路）。
- **KWS 对喂入音频响度/信噪比敏感（T12d 实测）**：把静音音频（peak~0.04、低 SNR，如 corpus s01
  「你好。」）放大到 peak≥0.15 时，KWS 会漏检「停下」——放大抬高了噪声底，污染流式解码特征；
  而 peak≈0.10 是"VAD 能断句 + KWS 能命中"的实测公共区间（VAD 能量门限 -35dB/最短句 250ms 会把
  过静音的句子当噪声丢弃）。**引擎不自动归一化**（真实麦克风电平通常达标）；demo 脚本
  `demonstrate_interrupt.py` 归一至 peak 0.10 以兼容任意响度输入，注释说明了区间来源。

## 实施记录（T13 已完成 · 2026-08-27）

**流式逐帧出字（streaming）**：`streaming=True` 时 worker 对未断句块逐块 `recognize_stream`
出 `on_partial` 部分字（"边说边出字"），句末 VAD 断句边界块以 `is_final=True` flush 定稿完整句。
首字延迟从"整句话说完"压到 ~0.3-0.9s（与句长无关），尾字延迟恒定 ≈ VAD 尾 + flush 耗时。
bench 对比（`reports/bench_streaming_T13_paraformer_cuda.txt`）：首字 **0.932s vs 3.110s**、
尾字 **0.348s vs 0.626s**、严格 CER **0.017 vs 0.053**——流式全面胜出且达标（尾字 max 0.486s
< 0.5s；整句 max 1.022s 破线）。CER 把关通过：流式定稿 0.017 远好于整句 0.053，**无需退回整句定稿**。

### 需求缘起（整句路径的延迟天花板）

T13 之前引擎只有整句识别：VAD 断句 → 整句前向 → 回调。两个缺陷被用户质疑后量化证实：

1. **尾字延迟 = 整句推理耗时，随句长线性涨**。paraformer/cuda 长句（4.7s）≈0.93s（含 VAD 尾
   250ms），cpu ≈1.5s——破 0.5s 硬指标、长句破 1s。5s 长句按 0.15×句长（cuda）/0.28×句长
   （cpu）外推更糟。
2. **首字延迟等于尾字延迟**（整句话说完了，首尾字一起原子出来）——5s 长句首字延迟 ≈ 5.6s，
   真人感知是"我说完了 5 秒才出第一个字"。

`recognize_stream()` 三后端都有定义（paraformer cache 模式、sherpa OnlineRecognizer 真实现、
whisper 抛 NotImplementedError）但引擎零调用（engine.py 仅 `recognize`）。RTF 全达标（识别快）
但出字时机晚——这是「VAD 断句 + 整句识别」架构的天花板。

### 定案

- **后端契约**（`asr/core/backend.py`）：`ASRBackend.supports_streaming = False` +
  `recognize_stream(self, chunk, is_final=False)`。`is_final=False` 增量喂块返回当前部分文本；
  `is_final=True` 结束当前流返回最终文本，之后流状态失效（调用方随后 `reset()`）。paraformer
  与 sherpa `supports_streaming=True`；whisper 保持抛 NotImplementedError。
- **流式 flush 定稿**（用户拍板）：句末最终文本由流式定稿承担（边界块 `is_final=True` 收尾），
  尾字延迟压到 ~0.3s 恒达标。**CER 必须 bench 把关**——当时严格 CER 0.059 已在 5% 线附近，
  有回归风险；实测流式定稿 0.017，反而大幅改善，无回归。
- **退化**：流式调用抛异常 → 置 `stream_capable=False`，本会话剩余走整句（whisper/异常兜底）。
  后端不支持流式（whisper）→ 构造时告警降级，不中断主识别。
- **打断共存**：`interrupt()`/「停下」的 `_gen` 守卫同时管 partial 与 final；流式 cache 在
  `reset()` 里清空，打断后新会话的 partial 从零开始，不串旧句。

### 实现要点（坑）

- **FunASR cache 模式返回 DELTA 非累计**（`tmp/probe_stream_flush.py` 实测）：`generate(input=chunk,
  cache=cache, is_final=False)` 的 `res[0]["text"]` 是**本块新增**片段（"明天早"→"上八点"→"开会"），
  必须内部累加进 `_partial_buf`，`is_final=True` 时返回累加结果并清空。sherpa `get_result()` 本身
  返回累计（无需累加）。
- **流式吞吐达标**：paraformer recognize_stream RTF 0.16（4.7s 音频 47×100ms 块 0.73s 处理完），
  逐块出字远快于实时。
- **安静文件在流式路径漏断句**（`tmp/dbg_vad_state.py` 实测）：VAD 能量门限 -35dB、最短句 250ms，
  corpus s01「你好。」（peak 0.039）仅 **2 帧**过阈 < 12 帧 → 被当噪声丢弃 → 句子永不闭合 →
  与下一文件的语音合并成一句（实测 s01+s02 出 "你好今天天气不错"）。整句路径靠文件末
  `vad.flush()` 兜住残留缓冲，流式路径**无文件边界**（连续麦克风流）→ 必须让语料响度达标。
- **归一化只能放大、不能统一压到 0.10**（`bench_streaming` 迭代中踩坑）：把全部文件压到 peak
  0.10，会把 RMS 在 -34dB 附近的文件（s03/s06/s07）再压低 → 同样跌破 VAD 门限 → 漏断句 → 单
  文件卡 ~19s 拖垮整趟时序，整句 CER 0.059→0.108。修复：`_read()` 只把 `peak<0.10` 的安静文件
  放大到 0.10，响亮文件保持原电平（VAD 公平断句、识别电平不被改动）。
- **bench 的 paced 趟必须连续喂入、不能"喂一文件等一文件"**：后者每文件等 final 的 wait 期间
  VAD 时间线（`_ts_cur`）不推进、与真实墙钟脱节 → `audio_end` 被低估 → ttfb 虚高且逐文件累积
  （实测 24 文件后虚高到 1.8s）。连续喂入（文件间靠各自 400ms 尾静音停顿）模拟真实麦克风会话，
  VAD 时间线无空洞，ttfb 真实。

## 候选评估：whisper 滑动窗口真流式（已评估 · 暂不采用 · 2026-08-27）

**是什么**：whisper 架构不支持增量解码（整句 encoder 自注意力，无跨块状态），"真流式"只能靠
**滚动窗口 + 整窗反复重听**模拟：维护最近 N 秒音频窗口，每来新块对整窗 transcribe 一遍，用
whisper 自带 segment 时间戳只取窗口后段**新增**文本（重叠旧文本丢弃）输出为 partial。

**为何暂不采用**（用户问起后评估）：

| 维度 | 结论 |
|---|---|
| 首字延迟 | 短音频质量差须攒最小窗口（2~3s）→ 首字 ≥2s，**不达 0.5s 硬指标**（paraformer 流式 0.93s） |
| CPU 实时 | whisper CPU RTF 本就 2.23，整窗重推理再放大块数倍 → **完全不可行** |
| 去重/边界 | 重叠窗口靠时间戳裁剪，偏差即漏字/重字——方案 90% 的坑在这 |
| 价值 | whisper 定位是"可选高精度离线"，非实时主力；实时硬指标已由 paraformer 流式达成 |

**复用时钩子**（若将来要"whisper 精度 + 实时"）：不动 engine.py（T13 的 `recognize_stream` 契约已够），
只在 `asr/whisper/backend.py` 实现——维护 `self._window` 环形缓冲 + `self._last_ts`，
`is_final=False` 时整窗 transcribe 取 `seg.end > _last_ts` 的新段拼接返回；`is_final=True` 整窗定稿。
**触发条件**：仅在 cuda 场景、且用户明确要 whisper 精度实时时再启用。

## 实施记录（T16 已完成 · 2026-08-28）

**热词纠错（同音字）+ whisper-large 实测否决**。用户锚定问题：文学文本同音字识别错误
（神妙→神庙×2、心天→新天、四时→四十、于→与），要求"上更大的模型"。实测证明**换模型是
死路**，真正的解法是 FunASR 内置的**文本级热词后纠错（拼音级模糊匹配）**。

### 需求缘起（同音字是纯音频歧义）

`E:\temp\语音包\xiaoshuang.mp3`（里尔克《给青年诗人的信》节选）paraformer 流式输出错误：
神庙/新天/四十/与。同音字 神庙↔神妙 拼音同为 `shenmiao`，纯声学上不可区分——NAT（paraformer）
与自回归（whisper）的声学先验都不足以解决，必须引入**词汇先验**（热词偏置）。

### whisper-large 实测（否决，T16）

接入 large-v3-turbo（`mobiuslabsgmbh/faster-whisper-large-v3-turbo`，hf-mirror xet 仓库
curl 手动落盘绕过 huggingface_hub 跨域坑）。实测结果：
- **锚定句不修复**：神庙（句2）✗、新天+季后（句3）✗、四十（句4）✗。
- **语料严格 CER 0.141**（规范 0.098）——全部后端最差（medium 0.120、paraformer 0.031），
  s09 幻觉出"感谢观看"（0.615）。RTF 0.112✓ / ttfb 0.266s✓ 达标。
- **结论**：turbo 蒸馏版在中文短句语料上质量不足；同音字问题换模型不解。**保留为可选后端**
  （已接入、可加载、RTF 快），但文档标注"不建议"，默认推荐 paraformer + 热词。

### 热词纠错（定案，用户拍板"接入热词纠错"）

FunASR 1.4.4 `AutoModel.generate` 内置 `postprocess_hotword_file`：rapidfuzz + pypinyin
**拼音级模糊匹配**（神庙/神妙 拼音相同 → 相似度 1.0 自动替换）。接线到 paraformer 后端：

- `ParaformerBackend(hotword_file=...)`：load() 编译 `PostprocessHotwordMatcher`；
  `_correct()` 对返回文本统一应用。**坑**：流式 generate 返回**增量 delta**，跨块单词
  （"神/庙"分两次返回）片段内匹配不到目标词 → 纠错在**本类统一对累计 `_partial_buf`**
  应用（online 流式）或整句文本（offline），不在 generate 时透传。
- 引擎 `RealtimeASR(hotword_file=...)` 透传；后端不支持（whisper）→ TypeError 捕获告警降级。
- 文件两种模式：**显式映射** `神庙=>神妙`（零误伤，推荐）+ **模糊目标** `神妙`（拼音级兜底）。
- 实测（`assets/hotwords/xiaoshuang.txt`）：显式映射下 paraformer offline 四句**全部精确命中
  原文**（神妙×2/心天/季候/四时/减少于）；online 流式同 100% 命中（ttfb ~0.1s）。
- **坑（模糊目标的副作用）**：对 2 字词可能吞相邻同音字——"的神妙"（deshenmiao vs shenmiao
  0.94）整窗替换删掉"的"；"必减少于"被"减少于"命中删掉"必"。**精度要求高必须用显式映射**。
- 回归：paraformer/cuda 语料严格 CER 0.031（无热词 matcher=None 路径）零影响。
- 软目标回填：ADR 目标清单第 17 行"热词纠偏"软目标由此达成。

- [x] T3 克隆 `voice-asr` 环境并验证 torch/CUDA 可用，快照入档（2026-08-27 通过）
- [x] T4 候选后端最小验证：FunASR / faster-whisper / sherpa-onnx——RTF、峰值显存、权重可达性、流式 → 已回填对比表
- [x] T5 建立验收语料：24 句中文已知文本 + voice0 melo TTS 合成音频（`assets/corpus/`，UTF-8 manifest，总时长 70.1s）

### 实施阶段 T6-T8
- [x] T6 搭 core 骨架（ASRBackend 抽象 + RealtimeASR 引擎 + EnergyVAD + SentenceResult），sherpa 后端全链路冒烟通过（2026-08-27）
- [x] T7 接入主力后端 paraformer-streaming 跑通端到端（24 句全出结果，2026-08-27）
- [x] T8 接入可选项 faster-whisper 后端（24 句全出结果，CER 归因已记，2026-08-27）

### 测试 / 迭代 / 复现（T9-T11）
- [x] T9 bench_asr.py + 内容级验收（CER/RTF/延迟，CPU/GPU 双测；paraformer/cuda 全达标，2026-08-27）
- [x] T10 VAD 参数标定 + 瓶颈量化 + 复验（默认 tail=250，权衡曲线已记，2026-08-27）
- [x] T11 文档+复现：README/CLAUDE.md 完善 + 版本锁定 + preload 脚本 + examples + 从零复现（2026-08-27）

### 打断（T12）
- [x] T12 打断词旁路：KWS 流式 feed + interrupt 拆步 + stale 标记 + task_gen 竞态修复 + 文档/代码 case（2026-08-27）

---
*本文为 voice1 项目 ADR。立项日期 2026-08-27。*
