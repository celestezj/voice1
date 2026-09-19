# CLAUDE.md — voice1 项目指南

离线实时中文语音识别（ASR，音频转文字）系统。核心指标：**RTF <0.3(GPU) / <1(CPU)**、
**尾字延迟 <0.5s**、**CER <5%**、**离线运行**（权重缓存后零网络请求）。

两个后端（**选择性安装**，互不影响）：
- **paraformer**（默认，实时主力）：FunASR / paraformer-zh-streaming，chunk 流式，边说边出字。
- **whisper**（可选项，高精度离线）：faster-whisper，通用中文精度上限，滑动窗口模拟流式作后备。
- 对照基线（非神经）：sherpa-onnx tiny / Vosk（探索后定，类比 voice0 的 `sapi/`）。

## 快速上手（安装 → 使用）

1. **环境**：先 `conda activate voice-asr` 再跑 `python`。**优先使用 `voice-asr`**（它是
   voice-tts 的严格超集，含全部 ASR + agent 依赖如 `claude_agent_sdk`）；**仅当本地只有
   `voice-tts`（voice0 共享基座）时**才用 voice-tts 并就地补装缺失依赖——**不必新建
   voice-asr**。三态判定：两者都有→voice-asr；只有 voice-asr→voice-asr；只有
   voice-tts→voice-tts（缺啥装啥，torch 2.11+cu126 等）。voice-asr 原从 voice-tts 克隆，
   如遇版本冲突在其内单独重装，**不碰 voice0 的 voice-tts 环境**。
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

- **Python 必须用 `voice-asr` / `voice-tts` conda 环境，`voice-asr` 优先**：先
  `conda activate voice-asr`；**仅当本地没有 voice-asr（只有 voice0 的 voice-tts）时才
  `conda activate voice-tts`**——二选一、缺依赖就地补装，**绝不新建 voice-asr**。base
  Python 3.12 无 torch。**后续文档/代码里凡提到 `voice-asr`，要知道环境判定的三态：两者都有
  →voice-asr；只有 voice-asr→voice-asr；只有 voice-tts→voice-tts 替代（voice-asr 是
  voice-tts 的严格超集，含 agent 依赖如 `claude_agent_sdk`，所以优先它）**。
- **跑带中文输出的命令加 `PYTHONIOENCODING=utf-8`**：Windows 默认 GBK 会直接崩。
- **权重/缓存重定向到项目内 `.cache/`**（`HF_HOME`/`HF_ENDPOINT`/`MODELSCOPE_CACHE` 在模块里已设好）；首次下载走 `HF_ENDPOINT=https://hf-mirror.com` 镜像（huggingface.co 直连被墙）。
- git 仓库根即本目录（独立仓库，**不含 voice0 内容**；`third_party/` 是 gitignored 的上游 clone，别在里面提交）。GitHub 地址：https://github.com/celestezj/voice1。新用户一键安装：`python setup_env.py`（复用 voice0 脚本建 voice-tts 底座 → 克隆出 voice0/voice1 共用的唯一环境 voice-asr → 装 ASR 依赖 → preload 权重 → 验证）。**本地已有 voice0 的 `voice-tts` 且无 voice-asr 时，直接用 voice-tts 就地补装缺失依赖（同基座），不必克隆新建 voice-asr。**
- **`voice-asr` 也是 voice0(TTS) 的运行环境（两项目共享，实测 2026-08-28）**：voice-asr 当初从 voice-tts 克隆，是 voice-tts 的**严格超集**（等价地，直接在 voice-tts 里补装 ASR 依赖后也可作 voice1 运行环境，见上「环境安装」）——voice0 的 melo 后端（`from melo.api import TTS`，导入名是 `melo` 不是 `MeloTTS`）在 voice-asr 可直接跑（合成冒烟通过）。同进程组合示例见 `examples/use_with_voice0_tts.py`（TTS→ASR→TTS 闭环）。**组合时 HF_HOME 是唯一可能冲突的环境变量**：melo 用 voice0/.cache/hf，voice1 仅 whisper 后端才用 HF_HOME（默认 paraformer 走 MODELSCOPE_CACHE，无冲突）——显式播种见该示例。**cosy 后端不能与 melo/ASR 同进程**（voice0 设计约束：cosy 注入 transformers 4.51.3，与主环境 4.57.6 冲突，须独立子进程）。

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
- **麦克风电平够不着 VAD 门限 → "说话没反应"（T17c 实测）**：VAD 断句门限 -35dB，但不少麦克风说话 RMS 只有 -36~-46dB（本机 HD Audio 麦实测 6s 仅 24/300 帧过阈）——**录音正常、识别全无**。`check_mic_signal` 只拦"全哑（<-80dB）"拦不住这个。解法：record_mic 用 `MicAGC` 采集层自适应放大（目标 peak 0.3、只放大不压小、上限 24x）。**引擎层故意不归一（T12d），mic 层负责**。排查 mic 无反应先跑 `tmp/probe_mic.py` 看电平与过阈帧数。
- **MicAGC v2：锁存噪声门控保证远距离句子能定稿 + 不切弱音节（2026-09 实测）**：旧 AGC
  上限 8x + 快攻慢放有个隐性 bug——远距离说话把增益顶到上限，说话结束后增益停在原位，
  房间底噪 ×8（+18dB）后 ≥ -35dB → VAD 把底噪当"还在说话"，静音尾永远凑不满 → 句子
  永不定稿、只出 partial 不提交（"离远说完了还在等我"）。修法（分两步，第二步是用户
  实测逼出来的）：①**锁存门控**——低于 `底噪×margin` 的块**持续 ≥120ms 才置零**（尾静音
  真静音 → VAD 正常收句），说话时短暂弱音节（<120ms）不被切——初版"低于门限直接置零"
  导致"距离一远文字出错/半截话"（弱音节被吞，实测 -42/-50 下旧硬门控 4 句全啃烂）；
  ②**底噪只在锁存确认的真静音块上更新**（说话期间完全冻结）——低信噪比远距说话不会把
  底噪估计慢慢抬进门限、把句子尾巴吞掉。上限提到 24x（门控保证底噪不被一起抬上去）。
  门控同时让唤醒/打断 KWS 只看到干净语音。**SNR <~3dB 是门控下限**（语音均值低于门限，
  无 AGC 也识别不清，属声学极限需靠近）。`--mic-gain` 可调上限（默认 24）。
  `--vad-threshold-db` 调低是旧补救（更易误断句），一般不再需要。
  **改 MicAGC 记得同步 `examples/record_mic.py` 的同源副本**。
- **麦克风 16kHz / 模型 16kHz**：采样率与 voice0 TTS（44.1kHz）不同，两条链路各管各的。
- **回声/双讲（AEC）**：若与 voice0 组合成语音对话，麦克风会收到喇叭声音，需回声消除。
- **MeloTTS-Chinese 被切成 Xet 存储 → huggingface_hub 绕开缓存重下 208M（2026-08-29 实测）**：仓库启用 Xet 后，新版 huggingface_hub 把 xet 仓库当"未缓存"，即使权重完整躺在 voice0/.cache/hf 也重新下载 config.json+checkpoint.pth（hf-mirror ~70kB/s，卡 46 分钟）。修复：组合程序 import 前设 `HF_HOME=voice0/.cache/hf` **且** `HF_HUB_DISABLE_XET=1`（实测 0.55s 命中缓存零下载）。voice_dialogue/use_with_voice0_tts/test_e2e 已内置；写新的 voice0-melo 组合程序时别忘了这两行。

## 语音对话子程序（voice1 ASR + DeepSeek LLM + voice0 TTS）

单进程非阻塞编排：`dialogue/` 包（LLM 客户端 + 对话控制器 + 麦克风基建）＋
`examples/voice_dialogue.py` 主程序。**只读引用 voice0**（TTS 组件在 voice0 仓库，
不在这里改；本程序把 voice0 路径塞进 `sys.path` 导入 `from tts import RealtimeTTS`；
voice0 的参数在 voice1 的 CLI 上透传，如 `--tts-normalize` rms/agc 响度归一化）。
voice0 仓库地址：https://github.com/celestezj/voice0

- **跑法（推荐）**（中文输出必须 `PYTHONIOENCODING=utf-8`）：
  `python examples/voice_dialogue.py --asr-device cuda --tts-device cuda --vad-tail 300 --system-prompt dialogue/user_prompt.txt`
  （`--vad-tail 300` 比默认 600 每轮首包快 300ms；残句由 post-commit barge 兜底，停顿多
  就调回 600）
- **TTS 后端/音色**：默认 **melo**（一键启动即 melo）。换 vits 多音色：`start_dialogue.bat vits`
  快捷词（或 `--tts-backend vits`）；音色用 `--tts-voice-id <id或名字>`（默认 551 派蒙；
  `--tts-list-voices` 打印 804 个音色）。vits 权重在 voice0/.cache/vits/（voice0
  `preload_vits.py` 一次性下载，voice0 只读不代管）。换 **moss**（MOSS-TTS-Nano，
  CPU 实时 + 原生流式 + 零样本克隆）：`start_dialogue.bat moss`（或 `--tts-backend moss`）；
  音色 `--tts-voice-id` 用内置名（默认 Xiaoyu 中文女声，共 18 个，`--tts-list-voices`
  看清单）或 `clone:<参考wav路径>` 零样本克隆（参考音频 3-10s 最佳）；moss 权重在
  voice0/.cache/moss/（voice0 `preload_moss.py` 一次性下载，voice0 只读不代管）。
- **一键启动脚本** `start_dialogue.bat [llm|agent] [vits|moss|melo] [额外参数...]`（sh 同理），
  五个快捷词**任意顺序**，其余参数原样透传（放在最后）。完整签名与组合见
  `docs/voice-dialogue.md`「一键启动脚本」；默认不传 = **llm + melo**；
  `start_dialogue.bat agent` = **agent** 接入（本地 claude 常驻会话，`--brain agent`），
  **若 `sessions/agent_session_id.txt` 已有历史则自动加 `--agent-resume` 续上次会话**，
  无则新建；llm 模式自动带 `--llm-config dialogue\config.local.json`（agent 模式不带）；
  `vits` = 换 vits 多音色、`moss` = 换 moss（CPU 实时 + 克隆）、`melo` = 显式 melo。透传例：
  `start_dialogue.bat agent vits --vad-tail 600 --tts-voice-id 可莉`。
  **默认参数（2026-09-14）**：一键启动固定带 `--agent-stream-tts --debug-tts`——agent
  流式增量送 TTS + 会话调试日志落 `sessions/`（排"说了 X 就卡住"全靠它）。两者都是
  `store_true` **无法命令行取反**，要关就改脚本删 `DEFAULTS` 里的参数（见
  `docs/voice-dialogue.md`「一键启动脚本」）。`--agent-stream-tts` 仅 agent 模式生效、
  LLM 模式被忽略（⑤d），`--debug-tts` 各模式都写日志。
- **参数含义白话版 + 快速开始 + 架构时序图**（vad-tail / post-commit-window / echo-guard /
  merge-window 的直觉 + 时间线 + 校准 + mermaid 线程时序）：见
  [`docs/voice-dialogue.md`](docs/voice-dialogue.md)。用户强调这些参数很难懂，解释时先讲
  直觉（"你停多久算说完""AI 答完但音频没播的空档""回声防护"），别只念数值。
- **文本输入源（调试，可选）** `--text-input-port <端口>`（默认关=原程序零变化）：起本地
  TCP 监听，`examples/text_input.py` 连入后 `while input()` 逐行输入，**与麦克风语音并存**
  ——不方便对麦克风说话时用它调试对话。**输出侧零改动**（控制台/音频/live2d 全走原逻辑）。
  **文本模式语义**（用户拍板）：唤醒词/退出词**无效**（当普通句子送 LLM/agent，不触发状态
  机）；打断词（默认"停下"）**整行完全等于**→ 立即 `hard_stop` 停当前输出（不进历史/LLM，
  无输出在途=no-op，**live2d 同步复位**=收说话框+表情回平和，与语音 KWS 打断一致）；
  休眠态输入问题**自动唤醒直接对话**（不播就绪语）；发一句可立即敲下一
  句（非阻塞），正在输出时敲下句立即打断重发（同语音 barge-in）。**打断语义比语音更强**
  （`feed_asr_sentence(..., barge_audio=True)`）：**只要 TTS 还在播就立即切掉**——哪怕 LLM
  已答完、仅剩音频在播也打断（用户实测"播放时输入下一问旧音频还在播"定位的差别）；语音定稿句
  默认 barge_audio=False **不**打断已答完音频（让回答播完、新回复排队）。注入点复用
  `ctrl.feed_asr_sentence(SentenceResult(...), barge_audio=True)`——与麦克风定稿句同构，
  barge-in/post-commit/历史/存档/live2d 全自动继承。实现 `dialogue/text_input.py`
  （TextInputServer + route_text_line，可 headless 测），客户端 `examples/text_input.py`，
  测试 `tmp/test_text_input.py`。详见 docs/voice-dialogue.md「文本输入源」。
- **LLM 模式工具（可选）** `--tools all|名字列表`（默认关=旧行为零变化，仅 `--brain llm`
  生效，agent 模式忽略走 claude 原生工具/MCP）：**XML 内联工具调用**（参照 Alife）——模型在
  输出文本流里写 `<get_weather city="北京"/>` 自闭合标签，`dialogue/toolparse.py`
  `ToolXmlParser` 字符级流式解析，标签闭合立即执行本地工具（工具包在**仓库根 `tool/`**：
  `@tool` 装饰器 + 包内自动扫描，**新增工具=丢一个 py 文件零改码**）；结果回灌成
  `[工具结果]` user 消息 → 第二轮流式出最终答案。**过渡句先出声**：LLM 路径 `_find_cut`
  不在语气词切句，工具标签捕获瞬间须显式把累积缓冲送 TTS，否则工具执行期用户听不到声音。
  参数 `--tools-max-rounds`（默认 3，防无限循环）/ `--tools-timeout`（覆盖默认超时）。
  第一批工具：`get_time`（零网络）/ `get_weather`（复用 assistant/qweather 技能直接
  HTTP 调，不走 MCP；**默认取整周 7 天**）/ `get_gold_history`（复用 assistant/gold
  数据管线直接 HTTP 调，不走 MCP；国内沪金 AU0 全历史统计 + 国际现货金实时，含免责声明；
  参照 soviet-joke 模式——确定性逻辑全在数据脚本，Tool 只薄封装不造数）。
  **工具包网络策略**：代码**不写死代理地址**；`load_tools()` 未显式配置 HTTP_PROXY/HTTPS_PROXY
  → 自动 `NO_PROXY=*` 绕过 Windows 系统代理直连（Clash 没开也能查国内源）；显式配了则尊重。
  **追问自动重查**：
  提示词明确"每轮都可调用、
  随时可再次调用"——用户追问新日期/新城市等旧结果没覆盖的信息时模型会重新调用工具，不
  硬答旧数据。**结果权威一次说清**：`[工具结果]` 注入消息与 system 都声明结果是权威事实、
  一次回答、不重复不编造（防单次输出重复两版自相矛盾，2026-09-16 实测）。
  headless 测试 `tmp/test_llm_tools.py`。完整设计见 `docs/llm-tools.md`。
- **机密**：DeepSeek API key 只放 `dialogue/config.local.json`（`.gitignore` 已排除，
  **绝不提交/绝不外传**）；读取优先级 显式参数 > `--llm-config` 指定文件 > 默认
  `config.local.json` > 环境变量 `DEEPSEEK_API_KEY`（`--llm-config` 可换整份配置）。
- **自定义系统提示词**：`--system-prompt <文件>`。系统提示词**永不压缩、永远放消息最前**
  （`_build_messages_locked` 把它作 system role，摘要拼在其后、历史之前）。
- **agent 大脑（可选）**：`--brain agent` 把大脑换成**本地 claude code 常驻会话**
  （`dialogue/agent.py` `ClaudeAgentClient`，claude-agent-sdk）——旁路自实现历史/压缩/
  系统提示词（上下文在 claude 会话，人格=`assistant/CLAUDE.md` 显式传 system_prompt，
  SDK 实测不自动加载 cwd 的 CLAUDE.md）；「停下」=ESC（abort 不 kill 进程，不进上下文）；
  敏感操作走 `【询问】` 语音确认（`_ASK_RE` 送 TTS 剥掉不念）；工具权限默认放行
  `PowerShell/Bash/Read/Write/Edit/Glob/Grep/WebFetch/WebSearch/Skill`（`agent.py`
  `_DEFAULT_ALLOWED_TOOLS`，SDK 无终端须预放行技能才能跑；**Windows 两个 shell 都能跑**——
  模型可能走 Bash（读了 SKILL.md 的 `bash fetch.sh`）也可能走 PowerShell，**当初只放行
  PowerShell 导致走 Bash 的会话报"脚本被拦住了"（天气查不到根因）**，故双 shell 都放行）；
  **MCP 工具自动放行**：启用 MCP 时按 `.mcp.json` 实际 server 名自动补 `mcp__<name>__*`
  白名单（只放行配置的 server，新增 MCP 无需改码）；`--no-mcp` 可整体不挂 MCP）；
  带独立 venv 的 MCP（如 search `free-search-mcp`）：`command` 写相对 assistant 目录的
  venv python（如 `.venv-search/Scripts/python.exe`）+ `args: ["-m", "search_mcp"]`——
  裸 `python` 会被替换成 voice-asr，独立 venv 的包找不到（实测 search MCP 挂载失败根因）；
  `-m` 后的参数是模块名不转绝对路径（`agent.py` 已支持）；
  `--agent-resume` 续上次会话
  （session_id 落盘 `sessions/agent_session_id.txt`）。默认 `--brain llm` 时现有 LLM 集成
  **零改动**。详见 `docs/agent-integration.md`。**写新 agent 代码注意**：partial 增量来自
  SDK `StreamEvent.content_block_delta.text_delta`（不是 AssistantMessage）；多轮 query 间
  须 drain 到 ResultMessage 再发下一条（打断残留消息会污染下一轮 receive）。
  **agent 延迟治理（2026-09-10 实测）**：① **默认关思考**（`max_thinking_tokens=0`）——
  模型走方舟 `ark-code-latest` 且 CLI 不识别（stderr 见 `unrecognized_model`）时按超大默认
  thinking 预算先"想"约 45s 才开口，实测同查询关 thinking 后 48.8s→1.9s；要思考质量用
  `--agent-thinking <预算>`（如 2048）。② **单回合看门狗** `--agent-query-timeout`（默认 90s）：
  receive 等 ResultMessage 超时 → 中断回合并报 "× LLM 出错：agent 超时"，**绝不无限挂起**
  （曾实测 resumed 会话被中断残留污染后静默 2-3 分钟无任何事件）。③ stderr 环形缓存 + 报错
  时 dump `[agent-cli]` 最近输出（不再全吞，诊断超时/报错可查）。④ `close()` 先直接 interrupt
  在途回合，**不留脏回合给下次 resume**（中断残留会污染下一轮 receive）——遇 agent 卡死/
  会话疑似被污染，删 `sessions/agent_session_id.txt` 换全新会话（病会话删除即弃）。
  ④b **「停下」abort() 同款直接 interrupt（2026-09-14 实测）**：`_worker` 在
  `await self._inflight` 阻塞期间**处理不了队列里的 ("abort",)**——旧队列式 abort 排不上队，
  在途 query 会**跑满整轮才作废**（实测 LLM 请求中喊"停下"4 次无效、回复慢 20s、结果
  DISCARD；224716 日志 4× hard_stop 后 query 仍跑到 217.563s）。修法：abort() 与 close()
  同款**直接 `_client.interrupt()`**（`asyncio.run_coroutine_threadsafe`，3s 超时吞异常），
  ESC 让 CLI 回合干净收尾——`_do_query` 的 `_drain` 收到终结后正常返回（无 ResultMessage
  → 不回调），worker 随 `await self._inflight` 返回继续下一个 query，**会话保留不杀进程**。
  headless 验证 `tmp/test_agent_abort_inflight.py`（A 复刻根因：队列式 abort 处理不到 →
  B 直接 interrupt 立即收尾 → C 下一 query 0.02s 出结果）。
  （⑤ **流式增量送 TTS** `--agent-stream-tts`，默认关：agent 只播最终结论意味着工具调用前
  的过渡句/思考段（实测「我把未来七天的天气捋一遍给你哈」）只显示不播、出声前干等工具
  7-8s；开此开关后流式增量也按句送 TTS——心态标记跨 delta 未闭合不切句、句中心态标记也
  作切点（"…哈【心态：开心】阿阳…"不粘成一句）、无标点缓冲以句末语气词（哈/哦/吧…）
  兜底切（"我再确认一下…哈"这类过渡句工具调用期间能先出声）；最终结论到达**三态收尾**：
  结论整段已进流式队列 → **不打断**自然播完（打断会把已入队未开播的结论音频全取消 →
  完全静音，2026-09-13 实测）；已播全是结论前缀 → 不打断只补送剩余；已播含过渡句/思考段
  → 立即 `tts.interrupt()` + 从 `_tail_overlap` 跳过已播开头重播（防"阿阳"整句播两遍）。
  **去重重叠只对"最后一个心态标记之后"的结论本体算**——ResultMessage 全文常以过渡句开头，
  对全文算会把过渡句误当已播结论跳过打断（2026-09-13 实测）。关=只播最终结论，旧行为零
  变化。
  ⑤b **过渡句静默兜底切 + 换行对齐去重（2026-09-14 实测笑话场景）**：① **idle 切句**——
  工具调用期 agent 长时间无流式增量，缓冲里以"你"等非语气词结尾的过渡句（"好，讲个新笑话
  给你"）没有标点/语气词切点，会**干等 3.7s 到结论才出声**（金价过渡句后跟 `\n` 能立即切、
  笑话这句卡死——差别就是有无边界符）。修法：`_agent_stream_thread` 把 `evt.wait()` 改成
  0.5s 分片睡循环，静默 ≥ `_AGENT_IDLE_FLUSH_MS`(1.5s) 且心态标记闭合 → 整段缓冲先送出声
  （`_idle_flush_agent_stream_locked`，过渡句计入 `_agent_tts_played` 供结论去重，不会播两
  遍）。② **`\n` 归一化**——流式切句在 `\n` **边界处切**（句子文本不含 `\n`），而
  `clean_full` 保留 `\n` → 两侧字符错位 → `_tail_overlap` skip=0 → 结论明明全量流式却误落
  branch c 取消+整段重播。修法：`_on_agent_result` 比较/去重前 `replace("\n","")`（`\n` 对
  TTS 发音无影响）。两个问题叠加就是"笑话过渡句打印了但音频等结论才播"（probe_joke_
  transition.py 复现，修后 submits=6 interrupts=0：过渡句 1.5s 内出声、结论自然播完）。
  ⑤c **流式每句控制台定稿行（2026-09-14 用户实测）**：agent 流式句送 TTS 时只走
  `on_ai_delta` 的**累计全文预览**（`con.update`），不触发 `on_ai_sentence` 定稿 → 屏幕
  残留"过渡句+结论拼一行、带省略号截断"（`_clamp` head…tail），且首句无 `[ts]` 时间戳。
  修法：agent 流式每句（`_flush_agent_stream_locked`/`_idle_flush_agent_stream_locked`
  统一走 `_announce_agent_sentence`）submit 后也调 `_on_ai_sentence`——与 LLM 路径
  `_emit_sentences` 一致，控制台每句一行完整定稿、首句带 `[ts]` 首答时刻；纯心态标记段
  （剥净后无可念内容）不刷行。配套：agent 流式模式 `on_ai_delta` **跳过累计预览**
  （每句已定稿，预览只造成拼行/省略号）；`_Console.finalize` 定稿行 `clamp=False` 完整
  显示（可折行，无省略号），`update` 实时预览仍截断防折行刷屏。验证 probe_joke_transition
  announced=6（过渡句+结论5句各定稿一行）。
  ⑤d **心态标记回定稿行 + `--agent-stream-tts` 只在 agent 模式生效（2026-09-14 用户实测
  `--brain llm --agent-stream-tts`）**：① 开关照读裸 `args.agent_stream_tts` 会让 LLM 模式
  `on_ai_delta` 把含【心态：xxx】的流式预览也跳过（该跳过本为 agent 流式每句定稿后防拼行，
  LLM 模式增量仍走预览）——修法：闭包变量 `bool(args.agent_stream_tts and agent is not None)`，
  LLM 模式传此开关=被忽略。② 定稿行只显示切出的句子文本、心态标记被 `_find_cut` 单独切走
  → 控制台首句无标记（曾见 `[98.91s] AI: 阿阳 那我讲一个哦`）。修法：`_pending_mood_announce`
  （实例属性，跨多次调用存活）攒纯标记段、拼回下一个真实句子显示；连续相同标记去重（agent
  结论开头自带标记+到达前已流式吐过同款 → 不查重拼成【心态：开心】【心态：开心】…）。TTS
  仍剥掉不念。**③ tail 直通路径是"第一轮带标记、二轮起消失"真根因（2026-09-14 用户实测）**：
  `_find_cut` 把句首标记切进 `_pending_mood_announce` 后，若整条回复无标点边界 → 不经过
  `_emit_sentences`、整个落 `_llm_loop` finally 的 `tail` 直通 `_submit_tts`/`_on_ai_sentence`
  —— 标记攒着却没拼回，控制台丢失。所有"切句→送 TTS→通知定稿"出口必须统一走
  `_with_pending_mood(sentence)`（`_emit_sentences` / `_announce_agent_sentence` / LLM finally
  tail / agent stream 收尾 tail 共 4 处，漏一处就丢标记）。验证 probe_llm_mood_console /
  probe_llm_mood_two_rounds（两轮首句都带标记）/ probe_joke_transition（announced 首句带标记、
  无重复）。**写新"切句+显示"路径记得：跨调用累计用实例属性、所有出口走 `_with_pending_mood`、
  别用函数局部变量。**
  ⑤e **结论全量流式重播 = LCS 占比判全覆盖（2026-09-14 鬼故事实测「好啊阿阳说两遍」）**：
  ResultMessage 全文含过渡句前缀（agent 只在开头带一次心态标记 → `concl_start` 掐不到过渡句），
  而过渡句+整篇都已在 played **开头**——`_tail_overlap` 只比 played **尾部**，匹配不上 → skip=0
  → branch c interrupt + 整段重播"好啊阿阳"两遍（日志 `debug_tts_20260914_201640.log`
  RESULT ctx=14 skip=0 branch=c，随后重播行重发同句）。修法：新增 `_overlap_ratio(a,b)`
  （最长公共子序列占比，一维滚动 DP，O(n·m) 每回合一次可忽略）——**内容保序、容忍流式切句/
  丢标点的中段错位**（同份文本中段 particle 切句吞句号、`。」` 独立段被丢，played 与 clean_full
  错位仍 ~0.7）。≥0.6 → 视为已全量播过，`skip=len(clean)` 落 branch a' 不打断不重播；
  <0.6 才 `max(_tail_overlap, _lcp)` 求补送起点（只流式了结论开头一点 = 真没播完要补送）。
  **`_tail_overlap` 只查 played 尾部是旧设计盲区：全量流式时重叠在 played 开头，必须补查 LCP /
  LCS 这类"前缀/任意位置"判据。** headless 验证 `tmp/probe_ghost_overlap.py`（interrupts=0、
  "好啊阿阳"只 submit 一次、连续标点已塌缩）。
  ⑤f **连续相同标点禁止送 TTS（2026-09-14 用户实测鬼故事"过去……"合成怪声）**：`_PUNCT_RUN_RE`
  `([。！？…～、；：，,—])\1+` → 单字符，在 `_clean_for_tts` 里统一塌缩（TTS 念重复标点不稳、
  无朗读意义）——**只影响送 TTS 的文本，控制台/历史/存档保留原文**。排查"TTS 怪声"先看送
  TTS 的文本有没有连续标点；写新"送 TTS"路径记得过 `_clean_for_tts`（心态剥除/括号/连续标点
  一次到位）。
  ⑥ **结论重播回声自屏蔽** `--replay-echo-guard-ms`（默认 1500）：**结论重播启动后短窗口内
  丢弃新 ASR 定稿句**——被 `tts.interrupt()` 切掉的过渡句尾音此刻还在房间里绕，重播刚起、
  门控 grace 又把回声喂进 ASR，VAD 闭成一条**幻影句**落在 post-commit 窗口内 → 把重播也打断
  （2026-09-14 实测「金价卡在'给你'、live2d 气泡冻在'最近两月…'」根因：GPU 上重播两句都在
  合成中，被打断后全部 stale 跳过 → 无音频 + SayTTS 链空退出气泡冻住）。只拦定稿句、不碰
  partial，窗口过后恢复正常；agent 结论三态收尾里只有"含过渡句→打断重播"那一支才设置。
  同款门控在 mic 采集层还有回声门控（`--echo-guard`，播放期只听"停下"），两层是不同窗口：
  回声门控拦"播放期间"的麦克风采集，本自屏蔽拦"重播刚起瞬间"已被 ASR 闭成的定稿句。）
  ⑦ **AI 自播期 KWS「停下」自屏蔽 —— 已停用（2026-09-14 用户实测回归推翻）**：
  原设计（防御性）：`--replay-echo-guard-ms` 只拦 ASR 定稿句，**拦不住 KWS 旁路**——回声
  门控播放期把 mic 喂给「停下」KWS（`ingest_kws_only`），当时怀疑 **AI 自己的音频会被 KWS
  自触发**（守卫 = `kws_guard_active()`，`_on_interrupt` 命中 `return` 跳过 `hard_stop`；
  agent-stream 模式每次提交 TTS 按估算播放时长顺延守卫覆盖流式/自然播放/重播全程）。
  **实测推翻（`sessions/debug_tts_20260914_222157.log`）：金价/鬼故事播放期的守卫命中
  （6~7 次）全是用户在重复说"停下"被吞**（间隔 2.5~12s，金价文本无 tíng xià 匹配音），
  用户实测"说了好多次都没反应"——**"播放期只听'停下'"的文档契约被打破**；AI 音频自触发
  "停下"在**所有真实日志零实例**（金价冻结真根因是 ⑧ branch c 尾差打断，非 KWS 自触发）。
  KWS 无法从声学区分"AI 回声"与"真·停下"，任何按播放时长的宽守卫都会连真"停下"一起吞。
  **停用 = 删 3 处 `_replay_kws_until` 设定点（_submit_tts / covered 分支 / branch c），
  `kws_guard_active()` 恒 False**——真"停下"随时生效（流式/自然播放/重播全程可打断）。
  若将来 AI 音频真自触发（phonetics 恰好匹配 tíng xià，概率极低），**正解是 AEC 回声消除**
  （从 mic 信号减喇叭参考），不是宽守卫。headless 验证 `tmp/probe_kws_stop_playback.py`
  （播放期"停下"→ hard_stop 触发：gen+1、interrupt 杀在播音频、agent abort、问题保留历史）。
  ⑧ **branch c 尾差打断 = 金价冻结真根因（2026-09-14，真实 debug 日志铁证
  `sessions/debug_tts_20260914_175020.log`）**：`--agent-stream-tts` 下 agent 流式吐完整
  结论 8 句（边吐边播，voice0 队列深：O1 刚开播、后面全在排队），最后一句"…不构成投资
  建议哦"按语气词"哦"切出、句末"。"留在流式缓冲没切出来。ResultMessage 到达（全文含尾
  "。"）→ `_tail_overlap(played, clean_full)` 返回 155/156（结论文本几乎全量已入队，只差
  尾"。"）→ 旧逻辑 `skip<len_clean` 落 **branch c** → `tts.interrupt()` 把**已入队未开播**
  的整条结论音频全取消，重播却只有 1 字"。"（`_find_cut` 吐不出 <2 字句、`_submit_tts` 丢
  纯标点）→ 重播为空 → **彻底静音（"说了阿阳就卡住"）**。`played` 累计的是**提交文本**，
  不是实际播放位置——队列深时它远超前于真实出声，branch c 据此打断就把正确音频打掉。
  修法（`controller._on_agent_result`）：**残尾 ≤ `_TRIVIAL_TAIL`（6 字）→ 视为已全量流式
  （branch a'），不打断**，让队列自然播完（残尾是纯标点/尾词，值不得为它打断；结论文本与
  流式一致时打断 = 白白杀掉已入队正确音频）。真正的 branch c（过渡句被结论打断+重播）不受
  影响——那是 `_tail_overlap` 返回小 skip 的场景（结论头与流式尾不重叠）。headless 验证
  `tmp/probe_real_freeze.py`（interrupt=0、被杀音频=0、结论 8 句全进队）。**排查"说了 X 就
  卡住"先用 `--debug-tts` 看 `RESULT branch=/interrupt=` 行**：`branch=c interrupt=True` 但
  `remainder` 只有 1-2 字 = 尾差打断（⑧）；`branch=a interrupt=False` 却仍冻结 = 另有其因
  （KWS「停下」宽守卫已停用（⑦），不再有 `KWS_GUARD` 覆盖问题——自播冻结先查回声门控/
  回音自触发，真因大概率是回声或 queue 时序，用 `--debug-tts` 看 `RESULT` 与 `SAY` 行）。
- **本地会话存档（默认开，仅 LLM 模式）**：`--history-dump` / `--history-dump-dir`（默认
  `sessions/`，已 gitignore）/ `--history-dump-interval`（默认 300s）。每周期把
  `ctrl.snapshot()` 的**完整对话状态**（system+summary+history+进行中内容）原子覆盖写到
  `session_<启动时间戳>.json`（每次启动新建；退出再写一次）。后台线程写盘，`snapshot()`
  锁内只浅拷贝、IO 在锁外——**不阻塞 LLM 线程**。**agent 模式不写**（历史在 claude 会话里，
  claude 自己管理，只落 `sessions/agent_session_id.txt` 供 `--agent-resume` 续）。
- **打断（barge-in）**：LLM 在途时来新 ASR 句 → `gen` 代际 +1 弃流（生成器 close 关连接），
  **重发本轮累计**（句1+句2…）；被作废的回复不 commit 历史。
- **停用词"停下"**：`--interrupt-words`（默认"停下"）。**双路打断**（2026-09-14 补齐，
  对齐唤醒词）：① 块级 KWS 旁路 `feed()` 命中 → `interrupt()`；② KWS 漏检但 ASR 定稿文本
  含打断词 → 引擎 `_process_sentence_locked` 兜底（paraformer 比 3.3M KWS 灵敏 ~10dB），
  同样 `interrupt()` 且**本句吞掉不进回调**。命中 → `on_interrupt` 回调 → 控制器
  `hard_stop()`：立即终止 LLM 流与 TTS 输出；**被打断的问题保留进历史**，"停下"本身吞掉、
  绝不进历史/LLM 输入。控制台打 `已停下` 状态行（语音打断可见反馈，2026-09-14）。
- **休眠/唤醒/退出状态机**（`--wake-word` 默认"小爱小爱"，逗号多词）：启动默认休眠——
  **双路唤醒**（`dialogue/wake.py` `WakeSession` 两态）：① 唤醒词 KWS（sherpa 3.3M）
  逐块 `feed()` 近场低延迟命中；② **睡眠态也喂 ASR 流式**，`on_sentence` 定稿句含唤醒词
  → 唤醒（`_do_wake`：`asr.interrupt()` 作废唤醒词残句 + KWS reset + 播就绪语）。根因
  （2026-09-12 实测）：KWS 灵敏度比 paraformer-large 低 ~10dB，40cm（SNR+5dB 分水岭）
  KWS 漏、ASR 定稿仍识别"小爱小爱"——故**唤醒以定稿句为准**（流式 partial 边缘下识别歪
  "答爱小"，flush 定稿才完整）。就绪/告别语走直连 `tts.submit`（不入历史），其 Job 作
  "自播回声"门控——自播期只喂"停下"（告别语播放期只喂 KWS，自播语音不进识别）。退出词
  （`--exit-words` 默认"拜拜"）在 `on_sentence` 入口拦截（照常显示但不进历史/LLM）；静默
  超时 `--inactive-timeout`（默认 60s）无用户语音 → 回休眠。`--wake-word ""` 关闭状态机
  （启动即对话旧行为），且**无唤醒检测器 → 永不自动休眠**（休眠后无唤醒途径=死机）。静默
  超时告别语经 `wake.consume_farewell()` 由调用方播
  （feed_decision 内部 go_sleep 的返回传不回调用方）。历史跨休眠保留（同次运行不清空，重启
  才重建存档）。退出词仅 AI 沉默时可说（播放期只听"停下"）。**打断词与唤醒词同款 KWS
  （3.3M）、40cm 同漏检风险；打断词已双路兜底（2026-09-14）**——块级 KWS 旁路 + 引擎
  `_process_sentence_locked` 的**定稿文本含词兜底**（paraformer 比 3.3M KWS 灵敏 ~10dB，
  远距 KWS 漏检但 ASR 仍听清"停下" → 按打断处理、本句吞掉不进回调）。KWS 漏检**且** ASR
  也听歪（"停下"→"评相"实测）仍会当普通句子提交——属声学极限，无法可靠兜底。完整对比见
  `docs/voice-dialogue.md`「休眠 / 唤醒 / 退出」的对比表格。详见
  `docs/voice-dialogue.md`「休眠 / 唤醒 / 退出」。
- **回声半双工门控（v1）**：TTS 播放期（`ctrl.tts_busy`）mic 只喂 `asr.ingest_kws_only()`
  （只听"停下"，回声不进识别 → 无反馈自答）；`--no-echo-gate` 关（耳机近场可用）。
  忙碌跟踪靠 voice0 `Job.done` + 守护 watcher 线程（voice0 无播放回调且不可改）。
  **门控开启有滚动 grace**（`--echo-guard` 默认 1200ms）：回声还没到（首句仍在合成）的
  窗口内，mic 块有语音能量（> -38dB）就顺延"仍喂正常识别"到 现在+静音尾+0.2s，让"AI 开答
  瞬间用户还没说完的尾巴"走完 VAD 静音尾定稿（否则被切去 KWS-only、悬成 partial 被吞）；
  回声一到由 `--echo-guard` 硬上限兜住不自答。voice0 无播放回调，音频"是否已开播"无精确
  信号，post-commit 窗口以首句提交时刻作代理。
- **首句 hold-off**：`--reply-hold`（默认 **0=关**）——旧方案，每轮回复首句先锁外延迟给
  续句打断窗口，代价是每轮首包音频固定 +N 秒。续句打断已由 post-commit barge 零延迟接管，
  hold 只保护首句边界后 N 秒内续句的极窄窗口，故默认关。
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
  `… `前缀=ASR 流式出字**未定稿**（不会提交）；`[ts-ts]`=定稿句已提交给 LLM，
  **`[ts] AI: …`=本轮首个 AI 回复句**也附时刻（**首 token/AI 开口时刻**，**锚在用户问题
  定稿时刻上**：`问题audio_end + (此刻 − 问题提交时刻)`——与用户句 [x.xx-y.yys] 天然
  同一坐标轴、不受引擎会话起点影响，曾见用 `asr.session_t0` 差出 ~35s 错位，锚定后
  不可能再跑偏；用开口时刻而非定稿时刻——agent 文本到齐与送 TTS 间可能有桥接延迟，
  定稿时刻会把它算进时间戳造成误读；同轮后续 AI 句不重复打时间；新用户句/撤答复重答
  时复位）；
  `→ LLM 请求中…`=LLM 请求已发出等首 token（controller `on_llm_start`，首 delta 原地覆盖）；
  `× LLM 出错`=流抛异常（`on_llm_error`）；`[门控]`=回声门控转换提示
  （AI 播放期 mic 只听"停下"，此刻说话不被识别——离远/音量低时 VAD 不闭句，句子
  "悬在流式 cache"永远不定稿，正是`… `行无后续的成因）。
- **live2d 桌宠联动**（`--live2d-port PORT`，可选）：**两通道**——① LLM 心态【心态：xxx】→
  发 desktop_pet 切表情（**随句子播放发射**：心态标记**文本到达即发** = 全部挤在 LLM 流结束的
  ~1s 里、音频却要播几十秒——表情全程卡最后一个标签（曾卡第一个）；改由 SayTTS 播放链在
  **携带该标签的句子实际开播**瞬间发射（说话框同款 `job.done` 时序：前句播完≈下句开播），
  表情跟听感同步切换。controller `_submit_tts` 提交前 `_leading_mood` 提取句首心态随 submit
  带给 SayTTS（无标签=继承当前不切；超纲词兜底平和；`_parse_mood_locked` 只维护 `_mood`
  状态判"没带标记"、不再直接发 on_mood），2026-09-19；LLM/agent 流式所有切句路径都收敛在
  `_submit_tts`，两模式统一生效。**句中心态不丢**：`_find_cut` 按标签**前**切（标签领衔下一句，
  `_leading_mood` 取句首首个心态才成立）——旧按闭合处切把句中标签粘前句尾巴、"我懂【心态：
  温柔】但你别硬加…"的温柔随前句 submit 丢失）；② **所有送 TTS 的文本**→ 角色头顶说话框（对话回复/就绪语/告别语/
  启动问候一个不落，靠包一层 `SayTTS`（dialogue/say_tts.py）的 tts 代理自动 say，controller
  零改动）。**说话框逐句链式跟播不抢发**：voice0 queue 提交即入队、串行播放——LLM 一口气吐
  3 句时 3 个 Job 瞬间入队、音频还在播第 1 句；若 submit 时就发文本，气泡会被末句立刻刷新。
  SayTTS 让第 1 句提交即发、之后每句等前一句播完（job.done）才发（prev-done≈下句开播），
  气泡永远显示"正在播的那句"；被打断（hard_stop，job.canceled）→ 作废句文本丢弃。
  **启用三级门槛** = 心态标记开 + 给了端口 + **启动测活成功**——连上瞬间补发一条恢复初始
  状态 `{"emotion":null,"say":null}`（清桌宠遗留表情/气泡）；连不上打印一条
  `[live2d] 表情联动关闭…`并**彻底禁用不重试**。运行中 live2d 退出 → **继续发送** + 每次
  失败打印"live2d server 连接失败，请检查"（桌宠可能重启回来，不做自动停用）。复位 =
  **一条组合消息**（表情回平和 + 收框），触发点：初始化测活 / 拜拜 / "停下"打断 / 静默超时
  回休眠 / Ctrl+C 退出（无条件），以及**一轮播放真正播完**（`--live2d-idle-reset` 默认开，
  `--no-…` 关；判定用 controller 新增 `turn_active`=LLM 流在途或 TTS 队列非空，防句中停顿
  误收框）。协议 = 原始 TCP 127.0.0.1:PORT 一行 JSON、UTF-8+\n、无响应；**常驻长连接 +
  惰性重连**（worker 持一条 socket，发送失败关旧重建重发一次，不探测/无心跳——live2d 重启
  下次发送自动连上；断线后首条可能丢、次条重建送达）；16 心态与 live2d
  EMOTIONS 键**恒等映射**；只发 emotion/say 不碰 mouth（嘴由 live2d `--listen` 对口型负责）。
  实现 `dialogue/live2d.py` `Live2dEmitter`——`emit()`/`say()`/`reset()` 只在锁内入队
  （微秒级**不阻塞 LLM 线程**），daemon worker FIFO 串行发送；回休眠复位挂 `wake.go_sleep()`
  的 `on_sleep` 回调（bye/timeout 唯一汇聚点），"停下"挂 `asr.on_interrupt` 组合回调
  （hard_stop + reset）。headless 测试 `tests/test_live2d.py`（假 TCP server 断言
  emotion/say/reset 送达保序/测活失败禁用/bye/timeout 复位归位）。
  详见 docs/voice-dialogue.md「live2d 桌宠联动」。
- headless 测试：`tests/test_wake.py`（**已入库**）——WakeSession 状态机 + sherpa 关键词
  文件唯一化（唤醒/退出/超时/自播门控/幂等）；`tmp/test_dialogue.py`（gitignored）——fake
  LLM/TTS 覆盖切句/barge-in/hold-off/hard_stop/busy/压缩/引擎钩子（`RealtimeASR.__new__`
  绕过模型加载）。都用 `voice-asr`（或共用 `voice-tts`，见「环境硬性约束」）环境跑。

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
                 controller.py（DialogueController：barge-in/hard_stop/tts_busy/hold-off/历史压缩；
                   agent 参数=agent 模式旁路历史/压缩/系统提示词，【询问】送 TTS 剥掉不念）/
                 agent.py（ClaudeAgentClient：--brain agent 大脑，claude-agent-sdk 常驻会话，
                   abort()=ESC、session_id 落盘、StreamEvent 流式出字）/
                 mic.py（MicAGC 自适应增益+噪声门控/check_mic_signal/pick_input_device）/
                 wake.py（WakeSession：休眠/对话两态状态机，唤醒/退出/静默超时，纯逻辑可测）/
                 live2d.py（Live2dEmitter：心态→表情 + 全 TTS 文本→说话框，启动测活+组合复位）/
                 say_tts.py（SayTTS：tts 代理，文本→说话框逐句链式跟播+一轮播完复位，纯逻辑可测）/
                 text_input.py（TextInputServer：--text-input-port 文本输入源 TCP server +
                   route_text_line 纯路由：休眠自动唤醒/打断词整行/唤醒词·退出词无效，可测）/
                 toolparse.py（ToolXmlParser：--tools 的 XML 流式解析器，Alife 移植，
                   feed→(clean,calls)，跨 delta 拆分/透明容器/实体/注释，可测）/
                 config.local.json（机密 API key，gitignored，绝不提交）
tool/            LLM 模式工具包（--tools，docs/llm-tools.md）：base.py（Tool/@tool/超时守卫）+
                 __init__.py（pkgutil 自动扫描，新增工具=丢一个 py 文件零改码）+
                 time_tool.py（get_time 零网络）+ weather.py（get_weather 复用 qweather 技能）+
                 gold.py（get_gold_history 复用 assistant/gold 数据管线，参照 soviet-joke 模式）
assistant/       agent 大脑工作目录（cwd）：CLAUDE.md=人格（显式传 system_prompt）/ .mcp.json+skills/=能力；
                 独立 git 子模块（GitHub 私有仓库，凭据不入库），设计目标/目录结构见 docs/voice-dialogue.md「assistant/ 目录」节
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
- `docs/voice-dialogue.md` = **语音对话使用 + 架构**（快速开始含推荐 `--vad-tail 300`；自定义系统提示词 `--system-prompt`；会话历史存档 `--history-dump`；参数白话解释 vad-tail/post-commit-window/echo-guard/merge-window 的直觉、时间线、为什么 post-commit 是时间窗、校准表；mermaid 线程时序图 + 阻塞/非阻塞说明）。
- `docs/agent-integration.md` = **本地 agent 接入设计（方案稿，随实现更新）**（`--brain llm|agent` 开关、常驻 claude 会话、agent 模式旁路历史/压缩/系统提示词、打断=ESC、权限【询问】交互、重启续会话、文件级改造清单、技术风险）。
- `docs/llm-tools.md` = **LLM 模式工具调用设计（已实现，2026-09-16）**（`--tools` XML 内联方案：tool/ 注册表 / toolparse 解析器 / _llm_loop 多轮流 / 安全阀 / 首批工具 / 落点与验证）。
- `docs/backend-guide.md` = **新增后端接入指南**（流式/非流式后端契约、三步接入清单、引擎消费语义、验收纪律，接 SenseVoice 等新模型时先读）。
- `README.md`「引擎设计」（T6 后落地）= RealtimeASR 完整设计（一分钟上手）。
- `docs/ai-project-methodology.md`（在 voice0 仓库） = 本项目沿用并沉淀的 **AI 项目全流程方法论**，可复用。

## 协作习惯（本项目）

- 与用户用**中文**交流。
- 里程碑式改动后用户会说「commit吧」——按其节奏提交，commit message 用中文。
- **验收纪律**：每个后端/改动必须跑 CER 内容级复核，不要只报 RTF/延迟数字。
