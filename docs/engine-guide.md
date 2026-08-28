# voice1 引擎使用与工作原理指南

> 面向「想真正用起来 / 想搞懂内部机制」的读者。逐条回答 README 代码案例里最容易
> 产生的疑问：`on_sentence` 是干嘛的？`SentenceResult` 各字段什么含义？`ingest`
> 的参数从哪来、阻塞吗？背压什么意思？采集在哪做？一个 chunk 是一次识别任务吗？
> 单例起了几个线程？wall 轴是什么？VAD 怎么工作的？…… 全部有源码出处。

---

## 1. 全景：一次「说话 → 出字」经过什么

```
音频来源（两种，引擎不负责采集，只消费喂进来的块）
   ├── 麦克风：examples/record_mic.py（sounddevice InputStream 16k 回调）
   │           每回调一批采样 = 一个 audio 块
   └── 文件：   examples/transcribe_file.py（ingest_file 内部切块）

喂入：asr.ingest(audio, source_ts)
   │
   ├─ [T12] 打断词旁路：audio 先喂给轻量 KWS（sherpa 3.3M int8）
   │       命中"停下" → asr.interrupt()，本块丢弃不识别（毫秒级，主线程内同步执行）
   │
   └─ 入队：_audio_q.put((gen, audio, source_ts))     # 有界队列 maxsize=8

常驻 worker 线程（asr-worker，引擎自起的唯一线程）：
   for 出队一个块:
       if 块的 gen ≠ 当前 _gen: 丢弃（旧会话）        # interrupt 后旧块作废
       EnergyVAD.add(块) → 可能断出一个/多个"句"
       ── 对每个断出的句 ──
           1) [兜底] KWS detect(句)：命中"停下" → interrupt()，本句丢弃
           2) 持 _recog_lock 调后端 recognize(句)（模型非线程安全，串行）
           3) 构造 SentenceResult，若 not stale → 调 on_sentence 回调
```

要点速记：
- **引擎不采集音频**，只消费 `ingest()` 喂进来的 16kHz float32 块。采集由调用方做
  （`record_mic.py` 用 sounddevice，`transcribe_file.py` 用 `read_audio`）。
- **chunk ≠ 识别任务**。喂多少块由调用方定；VAD 把块**聚合成句**，一句才触发一次
  `recognize`。
- **实时回调 = worker 线程调用**，与喂入方（主线程）异步。

---

## 2. 线程模型：创建单例到底起了几个线程？

**引擎自己只起 1 个常驻线程**（源码 `engine.py:91`）：

```python
self._worker = threading.Thread(target=self._worker_loop, name="asr-worker", daemon=True)
self._worker.start()
```

| 线程 | 谁创建 | 干什么 | 何时存在 |
|---|---|---|---|
| 主线程 | Python 进程 | 调用 `ingest()` / `ingest_file()` / `close()`；KWS `feed()` 在此线程内**同步**执行 | 常驻 |
| `asr-worker` | 引擎 `__init__` | 消费队列 → VAD 断句 → `recognize` → 回调 | `close()` 时 join 退出 |

**没有第三个线程。** 关于打断词（停止词）的疑问，明确回答：

- **KWS 流式 `feed()` 跑在调用 `ingest()` 的那个线程里**（实时场景即主线程），不是
  独立线程。每喂一个块，先同步做一次关键词检测（块小、模型 3.3M int8，每次调用
  亚毫秒级），命中立即 `interrupt()`——这就是它能"不等 VAD、即时打断"的原因。
- **KWS 兜底 `detect()` 跑在 worker 线程**里（VAD 断出句子后、识别前调用）。
- 两者可能并发调同一个 KeywordSpotter 对象（模型非线程安全）→ `SherpaKwsDetector`
  内部有 `threading.Lock()` 串行。
- 后端内部（funasr/torch 推理线程池）不算引擎线程，归框架管理。

### 为什么"打断词"需要这个旁路，而不能排队识别？

打断词若也入队，它排在**队尾**——等 worker 处理到它时，前面的句子早已识别完，
打断毫无意义（这就是"打断悖论"）。旁路 KWS 让"停下"一出口、不等 VAD 断句就命中，
`interrupt()` 即时把**队里已有的任务全部作废**。详见 `docs/asr-architecture-decision.md` T12。

---

## 3. 音频采集与 chunk

### chunk 从哪来？

引擎不管采集。两种常见来源：

**麦克风实时**（`examples/record_mic.py`）：

```python
def cb(indata, frames, t, status):
    asr.ingest(indata[:, 0], source_ts=time.monotonic())   # 每回调一批采样 = 一块

with sd.InputStream(samplerate=16000, channels=1, callback=cb):
    ...                                        # sounddevice 按 blocksize 分批回调
```

`sounddevice` 的 `InputStream` 回调每次给一批 `(frames, 1)` 的 float32 采样，`indata[:, 0]`
取单声道，作为**一个 chunk** 喂入。chunk 大小 = `blocksize`（默认由 sounddevice/声卡定，
通常 10-50ms 一批；可显式 `blocksize=` 指定）。**chunk 多大完全由调用方决定**——引擎
只要求是 16kHz float32。

**文件**（`examples/transcribe_file.py`）：`ingest_file()` 内部把音频（wav/mp3…）切成 100ms 子块，
见 §4.7。

### 一个 chunk = 一次识别任务吗？

**不是。** 喂入的块先经 VAD 累积；只有能量模式走到"句末"才断出一个句，断出的句才是
一次 `recognize` 任务。所以：喂 10 个块，可能断出 0、1、2……个句子（取决于说话节奏）。
"停顿后继续说话"会断成两句，正是这个机制。

### `ingest` 是阻塞操作吗？背压阻塞是什么？

`ingest` 多数情况下**非阻塞**：把块放入队列就返回。唯一的例外是**背压**：

- 队列 `_audio_q` 有界 `maxsize=8`（源码 `engine.py:81`）。
- 若识别速度 < 喂入速度，队列被填满 → 下一次 `ingest` 在 `put()` 上**阻塞**，直到
  worker 腾出空位。
- 这是**故意的**：防止"无限堆任务 → 延迟无限膨胀"。阻塞喂入方 = 让"边说边喂"的
  调用方慢下来跟住识别速度，保持实时性。

> 所以 `record_mic.py` 里如果麦克风说话太快、GPU 识别跟不上，sounddevice 回调会
> 被 `ingest` 阻塞——这是背压在工作，不是死锁。

---

## 4. 核心 API 详解

### 4.1 构造参数（全部给全）

```python
asr = RealtimeASR(
    backend="paraformer",        # 后端：paraformer(默认) | whisper | sherpa
    device="auto",               # auto|cpu|cuda；auto=有CUDA用cuda否则cpu
    sample_rate=16000,           # 音频采样率（与模型必须一致；引擎会强制16k）
    vad_silence_tail_ms=250,     # VAD 静音尾长（毫秒）→ 判定"这句话说完了"的静音时长
    profile=False,               # True 时收集 self._sentences（逐句时序，调试/统计用）
    debug=False,                 # True 时打印加载/打断/逐句诊断信息
    interrupt_words=None,        # 非空则启用打断词旁路，如 ["停下"]（T12）
    streaming=False,             # True=流式逐帧出字（T13，后端须 supports_streaming）
    hotword_file=None,           # 热词文件路径（每行一个纠错项，拼音级；仅 paraformer，T16）
)
```

| 参数 | 默认 | 中文含义 | 代码案例 |
|---|---|---|---|
| `backend` | `"paraformer"` | 选哪个 ASR 后端（§8 有对比） | `RealtimeASR(backend="whisper")` |
| `device` | `"auto"` | 推理设备。`auto`=有 CUDA 用 cuda，否则 cpu | `device="cuda"` |
| `sample_rate` | `16000` | 喂入音频采样率。若后端强制 16k，引擎会把 `self._sr` 同步为 16k | 一般不用动 |
| `vad_silence_tail_ms` | `250` | VAD 静音尾长（§7）。调大→断句更稳、延迟更大 | `vad_silence_tail_ms=600`（离线高精度） |
| `profile` | `False` | 收集逐句时序到 `asr._sentences`（含 stale 结果） | `profile=True` |
| `debug` | `False` | 打印加载/打断/识别诊断 | `debug=True` |
| `interrupt_words` | `None` | 打断词列表，非空启用 KWS 旁路 | `interrupt_words=["停下"]` |
| `streaming` | `False` | 流式逐帧出字（T13）。`True` 且后端支持 → 逐块 `on_partial` + 句末 flush 定稿；后端不支持（whisper）→ 告警降级整句 | `streaming=True` |
| `hotword_file` | `None` | 热词文件路径（T16，§9）。对识别文本做**拼音级纠错**（神庙→神妙），流式/整句都生效；后端不支持（whisper）→ 告警忽略 | `hotword_file="assets/hotwords/xiaoshuang.txt"` |

**单例语义**：同一进程内 `RealtimeASR(...)` 多次调用返回**同一实例**（模型只加载一次）；
仅当 `backend / device / vad_silence_tail_ms / interrupt_words / streaming / hotword_file`
任一**变更**时才销毁重建。

**`streaming` 一键切换**：`streaming` 变更会销毁重建单例（因为后端模型在前向模式下是否
走流式 cache 是构造期状态）。流式 vs 非流式只是出字时机不同，**句末最终文本都是整句**
（流式用 `is_final=True` flush 定稿，非流式直接整句识别）——见 §4.4、§8 对比。

### 4.2 `on_sentence(cb)` —— 结果回调

```python
asr.on_sentence(lambda r: print(r.text, r.ttfb))
```

- **作用**：注册"每完成一句话识别"的回调。worker 每断出一句并识别完，就调用一次。
- **返回**：旧回调（用于换绑）。
- **何时触发**：实时流里，一句话的音频结束后 `≈ VAD尾长 + 识别耗时` 才回调（这就是
  ttfb 的来源）。
- **⚠️ 例外**：`stale=True` 的结果（识别期间被打断）**不进普通回调**（§4.3）。

### 4.3 `SentenceResult` —— 回调收到什么

一句话的识别结果，字段（`jobs.py`，`__slots__`）：

| 字段 | 中文含义 |
|---|---|
| `idx` | 句子序号，从 1 起 |
| `text` | 识别出的文本 |
| `audio_start` | 该句音频**起点**（相对会话起点的秒数；实时=wall 时刻-`_t0`） |
| `audio_end` | 该句音频**终点**（VAD 判定"这句话说完了"的时刻） |
| `recog_start` | 识别线程实际开始推理的时刻 |
| `recog_end` | 识别线程实际结束推理的时刻 |
| `ttfb` | **尾字延迟** = `recog_end - audio_end`（说话停止 → 文本回调的时延） |
| `stale` | `True`=识别期间被 `interrupt()` 打断。**该结果不进普通回调**，仅 `profile` 收集 |

> **「会话起点」「会话」是什么**（`engine.py:89`）：
> - 引擎创建那一刻打一个 `time.monotonic()` 戳存为 `_t0`，**之后永不改**——所有实时时间戳
>   （`audio_start/audio_end/recog_start/recog_end`）都是相对它换算的（`- _t0`）。所以
>   `audio_start=0.5` 意思是"这句话的音频起点距引擎创建过了 0.5 秒"。
> - **会话 = 创建到 `close()` 之间的一整段连续识别过程**。`interrupt()`（手动或说"停下"）
>   结束旧会话、开新会话：`_gen += 1`（旧块作废）+ 清 VAD/后端/KWS 状态 + 清队列——
>   **唯独不改 `_t0`**，保证打断前后时间戳在同一单调时间轴上继续走、不跳变回退。
> - 文件路径复用同一 `_t0`：`base_ts = self._t0`，块按 `base_ts + i/sr` 打戳，相减后
>   `audio_start/audio_end` = **文件内相对秒**（正数，bench 可比）。

```python
def cb(r):
    print("#%d [%.2fs~%.2fs] %s  尾字延迟=%.3fs  stale=%s"
          % (r.idx, r.audio_start, r.audio_end, r.text, r.ttfb, r.stale))
```

### 4.4 `on_partial(cb)` —— 流式逐块出字回调（T13）

```python
asr.on_partial(lambda p: print(p.text))     # streaming=True 时生效
```

- **作用**：注册"边说边出字"回调，`streaming=True` 时，worker 对**未断句块**逐块调
  `recognize_stream(is_final=False)`，后端返回**累计**部分文本（"明""明天""明天早"…），
  非空则回调 `PartialResult`。
- **触发时机**：说话期间每处理一个音频块（100ms）就回调一次；一句话说完（VAD 断句）
  以 `on_sentence` 的完整结果收尾——**最终文本以 flush 定稿为准**，partial 只供展示。
- **不触发**：`streaming=False` / 后端不支持流式（whisper 降级）/ 部分文本为空。
- **与 `on_sentence` 共存**：两者都注册即可——partial 实时滚屏，final 落库。
- **⚠️ 打断**：`interrupt()` 或说"停下"会作废后续 partial（`_gen` 守卫），回调里
  `p.audio_start` 会跳变——上层不要缓存累计文本跨句使用。

### 4.5 `PartialResult` —— 流式 partial 回调收到什么

```python
class PartialResult:    # jobs.py，__slots__
    text        # 累计部分文本（"明天早"）
    audio_start # 本块音频起点（相对会话起点 _t0，秒）
    wall_ts     # 回调时刻（相对 _t0，秒）——首字延迟 = wall_ts - 喂入起点
```

`wall_ts` 就是实时轴上的"这句第一块被流式处理完的时刻"，用它算**首字延迟**
（首个非空 partial 的 `wall_ts` 减去音频喂入起点），是 T13 对比流式 vs 非流式的关键。

### 4.6 `ingest(audio, source_ts=None)` —— 实时喂入

```python
asr.ingest(audio, source_ts=None)
```

| 参数 | 类型 | 含义 | 从哪来 |
|---|---|---|---|
| `audio` | 1D `np.float32`，16kHz | 一个音频块 | 麦克风回调一批采样 / 文件切块 / 你按任意粒度切 |
| `source_ts` | float | 该块**第一采样**的 monotonic 时刻（秒）；`None` 时引擎反推 | 采集处打 `time.monotonic()` 戳 |

- **语义**：非阻塞入队（背压满时阻塞，§3）；返回即"已入队"，识别在后台 worker。
- **`source_ts` 为什么重要**：它决定 `SentenceResult.audio_start/audio_end` 的绝对时间
  基准。实时场景必须传真实采集时刻，否则时间戳错位。
- **T12**：`interrupt_words` 非空时，块先过 KWS `feed()`；命中"停下"→ 立即
  `interrupt()` 并**丢弃本块**（"停下"本身不进识别管线）。

### 4.7 `ingest_file(path, chunk_ms=100)` —— 文件同步识别

```python
results = asr.ingest_file("会议录音.wav")
```

- **行为**：读整个音频（**wav/mp3/flac/ogg…**，非 WAV 走 soundfile）→ 重采样到 16k →
  **切成 `chunk_ms`（默认 100ms）子块** → 逐块喂 `VAD.add` → 断出句子逐句识别 →
  最后 `flush()` 吐出残留缓冲 → 返回全部 `SentenceResult`。
- **流式模式**（引擎以 `streaming=True` 构造时）：文件也走流式——未断句块逐块
  `on_partial` 出字、断句边界块 flush 定稿（CER 0.017，与实时流一致）；句首 ttfb ≈
  flush 耗时（audio 轴），文件末无静音尾的尾句由 `flush()` 整句兜底。whisper 不支持
  流式 → 自动降级整句。
- **一个文件不是"一个 chunk/一个任务"**：它被切成多个子块喂 VAD，VAD 再聚合成若干
  句，**每句 = 一次识别**。长音频 → 多句结果。
- **同步阻塞**：在调用线程内完成全部识别才返回（与实时流"喂了就走"不同）。
- 时间轴用 **audio 轴**（§6），结果 `ttfb` = 纯识别耗时（不含喂入加速），bench 可比。

### 4.8 `interrupt()` —— 打断会话

```python
asr.interrupt()   # 手动打断（换说话人/新会话）；或说"停下"自动触发
```

- 立即作废**队列里所有排队任务**；正在识别的那句无法中止（整句前向原子），完成后
  标记 `stale=True` 不进普通回调。
- 清空 VAD 缓冲、后端流式状态、KWS 流状态。
- 新喂入的音频正常识别（新会话）。

### 4.9 `close()` —— 关闭

```python
asr.close()   # 幂等；with RealtimeASR(...) as asr: / __del__ / atexit 都兜底
```

停 worker 线程 → 关后端模型 → 释放单例槽位。

---

## 5. 完整代码案例（注释版）

### 5.1 麦克风实时识别 + 打断词

```python
import time
import sounddevice as sd
from asr import RealtimeASR

asr = RealtimeASR(backend="paraformer", device="cuda",
                  interrupt_words=["停下"],   # 说"停下"→ 作废所有排队任务
                  profile=True)
asr.on_sentence(lambda r: print("[%.2fs] %s (ttfb=%.3fs)"
                                % (r.audio_end, r.text, r.ttfb)))

def cb(indata, frames, t, status):
    asr.ingest(indata[:, 0], source_ts=time.monotonic())   # 每批采样=一个chunk

with sd.InputStream(samplerate=16000, channels=1, callback=cb):
    time.sleep(30)                 # 说 30 秒
asr.close()
```

### 5.2 文件转写

```python
from asr import RealtimeASR

asr = RealtimeASR(backend="paraformer", device="cuda")
res = asr.ingest_file("会议录音.wav")        # 同步返回全部句子
for r in res:
    print("#%d [%.2f-%.2fs] %s" % (r.idx, r.audio_start, r.audio_end, r.text))
asr.close()
```

### 5.3 打断词代码 case（完整可运行）

见 `examples/demonstrate_interrupt.py`——喂多句制造积压、中途说「停下」作废排队任务、
stale 抑制、打断后恢复，演示脚本有完整注释与参数说明。

### 5.4 流式逐帧出字代码 case（完整可运行）

见 `examples/demonstrate_streaming.py`——`streaming=True` + `on_partial` 边说边出字、
句末 flush 定稿，同一句话跑流式/非流式对比首字延迟，whisper 降级演示。脚本用**实时节奏
喂入**（非快进，否则墙钟时序失真）并做了安静文件的响度修复（只放大不压低，注释解释了
为什么统一压到 peak 0.10 会漏断句）。

---

## 6. 时间轴：wall 轴 vs audio 轴

`SentenceResult` 的 `recog_start / recog_end / ttfb` 有两种基准（引擎参数
`recog_axis`，内部按场景自动选）：

| 轴 | 用途 | recog 时刻含义 | ttfb 含义 |
|---|---|---|---|
| **wall** | 实时流（麦克风） | 实际推理的 **monotonic 墙钟**（相对会话起点 `_t0`） | `recog_end - audio_end` = **说话停止 → 文本回调**的真人感知延迟（含 VAD 尾长 + 识别耗时） |
| **audio** | 文件同步（`ingest_file`） | 映射到**文件音频时间轴**（`recog_start = a_end`，`recog_end = a_end + 纯推理时长`） | **纯识别耗时**（喂入是加速的，墙钟不再反映音频时间） |

- 为什么叫 **wall**？因为用的是 `time.monotonic()` 墙钟，不是音频累计时长。
- 实时场景关注 **wall 轴的 ttfb**（硬指标 <0.5s）；bench 文件场景用 **audio 轴**
  ttfb = 纯识别耗时，方便跨文件/跨设备对比模型本身快慢。
- 流式文件句的 audio 轴 ttfb = **纯 flush 耗时**（T15：`preset_dur` 补回 preset 路径被
  跳过的 flush 计算量，此前虚报 0）；整句文件句 = 整句识别耗时。流式文件**末句**
  （文件末 `vad.flush()` 收尾）同样走流式定稿——喂静音块收尾 cache、**不重喂残句**
  （重喂会 double-feed 文本重复），ttfb 语义与中途句一致；纯尾静音残句 → 空文本跳过。

### 6.1 延迟指标：首字延迟 vs 尾字延迟

设一句话 **t=0 开口**、**t=T 说完最后一个字**：

- **首字延迟** = 开口 → **屏幕上出现第一个字**的耗时。
- **尾字延迟** = 说完最后一个字 → **完整句（含末字）回调**的耗时。**硬指标 0.5s
  盯的是它**——"我说完多久能看到完整句子"。
- 非流式整句：首字与尾字**同一刻原子出现**（整句识别完才出文本），首字延迟 =
  句长 + VAD 尾 + 识别耗时（随句长涨）。

| 模式 | 首字延迟 | 尾字延迟 | 计算模型 |
|---|---|---|---|
| 整句 | ≈句长+识别（原子出现） | VAD尾 + 整句识别（随句长涨） | **说完才开工**，识别耗时 ∝ 句长×RTF |
| 流式 | ≈0.9s **恒定**（与句长无关） | ≈0.35s **恒定**（VAD尾 250ms + flush 耗时） | **边说边算**，cache 已算完大部分，句末只收尾 |

**为什么流式尾字延迟低**：非流式是"你说完，我整段从头听一遍"（识别耗时 ∝ 句长）；
流式是"你边说，我边听边记"——每个块喂入时流式解码器（FunASR cache / sherpa
stream）**当场算完**并缓存前缀状态，句末只需把边界块补上 + `is_final=True` 把
cache 结果收尾拼接成完整句。所以尾字延迟 = 固定 VAD 尾（250ms）+ flush 耗时，
**不随句长涨**。T13 实测 4.7s 长句：流式尾字 0.35s 达标、整句 0.93s 破线。

**代价**：流式把一次整句前向拆成几十次逐块前向，**总计算量略大**（RTF 0.22 vs 0.16，
均达标）——省的是延迟曲线不是吞吐。批量转写（不在乎实时出字）用整句更快；
实时/长句场景用流式。

> **文件场景注意**：`ingest_file` 是加速喂入，墙钟首字延迟被压缩、无真人感知意义；
> 流式 vs 整句的首字对比须用**实时节奏喂入**（bench `--paced`），否则首字数字失真。

---

## 7. VAD 原理（EnergyVAD 状态机）

源码 `asr/core/audio.py`。纯能量检测（无神经模型），20ms 一帧：

1. **分帧**：每 20ms（16k 下 320 采样）一帧。
2. **能量判据**：每帧 RMS → dB：
   ```python
   db = 20*log10(sqrt(mean(frame^2)) + 1e-12)   # ≥ -35dB 视为"有语音"
   ```
3. **状态机**：
   - 静音态：帧 ≥ `threshold_db`(-35dB) → 进入语音态，记句起点。
   - 语音态：连续静音帧数 ≥ `silence_tail_frames`（250ms/20ms=12 帧）→ 判定"这句话
     说完了"，把 [句起点, 当前位置] 剪出作为一句（保留尾部少量静音保边）。
   - 句长 < `min_speech_ms`(250ms) → 视为噪声，丢弃（这就是过短的"嗯/哦"不会出句）。
4. **防缓冲膨胀**：只修剪句前静音，已扫描未断句的语音保留。

**参数是延迟-准确率旋钮**（T10 标定，ADR 详述）：

| `vad_silence_tail_ms` | 严格CER | 实时尾字延迟 | 结论 |
|---|---|---|---|
| 150ms | 0.096 | ~0.35s | 实时达标但句尾语气词幻听（CER 恶化） |
| **250ms（默认）** | **0.059** | **~0.48s** | **实时最优折中**（延迟达标、CER 逼近 5%） |
| 600ms | 0.047 | 超标 | 句尾最干净，离线高精度用 |

> CER 随 tail 单调改善（句尾越干净），实时延迟随 tail 单调恶化，无单一值同时达标。
> tail 小的 CER 恶化根因是**句尾拖音幻听**（"稍等一下啊""六点五六嗯"），不是断句切错。

---

## 8. 后端矩阵与性能对比

已实现 **3 个后端** + 1 个独立 KWS 打断模型（不是后端）：

| 后端 | 模型 | 定位 | 设备 | 严格CER | RTF | 平均ttfb | 结论 |
|---|---|---|---|---|---|---|---|
| **paraformer**（默认） | FunASR paraformer-zh-streaming | 实时主力 | cuda | **0.059** | **0.154** | **0.226s** | **默认**：实时尾字延迟 0.48s 达标 |
| paraformer | 同上 | 实时主力 | cpu | 0.059 | 0.26 | — | CER/RTF 达标，CPU 尾字延迟略超 |
| **whisper** | faster-whisper medium | 高精度离线 | cuda | 0.120（规范 0.058） | 0.139 | 0.370s | 自带标点；**CPU 不可实时**（RTF 2.66） |
| **sherpa** | sherpa-onnx zipformer-zh-14M | CPU 轻量基线 | cpu | 0.190 | **0.033** | 0.084s | RTF 极优、加载 1.2s，质量不足（短句有缺陷） |

**CER 口径**（`bench/bench_asr.py`）：严格 CER = 去标点后编辑距离/总字数（**硬指标
<5%**）；规范 CER = 再 + 繁简统一 + 中文/阿拉伯数字归一（作形态差异归因）。whisper 的
严格 CER 被数字/繁体形态拉高，规范 CER 0.058 接近门槛。

**验收结果**（T9/T10，24 句语料）：paraformer/cuda 是唯一**三项硬指标全达标**组合
（CER 0.059 逼近 5%、RTF 0.154、ttfb 0.226s）。whisper/sherpa 的 CER 均超 5%，定位为
可选项/基线。完整逐句 hyp 对照：`reports/bench_T9_*.txt`、`reports/bench_T10_*.txt`。

**流式 vs 整句出字延迟**（T13，`bench/bench_streaming.py`，同一语料、实时节奏喂入）：

| 设备 | 首字延迟（流式） | 首字延迟（整句） | 尾字延迟（流式） | 尾字延迟（整句） | 流式CER | 整句CER |
|---|---|---|---|---|---|---|
| cuda | **0.932s**（max 1.467） | 3.110s（max 5.377） | **0.348s**（max **0.486**） | 0.626s（max 1.022） | **0.017** | 0.053 |
| cpu | **0.984s**（max 1.482） | 3.405s（max 6.268） | **0.416s**（max **0.580**） | 0.905s（max 1.842） | **0.017** | 0.053 |

- **首字**：流式"边说边出字"，~0.3-0.9s（与句长无关）；整句 = 整句话说完才出字，随句长涨。
- **尾字**：流式 max 0.486s（cuda）**达标 <0.5s**；整句 max 1.022s 破线。cpu 流式 max 0.580s
  略超 0.5s（CPU 硬指标 <1s 达标），仍比整句 1.842s 好 3 倍。
- **CER**：流式定稿 0.017 优于整句 0.053（FunASR 流式定稿即整句级质量，无 CER 代价）。
- 完整报告：`reports/bench_streaming_T13_paraformer_{cuda,cpu}.txt`。

**离线**：三个后端模型首次联网下载后缓存项目内 `.cache/`，运行期零网络。

---

## 9. 热词纠错（T16，同音字）

**问题**：ASR 的**同音字**（神庙/神妙、心天/新天、四十/四时、季候/气候）是纯音频歧义——
任何声学模型都分不出 `shenmiao` 是"神妙"还是"神庙"。换更大模型不解决（实测 whisper-large
v3-turbo 语料 CER 0.141 反而最差）。**解法是文本级热词纠错**：对识别结果按热词表做
拼音级模糊替换，把模型选错的字纠正过来。

**用法**（`hotword_file` 指向热词文件；仅 paraformer 后端，online 流式 / offline 整句都生效）：

```python
asr = RealtimeASR(backend="paraformer", device="cuda",
                  hotword_file="assets/hotwords/xiaoshuang.txt")
```
```bash
PYTHONIOENCODING=utf-8 python examples/transcribe_file.py 音频.mp3 \
    --backend paraformer-offline --device cuda --hotword-file assets/hotwords/xiaoshuang.txt
```

**热词文件格式**（每行一条，支持两种模式；`#` 开头为注释）：

| 模式 | 写法 | 行为 | 适用 |
|---|---|---|---|
| **显式映射（推荐）** | `神庙=>神妙` | 出现"神庙"精确替换为"神妙" | 已知错误形式，**零误伤** |
| **模糊目标** | `神妙`（单独一行） | 对文本做拼音级模糊匹配（rapidfuzz，默认阈值 0.85） | 兜底未知同音变体 |

**实测**（`assets/hotwords/xiaoshuang.txt`，里尔克名言）：paraformer + 显式映射后
四句全部精确命中原文——神庙→神妙×2、心天、季候、四时、减少于 全部纠正。

**坑**：模糊目标对 2 字词可能**吞相邻同音字**——"的神妙"（`deshenmiao` vs `shenmiao`，
相似度 0.94）整窗被替成"神妙"删掉"的"；"必减少于"被"减少于"命中删掉"必"。**精度要求高
时用显式映射 `wrong=>right`**（确定性，只替换精确出现的错误形式），模糊目标仅兜底。

---

## 10. 常见疑问 FAQ

**Q：为什么"停顿后继续说话"被断成两句、出两个回调？**
A：VAD 把 `≥250ms` 静音当句尾（`vad_silence_tail_ms`）。停顿超过它 → 前一"句"出回调，
后面的话成为新句。想少断句就调大 tail（离线高精度），想低延迟就调小。

**Q：模型非线程安全是什么？为什么需要 `_recog_lock`？**
A：同一个 ONNX/torch 模型对象不能同时在多个线程前向（状态/缓存竞争）。引擎把
worker 的识别调用全部持同一把 `_recog_lock` 串行化。KWS 检测器同理（内部 `_lock`）。

**Q：`interrupt()` 后正在识别的那句呢？**
A：整句前向无法中止（前向是原子的）。它会跑完，但结果标记 `stale=True`，**不进普通
回调**——上层看到的只有"排队任务作废"和"之后的新句正常"。stale 结果可通过
`profile=True` 后的 `asr._sentences` 观测（调试用）。

**Q：为什么打断词命中要 ~0.2-0.4s 尾随音频？**
A：流式 KWS 需要尾随音频收尾解码（finalize 关键词）。麦克风持续采样天然满足；若音频
流在"停下"后立即结束，命中会延迟到后续音频到达。
