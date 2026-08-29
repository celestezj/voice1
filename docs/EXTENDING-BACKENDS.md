# 扩展新 LLM 引擎指南（DeepSeek / llmx 之外）

> 面向「将来接入更多 LLM 引擎」的**实操步骤手册**：本地 llama.cpp、Ollama、其他云端
> OpenAI 兼容服务、离线 mock 等。回答三个问题：LLM 层长什么样、controller 怎么调它、
> 新引擎要改哪几处。
>
> 关联文档（动手前按需精读）：
> - `docs/backend-guide.md` = **ASR 后端扩展**（voice1 自己的 ASR 引擎，另一条轴，已很完整）。
> - `docs/voice-dialogue.md` = 对话编排（线程/时序/参数），LLM 调用点的上下文。
> - voice0 `docs/EXTENDING-BACKENDS.md` = **TTS 后端扩展**（voice1 的 TTS 走 voice0，只读，不在本仓库扩展）。

---

## 1. 先搞懂：voice1 三引擎的扩展轴（ASR / LLM / TTS）

voice1 是语音对话编排，三个引擎层归属不同，扩展各自读各自的文档：

| 层 | 归属 | 引擎 | 扩展读哪 |
|---|---|---|---|
| **ASR** | voice1 自研 | `asr/` 引擎 + `asr/core/backend.py` 的 `ASRBackend` 注册表 | `docs/backend-guide.md`（已有：新增流式/非流式 ASR 后端三步） |
| **LLM** | voice1 薄客户端 | `dialogue/llm.py` 的 `LLMClient`（现状只有一个 `OpenAICompatibleClient`） | **本文档**（新增） |
| **TTS** | voice0（只读库） | `RealtimeTTS`，voice1 当库调 | voice0 `docs/EXTENDING-BACKENDS.md`（改 TTS 引擎是 voice0 的活，voice1 只透传参数） |

**LLM 层在链路里的位置**：

```
DialogueController（dialogue/controller.py，编排层——只消费 LLM 四样东西）
   │  stream_chat / compress / estimate_tokens / last_usage
   ▼
dialogue/llm.py 的 LLMClient ABC
   ├── OpenAICompatibleClient（现状：DeepSeek / llmx / 通义 / Moonshot… 一切 OpenAI 兼容）
   └── <你的>Client（§5 轴 L2：非兼容引擎才需要写）
   │
   │   配置/接线：dialogue/config.local.json（gitignored）←→ examples/voice_dialogue.py 直接构造
```

**关键不变量**：controller 只认 `LLMClient` 的**四样东西**——`stream_chat` 文本增量、
`compress` 摘要、`estimate_tokens` 估算、`last_usage`。任何新引擎只要归一成这四样，
打断、按句切分、上下文压缩、token 预算**全部白拿**，controller 一行不改。

---

## 2. 两条扩展轴，先分清你要做哪种

**轴 L1 —— 换 OpenAI 兼容引擎（最常见，零代码）**：新引擎说 OpenAI `chat/completions`
协议（DeepSeek、llmx、通义、智谱、Moonshot、Ollama 的 `/v1` 兼容层…全是）。
**只改 `config.local.json` 三个字段**，controller / 打断 / 压缩 / usage 自动适配。
- 例：DeepSeek 云端 ↔ llmx 本地（本文 §4 已实操，含一键回退）。

**轴 L2 —— 接非兼容引擎（写一个 `LLMClient` 子类）**：引擎不走 OpenAI 协议
（如进程内 llama-cpp-python 直调、私有协议、离线 mock）。实现 `LLMClient` 四方法 +
在 `voice_dialogue.py` 接线一处。
- 例：纯离线无 HTTP 的本地推理、测试用假 LLM。

**一句话**：换兼容引擎 ≈ 改配置；换非兼容引擎 ≈ 新写一层 `LLMClient`。本文 §4 讲 L1，§5 讲 L2。

---

## 3. 接口契约（LLMClient ABC，`dialogue/llm.py`）

```python
class LLMClient:
    last_usage = None    # 最近一次响应的 usage（{prompt_tokens, ...}，无则 None）

    def stream_chat(self, messages) -> Iterator[str]:
        """逐 token yield content 增量。生成器 break/close 时必须关闭底层连接
        ——这是 barge-in 能立刻切断 LLM 吐词的基础。"""
    def compress(self, history, max_tokens=512) -> str:
        """把旧对话压成一段摘要（返回 str）。不实现则历史超限时退化为不压缩。"""
    def estimate_tokens(self, messages) -> int:
        """无 usage 时的兜底估算（中文≈1 token/字 ×1.3 + 消息开销）。"""
```

**controller 实际调用点（对号入座，别漏）**：

| 方法 | 调用处（dialogue/controller.py） | 用途 |
|---|---|---|
| `stream_chat(messages)` | `_llm_loop`（324 行） | 主对话流，逐 token 增量 |
| `compress(history)` | `_compress_task`（202 行） | 上下文超预算时后台压缩旧历史 |
| `estimate_tokens(messages)` | `_maybe_compress`（187 行）/ usage 兜底（341 行） | token 预算 → 决定是否压缩 |
| `last_usage` | 流末捕获（337 行） | 精确 prompt_tokens 计量 |

**`OpenAICompatibleClient` 的构造约定**（L2 子类参考签名）：

```python
def __init__(self, api_key=None, base_url=None, model=None, temperature=0.7,
             max_tokens=None, connect_timeout=10, read_timeout=120):
    # 读取优先级：显式参数 > config.local.json > 环境变量 DEEPSEEK_API_KEY
```

---

## 4. 轴 L1：换 OpenAI 兼容引擎（配置级，零代码）

`dialogue/config.local.json`（gitignored，**绝不提交**）——三字段决定引擎：

```json
{
  "api_key": "local",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen3-4B-Q4_K_M"
}
```

- **`api_key`**：仅需**非空**。本地引擎（llmx）填 `"local"` 占位即可；云端填真 key。
- **`base_url`**：决定引擎。DeepSeek `https://api.deepseek.com` ↔ llmx `http://127.0.0.1:8000/v1` ↔ 通义/智谱/…。
- **`model`**：任意字符串。DeepSeek `deepseek-chat`；llmx 用它的 model_id（如 `Qwen3-4B-Q4_K_M`；
  不一致仅告警，不影响）。
- 读取优先级：**CLI（`--api-key / --base-url / --llm-model`）> config.local.json > 环境变量 `DEEPSEEK_API_KEY`**。

**不用改任何代码**——`OpenAICompatibleClient` 是通用 OpenAI 客户端，SSE 契约、usage、
打断（`with requests.post` 生成器关闭即断连）都是引擎无关的。

**现状参考（已实操）**：voice1 已从 DeepSeek 切到本地 llmx；DeepSeek 配置备份在
`tmp/llm-config-deepseek.json`（gitignored）。换回一条命令：

```bat
copy tmp\llm-config-deepseek.json dialogue\config.local.json
```

---

## 5. 轴 L2：写 `LLMClient` 子类（非兼容引擎）——三步

### 第 1 步：`dialogue/llm.py` 加子类

**骨架（可直接抄）**：

```python
class MyClient(LLMClient):
    """MyEngine：一句话定位（本地离线 / 私有协议 / 测试 mock）。"""

    def __init__(self, api_key=None, base_url=None, model=None, **kw):
        cfg = load_config()
        self._base_url = base_url or cfg.get("base_url") or "http://127.0.0.1:8000/v1"
        self._model = model or cfg.get("model") or "my-model"
        self.last_usage = None

    def stream_chat(self, messages):
        # 把 MyEngine 的输出归一成 content 文本增量，逐段 yield
        for delta in self._gen(messages):        # 你的引擎流式接口
            if isinstance(delta, str) and delta:
                yield delta
        self.last_usage = {...}                  # 结束前填（可选，controller 有兜底）

    def compress(self, history, max_tokens=512):
        ...  # 整段一次生成，返回摘要 str；做不到可 raise NotImplementedError（退化为不压缩）

    def estimate_tokens(self, messages):
        return super().estimate_tokens(messages)  # 沿用默认字符估算，或按你的分词精算

    def close(self): ...                         # 释放引擎资源（若有）
```

**纪律（三个"必须"）**：

1. **`stream_chat` 必须能优雅关闭**：生成器被 `break`/`close()`（打断）时要真正关掉底层
   连接/句柄——controller 的 barge-in 依赖"打断后服务端立即停"。HTTP 用
   `with requests.post(...) as resp`（现成范式）；进程内引擎要有中断手段。
2. **重量依赖在 `__init__`/首用才 import**：缺依赖报错要带安装提示（照 `OpenAICompatibleClient`
   缺 key 的 `ValueError` 风格），别在模块顶层 import 炸掉整个对话程序。
3. **中文输出 UTF-8**：Windows 终端跑 voice1 加 `PYTHONIOENCODING=utf-8`（项目惯例）。

### 第 2 步：`examples/voice_dialogue.py` 接线（仅一处）

现状 287 行直接构造 `OpenAICompatibleClient`。换成按需选类：

```python
llm = MyClient(api_key=args.api_key, base_url=args.base_url, model=args.llm_model)
```

（若想 CLI 可切换类，加一个 `--llm-backend openai|my` 分支，两行 if。）

### 第 3 步：验收（§7）。

---

## 6. 权重 / 依赖 / 机密（L2 才需要）

- **本地权重**落 `voice1/.cache/`（gitignored）→ 运行期零网络（项目硬指标，照 ASR 后端
  `_local_model_dir()` 模式）。
- **依赖**写进 `setup_env.py` 或 requirements；注意与 `voice-asr` 环境已锁版本共存
  （别让新依赖把 torch / melo 换掉）。
- **机密**（云端 key）只放 `config.local.json`（gitignored），绝不放代码 / 文档 / git。

---

## 7. 验收纪律

1. **SSE 兼容精确核对**（OpenAI 兼容类，防头号坑）：
   - `[DONE]` 必须是**裸字符串** `data: [DONE]\n\n`（绝不能 `json.dumps("[DONE]")` → 挂死）；
   - 内容帧 `choices[0].delta.content`；首帧 `delta={"role":"assistant","content":""}`；
   - usage 帧 `choices=[]`；非流式响应含 `choices[0].message.content`（`compress` 用它）。
   - 用 voice1 自己的解析逻辑逐行断言（照 `tests/smoke.py` 的核对方式）。
2. **内容级复核**：中文回答内容正确、无 `<think>` 推理标签（接 Qwen3 要在引擎侧关思考）、
   回复长度适配语音（40-80 字为宜）。
3. **打断（barge-in）**：`stream_chat` 中途 `break` 后服务端真正停——AI 不回自己的回声。
4. **上下文预算**：`estimate_tokens` 估得够准，超预算能触发 `compress` 且摘要不丢关键信息。
5. **回归**：`tmp/llm-config-deepseek.json` 一条命令回退 DeepSeek 验证原链路没坏。

---

## 8. 排障优先读哪

| 症状 | 先查 |
|---|---|
| 缺 key / 构造报错 | `config.local.json` 三字段 / 环境变量 `DEEPSEEK_API_KEY` / CLI |
| 本地引擎连不上 | llmx server 是否在跑（`curl http://127.0.0.1:8000/healthz`）；base_url 端口 |
| 流挂死（不出 `[DONE]`） | 服务端 `[DONE]` 是否裸字符串 `data: [DONE]\n\n`（**头号坑**） |
| 回复以 `<think>` 开头 | Qwen3 思考模式没关（llmx 后端已内置关闭；自己接 Qwen3 也要关） |
| 打断失效 / AI 自答回声 | `stream_chat` 是否在 break 时真正关闭连接 |
| 上下文压缩不触发 | `estimate_tokens` 估算与真实 usage 差异（`--max-context-tokens` 阈值） |
| ASR 后端扩展 | `docs/backend-guide.md`（不是本文档） |
| TTS 引擎扩展 | voice0 `docs/EXTENDING-BACKENDS.md`（只读，voice1 只透传参数） |
