# 语音对话 agent 接入设计（方案稿）

> 状态：**已实现（2026-09-04），随使用微调**。本文记录 2026-08-30~09-04 的设计结论，
> 实现过程有实测修正（见「实现补充」），继续维护更新。

## 背景与动机

现有语音对话（ASR → LLM → TTS）只能"聊天"，不具备**解决问题的能力**：
问天气答不出、开不了灯、没有现成工具时不能现场写脚本。

推演过两条死路，最终指向直接接 agent：

- **纯 LLM 工具调用（不走 agent）**：LLM 返回 `tool_calls` 会被当作普通输出直接送 TTS，
  工具**根本得不到执行**——必须自己实现一层 agent 循环（多轮调用 LLM + 执行工具 + 回填）。
- **自己实现 agent 循环** = 重复造轮子。

结论：**现有架构保持不变，直接接入本地 agent（claude code CLI）**——
多轮 LLM 交互循环、脚本编写/调试、工具执行全由 agent 自己做，
对话系统只负责：**ASR 文本 → 提交给 agent → 取最终结论 → TTS**（agent 循环中间文本不进 TTS）。

开灯、查天气等能力不写在对话系统里，而是**单独为 agent 配置**（skills / MCP）。

## 核心设计原则

1. **开关旁路**：`--brain llm|agent`（默认 `llm`）。开关关闭时，现有 LLM 集成逻辑
   **零改动、完全可用**；只有 agent 模式下才走新路径。
2. **常驻会话**：不每轮冷启动 claude。模拟人用 CLI 的情形——进程后台常驻、
   随时输入、历史都在，claude 自己自动压缩历史。
3. **旁路自实现机制**：agent 模式下，自实现的**历史保存、会话压缩、系统提示词全旁路**。
   claude 用自己的会话机制 + 自己的 `CLAUDE.md`（agent 模式的系统提示词）。
4. **只取最终结论**：agent 循环中的中间思考/工具输出不进 TTS，只把最终结论送 TTS。
5. **打断 = ESC 按键信号**：「停下」等价于用户按 ESC——立即停当前回合，
   **不进 agent 上下文**（agent 不会记录 ESC 按下）。绝不让"停下"文本进历史/LLM。
6. **对话期间不 kill agent 进程**：打断只能 **abort 当前 query**（进程活着、会话活着、
   历史活着），绝不"杀进程换上下文"（冷启动耗时）。
7. **权限交互 = 两个相同子过程首尾相接**：需要请求权限时，
   "语音输入→LLM 思考→音频输出（询问用户）"→"用户语音回答→LLM 继续思考/用工具→音频输出（结果）"，
   拆成两半看是一模一样的子过程，现有流水线天然支持。
8. **同一上下文贯穿整个对话进程**：对话启动建立（或 resume）唯一会话，
   从启动到退出**只允许这一个上下文**；重启对话系统时用类似 `claude -c` 的开关
   **续上次对话的上下文**。

## 架构总览

```
                 ┌─────────────────────────────────────────────┐
  麦克风 → ASR ──►  对话 controller（现有流水线，几乎不动）       │
                 │                                             │
                 │   --brain llm（默认，现状零改动）             │
                 │     ├ OpenAICompatibleClient                 │
                 │     │   ├ 历史保存 / 会话压缩 / 系统提示词 ── 全走（原样）  │
                 │     │   └ 提交 → 取文本 → TTS               │
                 │                                             │
                 │   --brain agent（新增）                      │
                 │     └ ClaudeAgentClient（dialogue/agent.py） │
                 │         ├ 常驻 claude 进程 + 单会话           │
                 │         ├ 旁路：历史/压缩/系统提示词（全不用） │
                 │         ├ query() → 只取最终结论 → TTS       │
                 │         └ abort() = ESC（不 kill）           │
                 └─────────────────────────────────────────────┘
                                          │
                          assistant/（独立目录，agent 的 cwd）
                            CLAUDE.md = 人格 + 【询问】规则
                            .mcp.json / .claude/skills/ = 开灯、天气等能力
```

- **agent 模式不读 voice1 工程 CLAUDE.md**：claude 只加载它自己工作目录
  （`assistant/`）的 CLAUDE.md——工程规范与对话人格互不干扰。

## 上下文管理（两种模式对比）

| | LLM 模式（现状） | agent 模式（新增） |
|---|---|---|
| 上下文持有者 | controller 的 `_history` | claude 会话（agent 自己管理） |
| 历史保存 | controller（快照存档 sessions/*.json） | claude 会话（自动） |
| 历史压缩 | controller 后台压缩线程 | claude 自动压缩 |
| 系统提示词 | `_build_messages` 拼 system | `assistant/CLAUDE.md` |
| 上下文边界 | 每句一个独立请求（历史随消息带上） | 同会话多次 query，天然连续 |
| 旁路内容 | —— | 历史/压缩/系统提示词全部旁路 |

agent 模式下 controller 瘦身为：**ASR 句 → 提交 → 等最终结论 → TTS**。
**本地会话存档（`sessions/*.json`）agent 模式不写**——历史/压缩在 claude 会话里由 claude 自己
管理，只落一个 `sessions/agent_session_id.txt` 供 `--agent-resume` 续会话（见「重启续上次会话」）。

## 打断（「停下」= ESC）

- 「停下」经现有 KWS 旁路命中 → 控制器调 `agent.abort()`——等价交互式 claude 的 ESC：
  立即停当前回合，**进程不 kill、会话不丢、历史保留**。
- **「停下」文本不进 agent 上下文**（agent 不会记录 ESC 按下），对齐 claude 语义：
  ESC 丢弃当前输入 → 下一句全新 query。
- 与 LLM 模式语义差异：LLM 模式是"被打断的问题保留进历史"；agent 模式 v1 直接对齐
  claude（丢弃被打断的那句），更干净。

## 权限交互（【询问】标记）

权限三级：

| 级别 | 场景 | 动作 |
|---|---|---|
| ① 预允许 | agent 配置文件里已设置允许（如查天气、读文件） | 不问，直接做 |
| ② 需确认 | 开灯、执行命令等敏感操作 | `【询问】`标记 → 语音交互（见下） |
| ③ 拒绝 | 配置文件里禁止 | 永不执行 |

② 的交互流程（= 两个相同子过程首尾相接）：

```
用户：打开卧室灯
agent：……【询问】是否允许我打开卧室灯？   ← 检测到【询问】→ 照常送 TTS 播出去
用户：允许                            ← controller 进入"等用户回答"态（不当作新任务）
agent：……（继续思考 + 调用开灯工具）…卧室灯已打开
```

- controller 新增一个小状态：最终结论带 `【询问】` → TTS 播放 → 进入**等待态**
  → 用户下一句 ASR **喂回同一个上下文**继续（agent 模式=同会话续问，上下文天然在）→ 最终结论 → TTS。
- **该机制与模式无关**：LLM 模式下上下文在 controller 历史（本来就在，原样工作）；
  agent 模式下在 claude 会话。两种模式都吃到，且对现有核心逻辑零改动。
- agent 侧靠 `assistant/CLAUDE.md` 的人格规则执行（"敏感操作先【询问】征得同意再动手"），
  配合 MCP/工具配置的允许清单，构成 v1 的权限边界。

## 重启续上次会话

- 每次运行把 `session_id` 存到 `sessions/`（已 gitignore）。
- 重启带 `--agent-resume`（对应 `claude -c`）→ resume 上次的会话 → 历史/上下文原样续上。

## 文件级改造清单（已实施 2026-09-04）

| 动作 | 文件 | 内容 | 状态 |
|---|---|---|---|
| 新增 | `dialogue/agent.py` | `ClaudeAgentClient`：起/resume 常驻会话、`query()` 只取最终结论、`abort()`（不 kill）、`close()`、session_id 落盘 | ✅ |
| 新增 | `assistant/CLAUDE.md` | 人格（口语化、【心态】标记、【询问】权限规则、技能指引）——独立目录，不进 voice1 工程 CLAUDE.md | ✅ |
| 新增 | `assistant/.mcp.json`、`assistant/.claude/skills/` | 开灯/天气等能力（按需） | ✅ 骨架 |
| 修改 | `dialogue/controller.py` | `agent` 构造参数；agent 模式旁路 `_build_messages`/压缩/系统提示；`【询问】`送 TTS 剥掉不念（`_ASK_RE`）；`abort()` 钩子（hard_stop / barge） | ✅ |
| 修改 | `examples/voice_dialogue.py` | `--brain` / `--agent-resume` / `--agent-dir` / `--agent-model` / `--agent-permission-mode` / `--agent-thinking` / `--agent-query-timeout` 参数 + 接线 | ✅ |
| 修改 | `docs/voice-dialogue.md` + 根 `CLAUDE.md` | 文档 | ✅ |
| 新增 | `tmp/test_agent_flow.py`（gitignored） | agent 全链路集成测试：多轮上下文/partial/abort/【询问】/barge | ✅ |
| 不动 | mic / wake / llm.py 现有路径 | 开关关闭时零影响 | ✅（回归 13 项全过） |

## 实现补充（2026-09-04 实测修正）

- **partial 增量来自 `StreamEvent` 而非 `AssistantMessage`**：`include_partial_messages=True`
  时 CLI 发**原始 Anthropic API 流事件**（`StreamEvent.event`），文本增量在
  `content_block_delta` 的 `delta.text_delta.text`；完整 `AssistantMessage` 与
  `ResultMessage.result` 仍照常到达。agent.py `_do_query` 只从 StreamEvent 取流式出字、
  TTS 仍只取 ResultMessage。
- **controller 集成形态**：agent 结果**异步**回来（on_result 在 agent 循环线程），
  controller 用 **per-gen `threading.Event`**（`_agent_evts[gen]`）唤醒对应回合的收尾线程
  `_agent_stream_thread`；作废回合（ctx != 当前 gen）只唤醒不碰状态。`_maybe_compress`/
  `_build_messages` 在 agent 模式整条旁路（`self._agent is not None` 早退）。
- **「停下」/ barge 的 abort 顺序**：controller 先 `_gen += 1`（快路径状态清理），锁外再
  `agent.abort()`（ESC）。下一句提交时 worker 已 drain 干净，无残留消息污染。
- **权限模式实测（2026-09-04，probe_perm_*.py）**：SDK 流式场景下 claude 是**无终端**运行，
  权限弹窗没人能答——`permission_mode="default"` 时未预允许的工具**直接自动拒绝**
  （CLI 发 `system(subtype=permission_denied)`，agent 正常汇报失败，**不挂起不崩溃**）。
  无害只读命令（如 `echo`）在 default 下会被自动放行。推论：
  - `default`（默认）= 语音场景的安全硬兜底：agent 想干未预允许的危险事会被硬门挡住；
  - 但「语音允许 → agent 真执行」在 default 下**走不通**（硬门无视语音同意，仍自动拒绝）——
    要让【询问】→用户口头同意→真正执行成立，需对目标工具**预允许**
    （`allowed_tools` 白名单，此时不触发硬门，agent 可直接跑）或
    `acceptEdits`（仅自动接受文件编辑）/ `bypassPermissions`（全放行，危险）。
    v1 推荐姿势：把想真正放行的**具体工具**列进 `allowed_tools`（如开灯的 MCP 工具），
    其余保持 default 自动拒绝兜底；【询问】变成"用预允许工具前的社交许可层"。
- **默认白名单（2026-09-09 落地，09-09 天气被拦后扩充）**：`agent.py` 的 `_DEFAULT_ALLOWED_TOOLS`
  预放行 `PowerShell / Bash / Read / Write / Edit / Glob / Grep / WebFetch / WebSearch / Skill`
  （未显式传 allowed_tools 时生效）——技能（如天气）要跑脚本/读配置/写结果，须这些底层工具
  放行，否则 SDK 无终端可"点允许"，agent 只能让你去命令行手动批准。
  **实测 Windows 两个 shell 都能跑**：PowerShell 与 Bash（Git Bash）。模型可能任选其一——
  读了 SKILL.md（写 `bash fetch.sh`）的会走 Bash，没细读的直接 PowerShell 跑 python。
  **天气"脚本被拦"根因**：白名单最初只有 PowerShell，走 Bash 的会话被自动拒绝，模型报
  "运行脚本被拦住了…你去终端放行"（措辞与强制禁 shell 的探针一字不差）。故两个 shell 都放行。
  `Skill` 实测即使不在白名单也不会被拒（技能发现即放行），显式列出更稳；`WebFetch/WebSearch`
  是模型在脚本被拒/失败时退到"用网页查"的兜底，放行避免二次拒绝。
  安全性：放行 PowerShell/Bash = agent 可在本机执行任意命令，系统硬门消失，只剩人格【询问】
  这层社交许可；更严的语音级工具授权（`can_use_tool` 钩子 + 语音确认）留作后续。
- **MCP 工具自动放行（2026-09-10）**：MCP 工具名是动态的（`mcp__<server>__<tool>`），静态
  `_DEFAULT_ALLOWED_TOOLS` 覆盖不到——default 模式下未预放行的 MCP 工具会被自动拒绝（agent
  只会说"查XX的工具没放行/被拦住了"）。`agent.py` `_connect` 在启用 MCP 时按 `.mcp.json` 里
  实际的 server 名自动补 `mcp__<name>__*` 白名单模式（只放行配置的 server，不放开用户全局
  MCP；新增 MCP 无需改码，server 名即 json 键）。实测金价 MCP 放行后 agent 直接调
  `get_gold_history` 拿真实数据、不再报被拦。
  **独立 venv 的 MCP（search，2026-09-13）**：`free-search-mcp` 装在独立 venv
  `assistant/.venv-search`（editable 指向本机 `free-search-mcp` 源码 checkout），不能配裸
  `python`——`agent.py` 会把 `command: python` 替换成 voice-asr 的 python，`-m search_mcp`
  在其内报 No module named → server 启动失败 → 工具不挂载（实测 AI 只有金价工具、search
  隐形）。正确配置：`command: ".venv-search/Scripts/python.exe", args: ["-m", "search_mcp"]`
  （相对路径转绝对、`-m` 后参数不转路径，`agent.py` 已支持）。
- **agent 延迟治理（2026-09-10 实测）**：天气/普通查询曾"几分钟不回复"，根因有二——
  ① **未关思考预算**：模型走方舟 `ark-code-latest`，CLI 不认识它（stderr
  `[claude-code:unrecognized_model]`）→ 按超大默认 thinking 预算先"想"约 45s 才开口。
  实测同一查询：thinking 开 = 48.8s，`max_thinking_tokens=0` 关 = 1.9s。agent.py 默认
  关（`--agent-thinking <预算>` 可重开，如 2048 换质量）；② **被中断残留污染的 resumed 会话**
  （上一进程 mid-query 被杀/退出时 close 没 interrupt 在途回合）→ resume 后每步静默 2-3 分钟。
  治理三件套（agent.py 已落地）：
  - **看门狗** `--agent-query-timeout`（默认 90s）：`_do_query` 用 `asyncio.wait_for` 包
    receive 循环，整轮超时仍无 `ResultMessage` → `interrupt()`（ESC）+ 抛 TimeoutError →
    worker 报错给 controller → 控制台 "× LLM 出错：agent 超时"，**绝不无限挂起**；
  - **stderr 环形缓存**（`_stderr_buf`，300 行）：`stderr` 不再 `lambda line: None` 全吞，
    报错/超时时 `_notify_error` dump 最近 15 行（`[agent] CLI 最近输出：…`），`--debug` 则实时
    打印 `[agent-cli]` —— 曾因全吞查不出卡因；
  - **`close()` 先直接 interrupt 在途回合**（不经队列——worker 若卡在 receive 处理不了
    `("close",)`），保证 CLI 回合干净收尾、不留脏回合给下次 resume。
  遇 agent 卡死/疑似会话污染：删 `sessions/agent_session_id.txt` 换全新会话（病会话删除即弃，
  别 `--agent-resume` 续它）。
- **流式增量送 TTS（`--agent-stream-tts`，默认关）**：agent 只取最终结论送 TTS 意味着
  工具调用前的过渡句/思考段（实测「我把未来七天的天气捋一遍给你哈」）只显示在控制台、
  不出声，出声前干等工具 7-8s。开此开关后 `_on_agent_partial` 的增量也按句送 TTS
  （心态标记跨 delta 未闭合不切句，避免半截标记被念出来；句中心态标记也作切点，防
  "…哈【心态：开心】阿阳…"粘成一个 Job；**无标点缓冲以句末语气词兜底切**，让
  "我再确认一下…哈"这类无句号过渡句在工具调用期间先出声；超长无标点走既有硬切兜底）；
  **最终结论到达三态收尾**（`_on_agent_result`）：结论整段已进流式队列 → 不打断自然播完
  （**打断会把"已入队未开播"的结论音频全取消 → 完全静音**，2026-09-13 实测）；已播全是
  结论前缀 → 不打断只补送剩余（残句由 `clean_full` 补齐续播）；已播含过渡句/思考段
  （`skip < len(played)`）→ 立即 `tts.interrupt()` 打断 + 从 `_tail_overlap` 跳过已播开头
  重播，防"阿阳"整句播两遍。**去重只对"最后一个心态标记之后"的结论本体算**——ResultMessage
  全文常以过渡句开头，对全文算会把过渡句误当已播结论跳过打断（2026-09-13 实测：结论出来
  了过渡句还先播）。关 = 只播最终结论（旧行为零变化）。实现见 controller
  `_agent_stream_tts` / `_agent_tts_buf` / `_agent_tts_played` / `_flush_agent_stream_locked` /
  `_on_agent_result`（三态 + 最后心态标记界定结论） / `_find_cut`（心态标记作切点） /
  `_PARTICLES`（语气词兜底切）。

## free-search-mcp 接入详解与换设备重建（2026-09-25 实测）

> **开源项目来源**：https://github.com/sweetcornna/free-search-mcp
> （MIT，Python >= 3.11；本机接入实测版本 0.11.0 / playwright 1.62 / mcp 2.2）。
> 上游是"本地优先、免 API key"的搜索 MCP；独立 git clone（不在 voice1 仓库内），
> 装进独立 venv `assistant/.venv-search`（gitignored），以 **editable** 方式指向本地
> clone（`pip install -e`，`.pth` 硬编码指向 clone 的绝对路径——**机器相关**）。

### 接入链路

1. **挂载配置**在 `assistant/.mcp.json`（已入库、自包含，相对路径）：
   ```json
   "search": {
     "command": ".venv-search/Scripts/python.exe",
     "args": ["-m", "search_mcp"],
     "env": {}
   }
   ```
2. **路径解析**（`agent.py` `_resolve_mcp_config`，挂载与启动探测共用）：
   - `command` 是相对 assistant 目录的**非 python 可执行文件** → 解析成
     `assistant/.venv-search/Scripts/python.exe`（**不被替换成 voice-asr**）。
   - `args: ["-m", "search_mcp"]` 是 `-m` 模块名 → **保持原样不做路径解析**。
   - 这两个规则缺一不可，见下方"为什么不能配裸 python"。
3. **入口点**：free-search-mcp 安装后提供 `free-search-mcp` / `search-mcp` / `-m search_mcp`
   三种入口，都落到 `search_mcp.__main__:main`。stdin/stdout 走 MCP 2.x stdio 协议。
4. **能力本质**：**本地优先、免 API key**。默认四引擎全 HTTP：`duckduckgo` /
   `mojeek` / `googlenews` / `bing`（`src/search_mcp/config.py` `default_engines`），
   **无 key 零配置直接可用**。

### 为什么不能配裸 `python`（踩坑实测）

`agent.py` 会把 `command: python` 替换成 `sys.executable`（voice-asr），而 `search_mcp`
只装在 `.venv-search` 里，voice-asr 没有 → 启动即 `No module named search_mcp` →
server 启动失败 → 工具**静默不挂载**（实测 AI 只有金价工具、search 隐形，无任何报错）。
`command` 一旦换成 venv python、`-m` 保留，立即挂载成功。

### 换设备能否复现

**不能直接搬，但可四步重建**。机器相关的三点：

| 组件 | 状态 | 换设备影响 |
|---|---|---|
| `assistant/.mcp.json` | 已入库、全相对路径 | ✅ 直接可复现，不用改 |
| `assistant/.venv-search/` | **gitignored**（`assistant/.gitignore:30`） | ❌ 必须重建 |
| 上游源码（本地 clone） | **独立 clone**（不在 voice1 仓库） | ❌ 必须另行 clone |
| editable 安装的 `.pth` | 硬编码指向**clone 的绝对路径** | ❌ 重建 venv 时指向新路径 |

editable 是"指向本地源码 checkout"，**不是**把包复制进 venv——换设备光重建 venv 没用，
得先把源码 clone 下来。

### 重建步骤（新设备四步）

```bash
# ① clone 上游源码（路径随意，仓库内/外皆可，建议放仓库外）
git clone https://github.com/sweetcornna/free-search-mcp.git

# ② 建独立 venv（Python 需 >= 3.11，用任一 python 基座）
python -m venv <仓库根>/assistant/.venv-search

# ③ 以 editable 方式安装（指向刚才 clone 的路径）
<仓库根>/assistant/.venv-search/Scripts/pip install -e <clone路径>

# ④ （可选）装 Chromium 浏览器
<仓库根>/assistant/.venv-search/Scripts/playwright install chromium
```

**`assistant/.mcp.json` 一行不用改**——已是相对路径 + `-m` 模块名，venv 建好、源码装好即生效。
验证：`assistant/.venv-search/Scripts/python.exe -c "import search_mcp"` 不报错；或直接跑
主程序看启动 `[agent] MCP 工具：search : …` 清单。

### 要装什么

- **Python 3.11+**（建 venv 的 base，任意可用的 python 基座）。
- **free-search-mcp 包 + 依赖**：`pip install -e` 自动装 `playwright`、`httpx`、
  `mcp>=2.0`、`pydantic` 等——已验证 `search_mcp` 可正常 import。
- **Chromium（可选但推荐）**：默认四引擎全 HTTP，**不装也能搜**；但浏览器渲染引擎
  （`startpage` / `zhihu` 等）和 JS 重的页面抓取需要 Chromium。没装时那些调用返回
  "请先运行 `playwright install chromium`" 提示，**不报错**。

### 更省事的替代方案（可选）

上游 README：任何 MCP 客户端可直接配 `command: uvx free-search-mcp`——**不用 clone、
不用建 venv**，首次运行自动从 PyPI 拉包。代价：每次启动走 `uvx` 解析（首次较慢），且
与当前"editable 指向源码"的做法不一致。若换设备不想维护本地源码 checkout 可改用这条。

## lunar-python 接入（日历/八字/黄历查询，2026-09-25 实测）

> **开源项目来源**：https://github.com/6tail/lunar-python
> （MIT，纯 Python 零依赖日历库：公历/农历/佛历/道历、干支/生肖/节气/节日、彭祖百忌/每日宜忌、
> 吉神方位/胎神/冲煞/纳音/星宿、八字/五行/十神、建除值星/黄道黑道等）。
> 与 free-search-mcp 最大的不同：**零依赖** → 不建独立 venv，直接装进 voice-asr
> （`pip install lunar_python` 一条命令），换设备一条命令可复现。

### 接入链路

1. **MCP server 脚本**：`tool/mcp_tools/lunar.py`（主仓库，入库），mcp SDK 2.x `MCPServer`，
   暴露 5 个工具：
   - `get_calendar(date)` — 某日**黄历全览**（农历/干支/生肖/宜忌/彭祖百忌/吉神方位/冲煞/星宿/
     值星/黄道黑道/节日/数九三伏）；
   - `get_bazi(date, time, gender, da_yun)` — **八字排盘**（四柱/五行/十神/纳音/旬空/命宫身宫
     胎元/大运；23:00-23:59 属下一日子时，边界敏感会返回 error 提示）；
   - `get_holiday(date)` — 法定节假日/调休（`HolidayUtil`）；
   - `get_jieqi(year)` — 全年 24 节气表；
   - `get_festival(date)` — 公历+农历节日。
   带 `--selfcheck` 独立自检（仿 gold `--fetch`）；坏输入返回 `{"error": ...}` 不抛异常。
2. **双模式同一份配置复用同一个脚本**（相对路径解析以各自基准）：
   - **agent 模式** `assistant/.mcp.json`：
     ```json
     "lunar": { "command": "python", "args": ["../tool/mcp_tools/lunar.py"], "env": {} }
     ```
     （相对 args 以 assistant/ 为基准 → 仓库根 `tool/mcp_tools/lunar.py`；`command: python`
     → voice-asr，lunar_python 就装在那里。）
   - **LLM 模式** `tool/mcp.local.json`（gitignored；示例 `tool/mcp.local.example.json` 入库）：
     ```json
     "lunar": { "timeout": 60, "type": "stdio", "command": "python",
                "args": ["./tool/mcp_tools/lunar.py"] }
     ```
     （相对 args 以仓库根为基准；`--tools all|mcp` 时暴露 `lunar_get_*`。）
3. **挂载/探测/放行全自动零改码**：agent 模式 `mcp__lunar__*` 白名单自动补，
   `[agent] MCP 工具：lunar : …` 启动清单自动出现；LLM 模式 `[tools]` 分组自动列出。

### 换设备能否复现

**能，两步即可**（对比 free-search-mcp 的四步）：

| 组件 | 状态 | 换设备影响 |
|---|---|---|
| `tool/mcp_tools/lunar.py` | 已入库（主仓库） | ✅ 直接可复现 |
| `assistant/.mcp.json` + `tool/mcp.local.json` | 已入库 / 示例入库 | ✅ 直接可复现（`mcp.local.json` 需重建一份） |
| `lunar_python` 库 | 装进 voice-asr（**非独立 venv**） | ❌ 新设备 `pip install lunar_python` 一条命令 |

### 重建步骤（新设备两步）

```bash
# ① 装库（零依赖，一条命令进 voice-asr）
conda activate voice-asr && python -m pip install lunar_python

# ② 若 LLM 模式需要，重建本机业务配置（仓库内的脚本/示例已随 git 同步，直接可用）
#    抄 tool/mcp.local.example.json 里的 lunar 段到 tool/mcp.local.json 即可
```

验证：`python -c "from lunar_python import Solar; from lunar_python.util import HolidayUtil"` 不报错；
或 `python tool/mcp_tools/lunar.py --selfcheck` 打印各工具样例；或直接跑主程序看
`[agent] MCP 工具：lunar : …` / `[tools] …lunar_get_*` 清单。

## 技术风险 / 待验证点

- **abort 不 kill 的具体机制**（整个设计唯一的技术风险点，实现第一步先验证）：
  `claude-agent-sdk` 或裸 `claude --output-format stream-json` 常驻子进程，
  是否支持"中断当前回合、进程/会话存活"。SDK 不行就回退到裸子进程喂 stdin +
  发中断信号，语义等价交互式 claude 的 ESC。
- **会话 resume**：`session_id` 的获取/续接方式（对应 `claude --resume <id>` / `-c`）。

## 技术验证结论（2026-09-04 实测，SDK 0.2.152 / claude 2.1.260）

**机制选型：`claude-agent-sdk`（Python），不用裸 CLI 子进程。**
裸 `claude --output-format stream-json` 的 stdin 不是 TTY 时自动进 print 模式——
3 秒内收不到输入就报错退出，**根本不能常驻**。SDK 用内部 messaging socket
（命名管道）常驻控制，正是为"常驻会话 + 可中断"设计的。

| 验证点 | 结论 |
|---|---|
| 常驻多轮 + 上下文连续 | ✅ 同一 `ClaudeSDKClient` 多次 `query()`，会话记住之前内容 |
| `interrupt()` = ESC | ✅ 进程不 kill；旧回合以 `ResultMessage(subtype='error_during_execution', is_error=True)` 干净收尾，随后新 query 正常 |
| 跨进程 resume | ✅ `resume=<session_id>` 重建 client 后仍记得会话内容 |
| `session_id` 格式 | ⚠️ **必须是合法 UUID**（否则 "Invalid session ID"） |
| 最终结论 | ✅ `ResultMessage.result`；流式文本在 `AssistantMessage.content[].text` |
| `cwd` 的 CLAUDE.md 自动当人格 | ❌ **不自动加载**——必须显式 `system_prompt=<人格文本>`（实测生效） |
| 冷启动耗时 | ✅ connect ≈0.6s（冷启动只一次）；热查询 ≈2s |
| 权限 | `permission_mode` / `allowed_tools` / `disallowed_tools` / `can_use_tool` 钩子 |

**实现要点（踩坑）**：
- SDK 全部控制方法（connect/query/interrupt/disconnect）是**协程**，须 await；
  `receive_response()` 是异步迭代器。集成进 controller 的线程模型需要一个
  常驻 asyncio 事件循环线程（`asyncio.run_coroutine_threadsafe` 桥接）。
- 每回合必须 drain 到 ResultMessage 再发下一条（打断后尤其如此，否则残留消息
  污染下一轮 receive——round3 result=None 的成因）。
- 人格用 `assistant/CLAUDE.md` 作为唯一事实源，连接时读文件内容传 `system_prompt`。

## 实现顺序

1. 验证 abort 不 kill + resume（技术风险点）。**已完成（见上表）**。
2. 按文件级改造清单落代码（先 agent.py，再 controller 开关，再 CLI/文档）。**已完成**。
3. 验收：开关默认 llm 时行为与现状完全一致（回归 13 项全过）；agent 模式下开关灯/天气走
   能力配置、敏感操作走【询问】（实测 agent 回复带【询问】→ TTS 剥掉不念/全文保留）、
   打断走 abort 且不进上下文（实测 abort 后同会话存活）、重启可续会话。**已完成**。
   `--agent-resume` 跨进程续会话：**已实测通过**（tmp/test_agent_resume.py：进程1 记暗号
   落盘 session_id → close → 进程2 resume 同文件 → 记得暗号）。

## 更新记录

- 2026-09-04：方案闭环，记录设计结论（开关/常驻会话/旁路/打断=ESC/权限【询问】/重启续会话）。
- 2026-09-04：实现完成 + 实测修正（StreamEvent partial、per-gen 事件收尾、abort 顺序），
  验收通过（LLM 模式回归 13 项全过 + agent 全链路集成测试 6 项全过）。
- 2026-09-25：新增「free-search-mcp 接入详解与换设备重建」节——接入链路（.mcp.json →
  venv python → `-m search_mcp`）、裸 python 踩坑、可复现性判定表、四步重建流程、
  依赖清单、`uvx` 替代方案。
- 2026-09-25：新增「lunar-python 接入」节——日历/八字/黄历查询 MCP（5 工具）、双模式
  同一脚本复用（agent `../tool/mcp_tools/lunar.py` + LLM `./tool/mcp_tools/lunar.py`）、
  零依赖装进 voice-asr、换设备两步重建。
