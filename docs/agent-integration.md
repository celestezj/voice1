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
                            .mcp.json / skills/ = 开灯、天气等能力
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
| 新增 | `assistant/.mcp.json`、`assistant/skills/` | 开灯/天气等能力（按需） | ✅ 骨架 |
| 修改 | `dialogue/controller.py` | `agent` 构造参数；agent 模式旁路 `_build_messages`/压缩/系统提示；`【询问】`送 TTS 剥掉不念（`_ASK_RE`）；`abort()` 钩子（hard_stop / barge） | ✅ |
| 修改 | `examples/voice_dialogue.py` | `--brain` / `--agent-resume` / `--agent-dir` / `--agent-model` / `--agent-permission-mode` 参数 + 接线 | ✅ |
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
