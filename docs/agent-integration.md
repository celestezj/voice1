# 语音对话 agent 接入设计（方案稿）

> 状态：**方案讨论已闭环，待实现**。本文记录 2026-08-30~09-04 的设计结论，
> 随实现过程微调并持续维护更新。实现前先读本文「技术风险 / 待验证点」。

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
本地会话存档（`sessions/*.json`）建议**仍保留**（审计），与 claude 侧会话并存。

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

## 文件级改造清单（方案，未实施）

| 动作 | 文件 | 内容 |
|---|---|---|
| 新增 | `dialogue/agent.py` | `ClaudeAgentClient`：起/resume 常驻会话、`query()` 只取最终结论、`abort()`（不 kill）、`close()`、session_id 落盘 |
| 新增 | `assistant/CLAUDE.md` | 人格（口语化、【心态】标记、【询问】权限规则、技能指引）——独立目录，不进 voice1 工程 CLAUDE.md |
| 新增 | `assistant/.mcp.json`、`assistant/skills/` | 开灯/天气等能力（按需） |
| 修改 | `dialogue/controller.py` | `brain` 开关；agent 模式旁路 `_build_messages`/压缩/系统提示；`【询问】`检测 → 等待态 → 同上下文续；`abort()` 钩子 |
| 修改 | `examples/voice_dialogue.py` | `--brain` / `--agent-resume` / `--agent-dir` 参数 |
| 修改 | `docs/voice-dialogue.md` + 根 `CLAUDE.md` | 文档 |
| 不动 | mic / wake / llm.py 现有路径 | 开关关闭时零影响 |

## 技术风险 / 待验证点

- **abort 不 kill 的具体机制**（整个设计唯一的技术风险点，实现第一步先验证）：
  `claude-agent-sdk` 或裸 `claude --output-format stream-json` 常驻子进程，
  是否支持"中断当前回合、进程/会话存活"。SDK 不行就回退到裸子进程喂 stdin +
  发中断信号，语义等价交互式 claude 的 ESC。
- **会话 resume**：`session_id` 的获取/续接方式（对应 `claude --resume <id>` / `-c`）。

## 实现顺序

1. 验证 abort 不 kill + resume（技术风险点）。
2. 按文件级改造清单落代码（先 agent.py，再 controller 开关，再 CLI/文档）。
3. 验收：开关默认 llm 时行为与现状完全一致；agent 模式下开关灯/天气走能力配置、
   敏感操作走【询问】、打断走 abort 且不进上下文、重启可续会话。

## 更新记录

- 2026-09-04：方案闭环，记录设计结论（开关/常驻会话/旁路/打断=ESC/权限【询问】/重启续会话）。
