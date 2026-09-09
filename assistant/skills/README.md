# assistant 技能目录说明（agent 能力）

agent 大脑的工具能力都在 **`assistant/.claude/skills/`**（不是本目录）。

> **为什么不是 `assistant/skills/`？** Claude Code 只从标准位置发现技能：个人级
> `~/.claude/skills/`、项目级 `<项目根>/.claude/skills/`（启动目录 cwd 与 git 根都会被扫描）。
> 裸 `skills/` 目录（不在 `.claude/` 下）**不会被 `/skill` 发现**（2026-09-08 实测：
> 放裸目录 `/skill` 为空，挪到 `assistant/.claude/skills/` 即被列出）。
> 本目录只是仓库里跟踪的说明文档，技能实体在 `.claude/skills/`。

## 技能位置

每个技能一个子目录，含 `SKILL.md`（说明 + 用法 + 示例），放：

```
assistant/.claude/skills/qweather/
└── SKILL.md          # 用法 + 参数 + 注意事项 + 示例
    scripts/          # 技能自带脚本（$CLAUDE_SKILL_DIR 指到这里）
```

手动测试：在 `assistant/` 下启动 `claude`，跑 `/skill` 应列出全部技能。
agent 模式（voice_dialogue `--brain agent`）的 SDK 会话 cwd 也是 `assistant/`，同一套发现规则。
**技能在会话连接那一刻一次性加载**：新增/移动/修改技能后，正在跑的 voice_dialogue
看不到——必须重启 `voice_dialogue.py` 起新进程（`--agent-resume` 续会话也会重新扫描，
2026-09-09 SDK 实测）。

## 技能要跑脚本：工具白名单（已默认放行）

SDK 会话无终端、`permission_mode=default` 下，未预放行的工具调用被系统**自动拒绝**（agent
只能让你去命令行"点允许"，但你没有可点的终端）。`dialogue/agent.py` 的 `_DEFAULT_ALLOWED_TOOLS`
已默认放行 `PowerShell / Bash / Read / Write / Edit / Glob / Grep / WebFetch / WebSearch / Skill`——
写脚本类技能（跑 python、读写配置、网页兜底）够用。
**Windows 的 shell 有俩，都能跑**：`PowerShell` 与 `Bash`（Git Bash），模型可能任选其一——
读了 SKILL.md（写 `bash fetch.sh`）的会走 Bash，没细读的直接 PowerShell 跑 python。
**当初白名单只有 PowerShell，走 Bash 的会话被自动拒绝 → 模型报"脚本被拦住了"**（天气查不到
的根因），故两个 shell 都放行。需要别的工具时在 agent.py 白名单加。

## 注意：`.claude/` 被 gitignore，技能不进 git

`.gitignore` 排除了 `.claude/`，所以技能文件**不随仓库提交**（暂不跟踪，后续有需要再加例外）。
换机器/克隆后需手动重建技能目录。

## 添加技能

写好 `SKILL.md` 即被 claude 自动发现。**新增/修改技能前先记住：
执行类能力（开灯/写文件/跑命令）属敏感操作，agent 会按人格先【询问】征得同意再动手。**

## MCP 服务（assistant/.mcp.json）

外部工具走 `assistant/.mcp.json`（已建骨架）。按需填入 `mcpServers`，例如：

```json
{
  "mcpServers": {
    "weather": { "command": "uvx", "args": ["weather-mcp"] }
  }
}
```

启动 agent 后新增 MCP 服务需重启 `voice_dialogue.py` 才生效。
