# assistant skills 目录（agent 能力，按需添加）

agent 大脑的工具能力都配在这里。v1 骨架：暂无内置技能，按需补。

## 添加技能

每个技能一个子目录，含 `SKILL.md`（说明 + 用法 + 示例），例如：

```
skills/light_control/
└── SKILL.md      # 开灯/关灯：如何调用，参数，注意事项
```

技能写好即被 agent 自动发现（claude 的 skills 约定）。**新增/修改技能前先记住：
执行类能力（开灯/写文件/跑命令）属敏感操作，agent 会按人格先【询问】征得同意再动手。**

## MCP 服务（assistant/.mcp.json）

外部工具走 `.mcp.json`（同目录，骨架已建）。按需填入 `mcpServers`，例如：

```json
{
  "mcpServers": {
    "weather": { "command": "uvx", "args": ["weather-mcp"] }
  }
}
```

启动 agent 后新增 MCP 服务需重启 `voice_dialogue.py` 才生效。
