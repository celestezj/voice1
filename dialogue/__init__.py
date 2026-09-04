# -*- coding: utf-8 -*-
"""语音对话编排包：ASR(voice1) + LLM/Agent + TTS(voice0)。

- `DialogueController`：打断状态机 + 按句切分（dialogue/controller.py）；`brain` 开关选
  LLM 模式（现状）或 agent 模式（旁路自实现历史/压缩/系统提示词）。
- `OpenAICompatibleClient` / `LLMClient`：SSE 流式 LLM 客户端（dialogue/llm.py）
- `ClaudeAgentClient`：本地 claude code 常驻会话适配器（dialogue/agent.py，agent 模式）
- `dialogue.mic`：麦克风基建（设备挑选 / 信号检测 / 自适应增益）

主程序见 `examples/voice_dialogue.py`。机密配置（API key）放 `dialogue/config.local.json`
（已 gitignore，**绝不提交**）。
"""
from .controller import DialogueController
from .llm import LLMClient, OpenAICompatibleClient, load_config
from .agent import ClaudeAgentClient

__all__ = ["DialogueController", "LLMClient", "OpenAICompatibleClient", "load_config",
           "ClaudeAgentClient"]
