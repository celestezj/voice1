# -*- coding: utf-8 -*-
"""语音对话编排包：ASR(voice1) + LLM(OpenAI 兼容 API) + TTS(voice0)。

- `DialogueController`：打断状态机 + 按句切分（dialogue/controller.py）
- `OpenAICompatibleClient` / `LLMClient`：SSE 流式 LLM 客户端（dialogue/llm.py）
- `dialogue.mic`：麦克风基建（设备挑选 / 信号检测 / 自适应增益）

主程序见 `examples/voice_dialogue.py`。机密配置（API key）放 `dialogue/config.local.json`
（已 gitignore，**绝不提交**）。
"""
from .controller import DialogueController
from .llm import LLMClient, OpenAICompatibleClient, load_config

__all__ = ["DialogueController", "LLMClient", "OpenAICompatibleClient", "load_config"]
