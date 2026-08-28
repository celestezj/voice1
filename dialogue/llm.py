# -*- coding: utf-8 -*-
"""LLM 客户端：OpenAI 兼容协议 + SSE 流式（DeepSeek 等）。

本地未部署大模型，走 HTTP API。用 `requests` 裸接 SSE（不引 SDK），生成器在
break/关闭（打断）时自动关闭连接——这是 barge-in 能立刻切断 LLM 吐词的基础。

机密配置（API key）放 `dialogue/config.local.json`（已在 .gitignore 里，**绝不
提交**）。读取优先级：显式参数 > config.local.json > 环境变量 DEEPSEEK_API_KEY。
"""
import json
import os

import requests


class LLMClient:
    """聊天客户端抽象：`stream_chat(messages)` 返回文本增量迭代器。"""

    last_usage = None            # 最近一次响应的 usage（{prompt_tokens, ...}，无则 None）

    def stream_chat(self, messages):
        raise NotImplementedError

    def compress(self, history):
        """把旧对话压成一段摘要（返回 str）。不实现则历史超限时退化为不压缩。"""
        raise NotImplementedError

    def estimate_tokens(self, messages):
        """无 usage 时的兜底估算：中文≈1 token/字 ×1.3 + 消息开销。"""
        n = 0
        for m in messages:
            n += len(m.get("content") or "") * 1.3 + 4
        return int(n)


def load_config(path=None):
    """读本地机密配置（含 API key / base_url / model）。缺文件返回 {}。"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config.local.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


class OpenAICompatibleClient(LLMClient):
    """OpenAI 兼容 `chat/completions` 流式客户端（DeepSeek / 通义 / Moonshot…通用）。

    - `stream_chat` 逐 token yield `content` 增量；生成器被 break/close（打断）时
      通过 `with requests.post(...) as resp` 关闭连接，服务端收到 TCP 断开即停。
    - 非 200 / SSE 里的 `error` 帧 → 抛带状态码与响应摘要的 RuntimeError。
    """

    def __init__(self, api_key=None, base_url=None, model=None, temperature=0.7,
                 max_tokens=None, connect_timeout=10, read_timeout=120):
        cfg = load_config()
        self._api_key = api_key or cfg.get("api_key") \
            or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "缺少 LLM API key：写 dialogue/config.local.json，或设环境变量 "
                "DEEPSEEK_API_KEY，或用 --api-key 传入。")
        self._base_url = (base_url or cfg.get("base_url")
                          or "https://api.deepseek.com").rstrip("/")
        self._model = model or cfg.get("model") or "deepseek-chat"
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._timeout = (connect_timeout, read_timeout)   # (连接, 单次读) 秒
        self.last_usage = None                # 最近一次响应 usage（流末 chunk 捕获）

    def stream_chat(self, messages):
        url = self._base_url + "/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": self._temperature,
            # 流末返回 usage（含 prompt_tokens 精确数）→ 供上下文预算/压缩触发
            "stream_options": {"include_usage": True},
        }
        if self._max_tokens:
            payload["max_tokens"] = int(self._max_tokens)
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }
        with requests.post(url, json=payload, headers=headers,
                           stream=True, timeout=self._timeout) as resp:
            if resp.status_code != 200:
                body = resp.text[:200]
                resp.close()
                raise RuntimeError("LLM API HTTP %d: %s" % (resp.status_code, body))
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue                  # 心跳/注释行，忽略
                if "error" in obj:
                    raise RuntimeError("LLM API 错误: %s" % obj["error"])
                if "usage" in obj:            # include_usage 的流末帧（choices 为空）
                    self.last_usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if isinstance(delta, dict) and delta.get("content"):
                    yield delta["content"]

    def compress(self, history, max_tokens=512):
        """把旧对话压成一段摘要（上下文逼近窗口时替换旧历史）。

        专用摘要 system prompt + 非流式 chat 调用；独立请求，与对话流互不干扰。
        不覆盖 `last_usage`（那是对话上下文的计量，由 `stream_chat` 维护）。
        """
        text = "\n".join("[%s] %s" % (m.get("role", "?"), (m.get("content") or ""))
                         for m in history)
        messages = [
            {"role": "system", "content": self._COMPRESS_SYSTEM},
            {"role": "user", "content": "[对话记录]\n" + text},
        ]
        url = self._base_url + "/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": int(max_tokens),
        }
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
        try:
            if resp.status_code != 200:
                raise RuntimeError("LLM API HTTP %d: %s" % (resp.status_code,
                                                           resp.text[:200]))
            obj = resp.json()
            return (obj["choices"][0]["message"]["content"] or "").strip()
        finally:
            resp.close()

    _COMPRESS_SYSTEM = (
        "你是对话历史压缩器。把下面的对话记录压缩成一段简洁的摘要（中文，2-5 句话）。"
        "必须保留：用户的核心诉求、偏好与约定、已确认的事实、未完成的任务、关键的人名/"
        "地名/时间/数字。省略寒暄、重复与无关细节。只输出摘要本身，不要任何解释或前缀。")
