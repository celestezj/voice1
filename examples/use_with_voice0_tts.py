# -*- coding: utf-8 -*-
"""示例：voice1(ASR) 与 voice0(TTS) 同一进程、同一 conda 环境（voice-asr）协同。

背景：`voice-asr` 环境当初是从 `voice-tts` 克隆的，是 voice-tts 的**严格超集**
（实测：voice-tts 有的包 voice-asr 全有，反向则多出 funasr/sherpa-onnx/whisper
等 ASR 栈）。所以两个项目**共用 voice-asr 环境即可，无需新建环境、无需重装**。

三个关键点：
1. **两个项目的根都要加 sys.path**：`sys.path.insert(0, voice1根)` + `voice0根`。
2. **HF_HOME 是唯一可能冲突的环境变量**：voice0 的 melo 后端 import 时 setdefault
   HF_HOME=voice0/.cache/hf；voice1 只有 **whisper** 后端才用 HF_HOME（默认
   paraformer 走 MODELSCOPE_CACHE，与 TTS 完全无冲突）。本示例显式把 HF_HOME 定死为
   voice0 缓存（供 melo）。**若你同时用 whisper 后端**，HF_HOME 得指到
   voice1/.cache/hf（或二选一，另一个项目的 HF 权重会按需下载一次到该处）。
3. **cosy 后端不能与 melo/ASR 同进程**（voice0 设计约束：cosy 注入 transformers
   4.51.3，与主环境 4.57.6 冲突，会明确报错）——用 cosy 必须起独立子进程，
   这是 voice0 的既有约束，与共享环境无关。

用法（中文输出需 UTF-8；TTS 会经声卡播放，介意可调小音量）：
    PYTHONIOENCODING=utf-8 python examples/use_with_voice0_tts.py [--device cpu]
"""
import argparse
import os
import sys

_PROJ1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # voice1
# voice0（只读引用，不改它任何文件）：约定放在 voice1 的**同级目录**（setup_env.py 自动克隆）
_PROJ0 = os.path.join(os.path.dirname(_PROJ1), "voice0")

# ── 在 import 两个项目之前播种缓存环境变量（setdefault 是"先到先得"，这里显式定死）──
os.environ["MODELSCOPE_CACHE"] = os.path.join(_PROJ1, ".cache", "modelscope")  # voice1 paraformer 权重
os.environ["HF_HOME"] = os.path.join(_PROJ0, ".cache", "hf")                   # voice0 melo 权重
os.environ["NLTK_DATA"] = os.path.join(_PROJ0, ".cache", "nltk_data")           # melo 英文 g2p
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# MeloTTS-Chinese 已被切成 Xet 存储：新版 huggingface_hub 对 xet 仓库绕开本地缓存
# 重下 208M checkpoint.pth（实测 hf-mirror ~70kB/s 卡 46 分钟）。禁用 xet → 命中缓存。
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, _PROJ1)
sys.path.insert(0, _PROJ0)

from asr import RealtimeASR                # noqa: E402  voice1
from tts import RealtimeTTS                # noqa: E402  voice0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu|cuda（两引擎共用）")
    args = ap.parse_args()

    tmp = os.path.join(_PROJ1, "tmp")
    os.makedirs(tmp, exist_ok=True)

    # 1) voice0 TTS：合成一句开场白 → 落盘 wav（同时经声卡播放）
    tts = RealtimeTTS(device=args.device, backend="melo", profile=True)
    greet = "你好，我是语音助手。语音识别引擎已就绪。"
    greet_wav = os.path.join(tmp, "both_greet.wav")
    tts.speak(greet, save_wav=greet_wav)          # 同步：合成完返回
    print("TTS(voice0) 合成完成 → %s" % greet_wav, flush=True)

    # 2) voice1 ASR：识别刚才那句（44.1k wav 会自动重采样到 16k）
    asr = RealtimeASR(backend="paraformer", device=args.device, profile=True)
    res = asr.ingest_file(greet_wav)
    for r in res:
        print("ASR(voice1) 识别: [%.2fs] %s" % (r.audio_end, r.text), flush=True)
    echo = res[0].text if res else "(无识别结果)"

    # 3) voice0 TTS：把识别结果复述出来（同进程往返闭环）
    echo_wav = os.path.join(tmp, "both_echo.wav")
    tts.speak("我听到你说：" + echo, save_wav=echo_wav)
    print("TTS(voice0) 复述完成 → %s" % echo_wav, flush=True)

    asr.close()
    tts.close()
    print("同进程 voice0(TTS) + voice1(ASR) 协同完成。", flush=True)


if __name__ == "__main__":
    main()
