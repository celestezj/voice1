# -*- coding: utf-8 -*-
"""语音对话主程序：麦克风 → voice1 ASR → DeepSeek LLM → voice0 TTS（单进程非阻塞编排）。

链路（全部非阻塞，各组件内部线程干活）：
    主线程：一直采麦克风块 → asr.ingest()
    ASR worker：识别 → on_partial（流式出字）/ on_sentence（定稿句）→ controller.feed_asr_sentence()
    LLM 线程：读 SSE 流 → 按句切分 → tts.submit(句)（非阻塞入队）
    TTS worker：queue 模式串行合成 + 播放

打断语义（用户已拍板）：
- 新 ASR 句若 LLM 在途 → 取消当前生成 + 重发「本轮累计」；此时联动 tts.interrupt() 切掉作废音频。
- LLM 已生成完、TTS 还在播时来新句 → 不打断语音，让它播完，新回复排到后面。
- 打断词（默认"停下"）→ 立即终止 LLM+TTS，被打断的问题保留进历史，"停下"不入历史。

回声（v1 半双工门控）：TTS 播放期 mic 只喂 `ingest_kws_only`（"停下"KWS），回声不进
识别 → 无反馈自答；播放期普通说话被忽略。`--no-echo-gate` 可关（耳机场景保留打断）。
LLM 首句 hold-off（`--reply-hold` 默认 0.35s）缩小"AI 已开播用户还在说"的窗口。

历史：能记多久记多久（token 预算 `--max-context-tokens` 默认 40000），逼近时后台
线程调 LLM 压缩旧历史为摘要（最近 6 条原样保留）。

机密：API key 放 `dialogue/config.local.json`（已 gitignore，**绝不提交**），
或环境变量 DEEPSEEK_API_KEY，或 `--api-key`。

用法（中文输出需 UTF-8）：
    PYTHONIOENCODING=utf-8 python examples/voice_dialogue.py [--asr-device cuda] [--tts-device cuda]
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r"E:\temp\voice0")    # voice0（只读引用，不改它任何文件）

# 缓存/镜像重定向（必须在 import asr/tts 之前）：voice0 权重全在 voice0/.cache，
# 若走默认 HF 缓存会漏 checkpoint.pth 从 hf-mirror 重下 208M（2026-08-29 实测卡 46 分钟）。
# HF_HUB_DISABLE_XET：MeloTTS-Chinese 已被切成 Xet 存储，新版 huggingface_hub 会把
# xet 仓库当"未缓存"绕开本地权重直接重下——禁用后正常命中缓存（0.55s 返回）。
os.environ["MODELSCOPE_CACHE"] = os.path.join(           # voice1 paraformer 权重
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "modelscope")
os.environ["HF_HOME"] = r"E:\temp\voice0\.cache\hf"      # voice0 melo/BERT 权重
os.environ["NLTK_DATA"] = r"E:\temp\voice0\.cache\nltk_data"   # melo g2p 语料
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"      # 首次下载走镜像
os.environ["HF_HUB_DISABLE_XET"] = "1"                   # 见上：xet 仓库会绕开缓存重下
import numpy as np                       # noqa: E402
import sounddevice as sd                 # noqa: E402
from asr import RealtimeASR              # noqa: E402  voice1
from tts import RealtimeTTS              # noqa: E402  voice0
from dialogue import DialogueController, OpenAICompatibleClient  # noqa: E402
from dialogue.mic import MicAGC, check_mic_signal, pick_input_device  # noqa: E402
from asr.core.audio import resample_to   # noqa: E402

SR = 16000


class _Console:
    """ASR 句 / AI 句两行类型的原地刷新状态机。

    - 同类型未定稿 → \\r 原地刷新（不换行刷屏）；
    - 切类型或上一行已定稿 → 先 \\n 再写新行；
    - finalize 标记当前行定稿（下次任何输出先换行）。
    """
    def __init__(self):
        self.kind = None     # "asr" | "ai"
        self.dirty = False
        self.final = False
        self._lock = threading.Lock()   # on_user(ASR线程) 与 on_ai_*(LLM线程) 并发写 → 加锁

    def _begin(self, kind):
        if self.dirty and (self.kind != kind or self.final):
            sys.stdout.write("\n")
            self.dirty = False
        self.kind = kind
        self.final = False

    def update(self, kind, text):
        with self._lock:
            self._begin(kind)
            sys.stdout.write("\r" + text)
            self.dirty = True
            sys.stdout.flush()

    def finalize(self, kind, text):
        with self._lock:
            self._begin(kind)
            sys.stdout.write("\r" + text)
            self.dirty = True
            self.final = True
            sys.stdout.flush()

    def newline(self):
        with self._lock:
            if self.dirty:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.dirty = False
                self.kind = None
                self.final = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-device", default="cuda", help="auto|cpu|cuda（ASR）")
    ap.add_argument("--tts-device", default="cuda", help="auto|cpu|cuda（TTS）")
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                    help="ASR 流式（默认开；--no-streaming 退化为整句）")
    ap.add_argument("--input-device", default=None, help="麦克风设备：序号或名称子串")
    ap.add_argument("--hotword-file", default=None, help="热词文件路径（引擎级拼音纠错）")
    ap.add_argument("--debug", action="store_true", help="打印调试信息（含后端框架输出）")
    ap.add_argument("--llm-model", default=None, help="LLM 模型名（默认 config.local.json 或 deepseek-chat）")
    ap.add_argument("--base-url", default=None, help="LLM base_url（默认 config.local.json 或 https://api.deepseek.com）")
    ap.add_argument("--api-key", default=None, help="LLM API key（默认 config.local.json 或 DEEPSEEK_API_KEY）")
    ap.add_argument("--interrupt-words", default="停下",
                    help="打断词（逗号分隔，默认\"停下\"）：命中即终止 LLM+TTS 输出")
    ap.add_argument("--reply-hold", type=float, default=0.35,
                    help="LLM 首句 hold-off 秒：给用户续句打断机会（0 关闭，默认 0.35）")
    ap.add_argument("--echo-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="回声门控：TTS 播放期只喂\"停下\"KWS、不识别（默认开；"
                         "--no-echo-gate 播放期可正常打断但接受回声，适合耳机）")
    ap.add_argument("--max-context-tokens", type=int, default=40000,
                    help="上下文 token 预算：超阈值触发历史压缩（默认 40000）")
    ap.add_argument("--max-history", type=int, default=0,
                    help="历史硬上限条数（0=不限制，由 token 预算管理）")
    ap.add_argument("--system-prompt", default=None, help="可选：系统提示词文件路径（UTF-8）")
    args = ap.parse_args()

    # ---- 麦克风 ----
    input_idx, dev_sr = pick_input_device(args.input_device)
    dev = sd.query_devices(input_idx)
    print("使用输入设备 [%d] %s（原生采样率 %d Hz）[流式]"
          % (input_idx, dev["name"], dev_sr), flush=True)
    rms = check_mic_signal(input_idx, dev_sr)
    if rms is None:
        print("错误：打不开设备 %s 的输入流，请换 --input-device。" % dev["name"], flush=True)
        sys.exit(1)
    if rms < 1e-4:
        print("警告：设备 %s 几乎采不到声音（RMS %.1f dBFS）——可能是摄像头无内置麦克风"
              "或麦克风被禁用，请换 --input-device 或接入真实麦克风。"
              % (dev["name"], 20 * np.log10(rms + 1e-12)), flush=True)
        sys.exit(1)
    print("麦克风信号正常（RMS %.1f dBFS）" % (20 * np.log10(rms + 1e-12)), flush=True)

    # ---- LLM（读 config.local.json / 环境变量 / CLI）----
    system_prompt = None
    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    llm = OpenAICompatibleClient(api_key=args.api_key, base_url=args.base_url,
                                 model=args.llm_model)
    masked = (llm._api_key[:4] + "…" + llm._api_key[-4:]) if len(llm._api_key) > 8 else "***"
    print("LLM: %s / %s（key %s，读本地配置文件，不提交 git）"
          % (llm._base_url, llm._model, masked), flush=True)

    # ---- 引擎 ----
    interrupt_words = [w.strip() for w in args.interrupt_words.split(",") if w.strip()] or None
    asr = RealtimeASR(backend="paraformer", device=args.asr_device, streaming=args.streaming,
                      profile=True, hotword_file=args.hotword_file, debug=args.debug,
                      interrupt_words=interrupt_words)
    tts = RealtimeTTS(device=args.tts_device, backend="melo", mode="queue",
                      profile=True, debug=args.debug)
    ctrl = DialogueController(llm, tts, system_prompt=system_prompt,
                              max_history_messages=args.max_history or None,
                              reply_hold=args.reply_hold,
                              max_context_tokens=args.max_context_tokens)

    # ---- 控制台 + 回调接线 ----
    con = _Console()
    asr_last = [""]

    def on_partial(p):
        if p.text and p.text != asr_last[0]:
            asr_last[0] = p.text
            con.update("asr", p.text)               # 流式出字：原地刷新，不换行

    def on_user(r):
        con.finalize("asr", "[%.2f-%.2fs] %s"
                     % (r.audio_start, r.audio_end, r.text))

    def on_ai_delta(_delta, full):
        con.update("ai", "AI: " + full)             # AI 流式：原地刷新

    def on_ai_sentence(_s):
        con.newline()                               # 这句已送 TTS（开播）→ 换行定稿

    asr.on_partial(on_partial)
    asr.on_sentence(lambda r: ctrl.feed_asr_sentence(r))
    asr.on_interrupt(ctrl.hard_stop)                # 用户说"停下"→ 立即终止 LLM+TTS
    ctrl.register_callbacks(on_user=on_user, on_ai_delta=on_ai_delta,
                            on_ai_sentence=on_ai_sentence)
    # 注意 on_ai_delta 由 LLM 线程回调，控制台写入线程安全（单写者串行 + 锁由 controller 保证时序）

    print("就绪！开始对话（Ctrl+C 退出）。打断词=%s，回声门控=%s"
          % (interrupt_words, "开" if args.echo_gate else "关"), flush=True)
    tts.submit("你好，我在听。")                    # 非阻塞：后台播就绪语
    agc = MicAGC()

    def cb(indata, frames, t, status):
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if dev_sr != SR:
            mono = resample_to(mono, dev_sr, SR)
        mono = agc.apply(mono)
        if args.echo_gate and ctrl.tts_busy:
            asr.ingest_kws_only(mono)              # TTS 播放期：只听"停下"，回声不进识别
        else:
            asr.ingest(mono, source_ts=time.monotonic())

    try:
        with sd.InputStream(samplerate=dev_sr, channels=1, device=input_idx, callback=cb):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        print("\n错误：打不开输入设备 %s 的音频流：%s" % (dev["name"], e), flush=True)
    finally:
        asr.close()
        ctrl.close()
        tts.close()
        if con.dirty:
            sys.stdout.write("\n")
        print("已退出。", flush=True)


if __name__ == "__main__":
    main()
