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

句末合并窗口（`--merge-window` 默认 1500ms）：ASR 断句后**不立即发 LLM**，等窗口内
有无补句——有则并入本轮一起发（窗口过期才发 LLM）。这是对"VAD 尾长"的根治：人说话
中间组织语言的停顿（常 1s+）任何固定尾长都拆不干净，合并窗口让"我每天晚上"+"下班回来
就是刷视频"合并成一整轮送进 LLM，AI 不再抢答残句。代价是每轮回复晚启动 ≈窗口时长。
多碎片显示为多条 `[时间戳]` 定稿行，但只触发**一次** `→ LLM 请求中…`、回复一份完整答案。

回声（v1 半双工门控）：TTS 播放期 mic 只喂 `ingest_kws_only`（"停下"KWS），回声不进
识别 → 无反馈自答；播放期普通说话被忽略。门控刚开启有**滚动 grace**（`--echo-guard`
默认 1200ms）：回声还没到（首句仍在合成）的窗口内，只要块有语音能量就顺延"仍喂正常
识别"到 现在+静音尾，让"AI 开答瞬间用户还没说完的尾巴"走完 VAD 静音尾定稿（否则被切去
KWS-only、句子悬成 partial 被吞）；回声一到由 `--echo-guard` 硬上限兜住不自答。
`--no-echo-gate` 可关（耳机近场）。首句 hold-off（`--reply-hold`）默认已关（0）：
post-commit barge 已接管续句打断，hold 只保护首句边界后 0.35s 内续句的极窄窗口，代价是
每轮首包音频固定 +0.35s——不划算，故默认 0。

休眠/对话状态机（`--wake-word`，默认开）：
    程序启动默认休眠（SLEEP）：mic 只喂唤醒词 KWS，其余一律不喂 ASR → 说什么都不识别。
    命中唤醒词 → 进对话（ACTIVE），播就绪语"在的，我在听"；唤醒词、就绪语、告别语都
    **不入对话历史/LLM**（就绪语、告别语走直连 tts.submit，不经 controller）。
    对话期保持原"麦克风→ASR→LLM→TTS"全链路（含回声门控/打断/post-commit barge）。
    结束有两个途径：(a) 说退出词（默认"拜拜"，仅 AI 沉默时可说——AI 播放期只喂"停下"
    听不到它）→ 播"好的，我先退下啦…"回休眠；(b) 静默超时 `--inactive-timeout` 默认
    60s 无用户语音 → 播"一直不说话，我先退下啦…"回休眠。唤醒/退出词都逗号分隔多词。
    `--wake-word ""` 关闭状态机 → 启动即对话（旧行为）。对话历史跨休眠保留：同一次
    程序运行内休眠不清历史，只有程序重启才重建本地 session 存档。

拆句根治（零固定延迟）：残句定稿立即发 LLM（无合并延迟）；AI 已答完、但音频还没开播
（`--post-commit-window` 默认 1500ms，≈melo 首句合成延迟）时用户补句 → 控制器撤下刚
答的残句答复、残句+新句连同历史重发。音频已开播后的续句才是新轮。

历史：能记多久记多久（token 预算 `--max-context-tokens` 默认 40000），逼近时后台
线程调 LLM 压缩旧历史为摘要（最近 6 条原样保留）。

机密：API key 放 `dialogue/config.local.json`（已 gitignore，**绝不提交**），
或环境变量 DEEPSEEK_API_KEY，或 `--api-key`。

用法（中文输出需 UTF-8）：
    PYTHONIOENCODING=utf-8 python examples/voice_dialogue.py [--asr-device cuda] [--tts-device cuda]
"""
import argparse
import json
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
from dialogue.wake import WakeSession, ACTIVE          # noqa: E402  休眠/对话状态机
from asr.core.audio import resample_to   # noqa: E402

SR = 16000


def dump_session(path, ctrl):
    """把控制器当前对话状态完整写入 path（原子：先写 .tmp 再 rename，写一半崩溃不损坏）。

    `ctrl.snapshot()` 锁内浅拷贝后即释放锁，磁盘 IO 在这里（调用方线程）做——
    周期性存档**不阻塞** LLM 读流线程。
    """
    snap = ctrl.snapshot()
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": snap["system"],          # 永不压缩，完整保留
        "compressed_summary": snap["summary"] or "",
        "history": snap["history"],               # [{"role","content"}, ...] 已 commit 轮次
        "user_turn_in_progress": snap["user_turn"],
        "assistant_in_progress": snap["assistant_full"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class _Console:
    """ASR 句 / AI 句两行类型的原地刷新状态机。

    - 同类型未定稿 → \\r 原地刷新（不换行刷屏）；
    - 切类型或上一行已定稿 → 先 \\n 再写新行；
    - finalize 标记当前行定稿（下次任何输出先换行）。
    - 新文本短于旧文本时补空格清尾随残留（否则 \r 覆盖不干净，如"→ LLM 请求中…"
      被"AI: 啊"覆盖会残留"请求中…"）。
    """
    def __init__(self):
        self.kind = None     # "asr" | "ai"
        self.dirty = False
        self.final = False
        self._last_len = 0   # 当前行已写列数（清尾随残留用）
        self._lock = threading.Lock()   # on_user(ASR线程) 与 on_ai_*(LLM线程) 并发写 → 加锁

    def _begin(self, kind):
        if self.dirty and (self.kind != kind or self.final):
            sys.stdout.write("\n")
            self.dirty = False
            self._last_len = 0
        self.kind = kind
        self.final = False

    def _write_line(self, text):
        """原地刷新当前行；新文本短于旧文本 → 补空格盖掉残留，再回卷列 0。"""
        prev = self._last_len
        sys.stdout.write("\r" + text)
        if prev > len(text):
            sys.stdout.write(" " * (prev - len(text)))
            sys.stdout.write("\r")
        sys.stdout.flush()
        self._last_len = len(text)

    def update(self, kind, text):
        with self._lock:
            self._begin(kind)
            self._write_line(text)
            self.dirty = True

    def finalize(self, kind, text):
        with self._lock:
            self._begin(kind)
            self._write_line(text)
            self.dirty = True
            self.final = True

    def newline(self):
        with self._lock:
            if self.dirty:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.dirty = False
                self.kind = None
                self.final = False
                self._last_len = 0

    def status(self, text):
        """打一行独立状态（门控等提示）：先收尾当前行，再换行写 text。"""
        with self._lock:
            if self.dirty:
                sys.stdout.write("\n")
                self.dirty = False
                self.kind = None
                self.final = False
                self._last_len = 0
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            self._last_len = 0


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
    ap.add_argument("--reply-hold", type=float, default=0.0,
                    help="首句 hold-off 秒（默认 0=关）：只在首句边界后 N 秒内续句才省得下这次"
                         "播报，窗口极窄且每轮固定 +N 秒，续句打断已由 --post-commit-window"
                         " 零延迟兜底，别开")
    ap.add_argument("--vad-tail", type=int, default=600,
                    help="你停多久算「说完」ms（默认 600）：停 600ms 静音就判定这句话说完、"
                         "发给 LLM。越小越灵敏但容易在你句中停顿处误判拆句；组织语言的停顿"
                         "常超 1s，拆句由 --post-commit-window 零延迟兜底，别一味调大它")
    ap.add_argument("--merge-window", type=int, default=0,
                    help="句末合并窗口 ms（默认 0=关闭）：ASR 断句后等这么久没新句才发 LLM，"
                         "窗口内补句并入本轮。0 关闭（每句立即发，推荐，拆句由 --post-commit-window"
                         " 零延迟处理）；>0 是固定延迟方案，越大越稳但每轮回复启动越慢")
    ap.add_argument("--post-commit-window", type=int, default=1500,
                    help="AI 已答完但音频还没播时的补话窗口 ms（默认 1500）：这 1.5s 是 TTS "
                         "合成音频的空档（喇叭没声）。你在这空档补话 → 取消这段音频、撤下刚对"
                         "残句的答复、整句+历史重发（控制台打 [合并]）。你没说话则什么都不发生，"
                         "不增加延迟。≈melo 首句合成延迟；音频开播后的续句视为新轮。"
                         "CUDA 机器偏快可调 900，常漏合并调 2000")
    ap.add_argument("--echo-guard", type=int, default=1200,
                    help="AI 开口后麦克风还听正常语音的时长 ms（默认 1200）：回声要 ~1.2s 才"
                         "传到麦克风，这段时间照常识别（抓你没说完的尾巴），过后只认「停下」"
                         "防 AI 自答回声。调大尾巴更易被抓但自答风险略增，宁小勿大")

    ap.add_argument("--echo-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="回声门控：TTS 播放期只喂\"停下\"KWS、不识别（默认开；"
                         "--no-echo-gate 播放期可正常打断但接受回声，适合耳机）")
    ap.add_argument("--wake-word", default="小爱小爱",
                    help="唤醒词（逗号分隔，默认\"小爱小爱\"）：休眠期只听这些词，命中才进入"
                         "对话。传空串（--wake-word \"\"）关闭唤醒 → 启动即对话（旧行为）")
    ap.add_argument("--exit-words", default="拜拜",
                    help="退出词（逗号分隔，默认\"拜拜\"）：对话期听到该词即播告别语并回休眠，"
                         "退出词本身不入对话历史。仅 AI 沉默时生效（AI 播放期只喂\"停下\"）")
    ap.add_argument("--inactive-timeout", type=int, default=60,
                    help="静默超时秒（默认 60）：对话期无用户语音持续这么久 → 播告别语并回休眠"
                         "（告别语不入历史）。0=关闭自动休眠")
    ap.add_argument("--vad-threshold-db", type=float, default=-35.0,
                    help="VAD 断句能量门槛 dB（默认 -35）：离麦克风远说话够不着门槛就不定稿"
                         "提交（只有 partial 不出字）。调低（如 -42）提升灵敏度，但环境噪声"
                         "更易误断句；配合 MicAGC 上限 8x 增益使用")
    ap.add_argument("--max-context-tokens", type=int, default=40000,
                    help="上下文 token 预算：超阈值触发历史压缩（默认 40000）")
    ap.add_argument("--max-history", type=int, default=0,
                    help="历史硬上限条数（0=不限制，由 token 预算管理）")
    ap.add_argument("--system-prompt", default=None,
                    help="可选：系统提示词文件路径（UTF-8）。系统提示词永不压缩、永远放在消息"
                         "最开头（历史压缩只动历史，摘要拼在系统提示词之后）")
    ap.add_argument("--history-dump", action=argparse.BooleanOptionalAction, default=True,
                    help="本地会话存档（默认开）：把完整对话历史（系统提示词+压缩摘要+全部"
                         "轮次）每 --history-dump-interval 秒覆盖写到 "
                         "--history-dump-dir/session_<启动时间>.json；每次启动新建一个文件，"
                         "程序退出再写一次；后台线程写盘不阻塞对话")
    ap.add_argument("--history-dump-dir", default="sessions",
                    help="会话存档目录（默认 sessions/，相对当前目录；已 gitignore）")
    ap.add_argument("--history-dump-interval", type=int, default=300,
                    help="会话存档间隔秒（默认 300=5 分钟；0=只退出时写一次）")
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
    wake_words = [w.strip() for w in args.wake_word.split(",") if w.strip()] or None
    exit_words = [w.strip() for w in args.exit_words.split(",") if w.strip()] or None
    asr = RealtimeASR(backend="paraformer", device=args.asr_device, streaming=args.streaming,
                      vad_silence_tail_ms=args.vad_tail,   # 默认 600ms：别在句中停顿处拆句
                      vad_threshold_db=args.vad_threshold_db,   # 默认 -35dB：调低提升远距离灵敏度
                      profile=True, hotword_file=args.hotword_file, debug=args.debug,
                      interrupt_words=interrupt_words)
    tts = RealtimeTTS(device=args.tts_device, backend="melo", mode="queue",
                      profile=True, debug=args.debug)
    ctrl = DialogueController(llm, tts, system_prompt=system_prompt,
                              max_history_messages=args.max_history or None,
                              reply_hold=args.reply_hold,
                              merge_window=args.merge_window / 1000.0,
                              post_commit_window=args.post_commit_window / 1000.0,
                              max_context_tokens=args.max_context_tokens)

    # ---- 控制台 + 回调接线 ----
    con = _Console()
    asr_last = [""]

    # ---- 休眠/对话状态机（唤醒功能）----
    wake = WakeSession(wake_enabled=bool(wake_words), inactive_timeout=args.inactive_timeout)
    wake_det = None
    if wake_words:
        try:
            from asr.kws.interrupt import get_interrupt_detector
            wake_det = get_interrupt_detector("sherpa", words=wake_words)
            wake_det.load()
            print("[唤醒] 唤醒词就绪：%s（启动即休眠，命中才对话）" % wake_words, flush=True)
        except Exception as e:
            print("[唤醒] 唤醒词检测加载失败（唤醒关闭，启动即对话）：%s" % e, flush=True)
            wake_det = None
            wake.state = ACTIVE              # 无唤醒检测器 → 不能休眠，直接对话

    def _say(text):
        """播就绪语/告别语（直连 TTS，不经 controller → 不入历史/LLM）。Job 记入
        wake.self_talk，mic 回调把它当"自播回声"门控——"在的，我在听"不会被识别成
        你说的话再提交一轮。"""
        if text is None:
            return
        try:
            wake.set_self_talk(tts.submit(text))
        except Exception as e:
            print("[自播] 语音播放失败：%s" % e, flush=True)

    def on_partial(p):
        wake.note_partial()                        # 有语音出字 → 用户在说话（任意距离都算）
        if p.text and p.text != asr_last[0]:
            asr_last[0] = p.text
            # "… "前缀 = 仍在出字、未定稿（不会提交）；定稿后 on_user 的时间戳覆盖它
            con.update("asr", "… " + p.text)

    def on_user(r):
        con.finalize("asr", "[%.2f-%.2fs] %s"
                     % (r.audio_start, r.audio_end, r.text))

    def on_ai_delta(_delta, full):
        con.update("ai", "AI: " + full)             # AI 流式：原地刷新

    def on_ai_sentence(_s):
        con.newline()                               # 这句已送 TTS（开播）→ 换行定稿

    def on_llm_start():
        # 定稿句 → LLM 请求已发出（等待首 token）；首 delta 到来时被 "AI: " 原地覆盖
        con.update("ai", "→ LLM 请求中…")

    def on_llm_error(e):
        con.update("ai", "× LLM 出错: %s" % str(e).strip()[:200])

    def on_merge_rollback():
        # post-commit barge：AI 已答完但音频未开播，用户补了尾巴 → 撤答复重答
        con.status("[合并] 撤回了刚才的答复，正在重答完整问题…")

    asr.on_partial(on_partial)

    def on_sentence(r):
        # 定稿句统一入口：休眠期防御返回；退出词拦截（不入历史/LLM）；其余正常提交。
        if wake.sleeping:                  # 休眠期不应有定稿句（没喂 ASR）——防御
            return
        if exit_words and any(w in r.text for w in exit_words):
            on_user(r)                     # 退出词照常显示（识别事实），但不进历史/LLM
            ctrl.hard_stop()               # 清理在途 LLM/TTS（已 commit 历史保留）
            _say(wake.go_sleep("bye"))     # 告别语回休眠（不入历史）
            return
        ctrl.feed_asr_sentence(r)

    asr.on_sentence(on_sentence)
    asr.on_interrupt(ctrl.hard_stop)                # 用户说"停下"→ 立即终止 LLM+TTS
    ctrl.register_callbacks(on_user=on_user, on_ai_delta=on_ai_delta,
                            on_ai_sentence=on_ai_sentence,
                            on_llm_start=on_llm_start, on_llm_error=on_llm_error,
                            on_merge_rollback=on_merge_rollback)
    # 注意 on_ai_delta 由 LLM 线程回调，控制台写入线程安全（单写者串行 + 锁由 controller 保证时序）

    print("就绪！开始对话（Ctrl+C 退出）。打断词=%s，回声门控=%s，VAD尾长=%dms，"
          "post-commit窗口=%dms，回声防护=%dms"
          % (interrupt_words, "开" if args.echo_gate else "关", args.vad_tail,
             args.post_commit_window, args.echo_guard),
          flush=True)

    # ---- 本地会话存档（后台线程，每 interval 覆盖写；不阻塞 LLM/识别/TTS）----
    dump_path = None
    dump_stop = threading.Event()
    dumper_thread = None
    if args.history_dump and args.history_dump_dir:
        os.makedirs(args.history_dump_dir, exist_ok=True)
        dump_path = os.path.join(args.history_dump_dir,
                                 "session_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        print("会话存档：每 %ds 更新到 %s（退出时再写一次）"
              % (args.history_dump_interval, dump_path), flush=True)

        def _dumper():
            while True:
                try:
                    dump_session(dump_path, ctrl)
                except Exception as e:
                    print("[存档] 写入失败: %s" % e, flush=True)
                if dump_stop.wait(args.history_dump_interval):
                    return

        if args.history_dump_interval > 0:
            dumper_thread = threading.Thread(target=_dumper, name="dialogue-dump",
                                             daemon=True)
            dumper_thread.start()
        else:
            dump_session(dump_path, ctrl)          # 间隔 0：只退出时写，先建文件占位

    if wake.sleeping:
        con.status("休眠中，随时唤醒我哦~")          # 唤醒开：启动即休眠，只说唤醒词才对话
    else:
        tts.submit("你好，我在听。")                # 唤醒关（旧行为）：启动即对话
    agc = MicAGC()

    gate_on = [False]          # 回声门控状态（跨回调跟踪转换，mic 回调里检测）
    gate_since = [0.0]         # 门控开启时刻（monotonic）
    grace_until = [0.0]        # 滚动 grace 截止（monotonic）：有语音能量就顺延
    VAD_S = args.vad_tail / 1000.0
    ECHO_GUARD = args.echo_guard / 1000.0          # 回声最早到达前的硬上限
    SPEECH_POW = 10.0 ** (-38.0 / 10.0)            # 扩展阈值：稍比 VAD(-35dB) 灵敏

    def cb(indata, frames, t, status):
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if dev_sr != SR:
            mono = resample_to(mono, dev_sr, SR)
        mono = agc.apply(mono)                     # 先 AGC：唤醒词与远距离说话同灵敏度
        rms2 = float(np.mean(mono * mono)) + 1e-12
        now = time.monotonic()

        # ---- 休眠：只听唤醒词，其余一律不喂 ASR（说什么都不识别）----
        if wake.sleeping:
            if wake_det is not None:
                try:
                    if wake_det.feed(mono):
                        _say(wake.on_wake())       # 就绪语直连 TTS（不入历史/LLM）
                        con.status("[唤醒] 已唤醒，开始对话")
                except Exception:
                    pass                          # 检测器异常不致命，跳过本块
            return

        # ---- 对话中 ----
        busy = bool(args.echo_gate and ctrl.tts_busy)
        decision = wake.feed_decision(now, rms2, SPEECH_POW, busy)
        if decision == "none":                     # 静默超时 → 已回休眠
            ctrl.hard_stop()                       # 清理在途 LLM/TTS（已 commit 历史保留）
            return
        if decision == "kws_only":
            # 自播语音（就绪语/告别语）回声门控：只喂"停下"，自播语音不进识别——否则
            # "在的，我在听"会被当成你说的话提交/入历史。这里不开滚动 grace：grace 会
            # 把自播回声当"用户尾巴"喂进识别。
            asr.ingest_kws_only(mono)
            return

        if busy != gate_on[0]:                     # 门控转换 → 状态行，让用户知道何时在听
            gate_on[0] = busy
            gate_since[0] = now if busy else 0.0
            grace_until[0] = (now + VAD_S) if busy else 0.0
            if busy:
                con.status("[门控] AI 播放中，此刻说话只会被当\"停下\"监听")
            else:
                con.status("[门控] 播放结束，可以说话了")
        if gate_on[0]:
            # 滚动 grace：回声还没到（首句仍在合成）的窗口内，只要块有语音能量就顺延
            # "仍喂正常识别"期限到 现在+静音尾+0.2s → 让"AI 开答瞬间用户还没说完的尾巴"
            # 走完 VAD 静音尾定稿（否则被切去 KWS-only，句子悬成 partial 被吞）。回声一到
            # 由 ECHO_GUARD 硬上限兜住，不会无限顺延自答。
            if rms2 > SPEECH_POW and (now - gate_since[0]) < ECHO_GUARD:
                grace_until[0] = now + VAD_S + 0.2
            in_grace = now < grace_until[0] and (now - gate_since[0]) < ECHO_GUARD
        else:
            in_grace = False
        if gate_on[0] and not in_grace:
            asr.ingest_kws_only(mono)              # TTS 播放期：只听"停下"，回声不进识别
        else:
            asr.ingest(mono, source_ts=now)

    try:
        with sd.InputStream(samplerate=dev_sr, channels=1, device=input_idx, callback=cb):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        print("\n错误：打不开输入设备 %s 的音频流：%s" % (dev["name"], e), flush=True)
    finally:
        if con.dirty:                              # 先收尾在途控制台行，避免后续打印穿插
            sys.stdout.write("\n")
            con.dirty = False
        if dump_path is not None:
            dump_stop.set()                        # 停后台存档线程（写完整再退）
            if dumper_thread is not None:
                dumper_thread.join(timeout=2)
            try:
                dump_session(dump_path, ctrl)      # 程序终止前最后写一次
                print("已写会话存档：%s" % dump_path, flush=True)
            except Exception as e:
                print("[存档] 退出写入失败: %s" % e, flush=True)
        if wake_det is not None:
            wake_det.close()
        asr.close()
        ctrl.close()
        tts.close()
        print("已退出。", flush=True)


if __name__ == "__main__":
    main()
