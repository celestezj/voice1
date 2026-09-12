# -*- coding: utf-8 -*-
"""示例：麦克风实时识别（sounddevice 采集 → 常驻引擎 → 逐句回调）。

启动前先**检测输入设备**：当前机器没有任何麦克风输入设备（台式机常见）→ 打印设备
清单并友好退出，不抛 PortAudioError 裸错。可用 `--input-device` 指定设备（序号或
名称子串）；非 16kHz 原生采样率的设备（如 48k 麦克风）自动重采样到引擎采样率。

**麦克风自适应增益（MicAGC）**：VAD 断句门限 -35dB，不少麦克风说话电平只有
-40dB 上下（录得到、但够不着门限 → "说话没反应"）。采集层自动放大到健康电平
（目标 peak≈0.3，只放大不压小，上限 24x）+ **噪声门控**（尾静音真静音 → 句子能
定稿，远距离说话也能提交）。引擎层故意不归一，mic 层负责。

用法（中文输出需 UTF-8 编码，Ctrl+C 退出）：
    PYTHONIOENCODING=utf-8 python examples/record_mic.py [--backend paraformer] [--device cuda] [--streaming] [--input-device <序号|名称>]
    --streaming：流式逐块出字（on_partial 边说边出 + 句末 flush 定稿，尾字延迟更低）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sounddevice as sd
from asr import RealtimeASR
from asr.core.audio import resample_to

SR = 16000


class MicAGC:
    """麦克风自适应增益：说话放大到 VAD 健康区间 + **噪声门控保证句子能定稿**。

    与 dialogue/mic.py 的 MicAGC 同源实现（voice_dialogue 主程序用 dialogue 版，
    record_mic 独立副本）——改这边必须同步那边。

    背景（T17c + 2026-09 远距离实测）：VAD 断句门限 -35dB。v1 只解决"近距说话电平
    够不着门限"（目标 peak 0.3、只放大不压小、上限 8x）。远距离（大客厅）实测两个问题：

     ① 说话电平低 → 8x（+18dB）不够，远距说话放大后仍悬在门限附近；
     ② **尾静音被 AGC 慢放放大 → 句子永不收尾**：说话结束后增益停在放大远距说话
        所需的高位，房间底噪 × 高增益后 ≥ -35dB → VAD 把底噪当"还在说话"，静音尾
        永远凑不满 → 句子永不定稿、只出 partial 不提交（"说完了还在等我继续输入"）。

    修复：**锁存噪声门控**（自适应底噪 × margin 以下、**持续 ≥120ms** 才输出静音 →
    尾静音真静音 → VAD 静音尾正常累计 → 句子正常定稿；说话时短暂弱音节不被切，避免
    半截话）+ **底噪只在锁存确认的真静音块上更新**（说话不动底噪，门限不会涨到吞掉
    句子尾巴）+ **上限提高到 24x**（远距说话能抬进门限）。保持"快攻慢放 + 只放大不压小"。
    """
    def __init__(self, target_peak=0.3, max_gain=24.0, release=0.95,
                 gate_margin=1.4, gate_floor_db=-58.0, gate_ceiling_db=-42.0,
                 noise_init_db=-50.0, noise_down=0.3, noise_up=0.05,
                 close_ms=120, sample_rate=16000):
        self._target = float(target_peak)
        self._max_gain = float(max_gain)
        self._release = float(release)
        self._gate_margin = float(gate_margin)
        self._gate_floor = 10.0 ** (gate_floor_db / 20.0)
        self._gate_ceiling = 10.0 ** (gate_ceiling_db / 20.0)
        self._sr = int(sample_rate)
        self._close_ms = float(close_ms)
        self._peak = 1e-6
        self._noise = 10.0 ** (noise_init_db / 20.0)
        self._noise_down = float(noise_down)
        self._noise_up = float(noise_up)
        self._quiet_ms = 0.0

    def apply(self, block):
        x = np.asarray(block, dtype=np.float32)
        if len(x) == 0:
            return x
        dur_ms = len(x) * 1000.0 / self._sr
        rms = float(np.sqrt(np.mean(np.square(x)))) + 1e-12

        # 噪声门限 = 底噪 × margin，夹在地板（太安静不误伤）与天花板（防说话自门）之间。
        gate = min(max(self._gate_floor, self._noise * self._gate_margin),
                   self._gate_ceiling)
        below = rms < gate
        if below:
            self._quiet_ms += dur_ms
        else:
            self._quiet_ms = 0.0

        if below and self._quiet_ms >= self._close_ms:
            # 锁存确认的静音（真尾静音 / 环境底噪）→ 输出静音，并且**只有这里才更新底噪**
            # 估计：说话期间的弱音节起伏即使低于门限也**不动底噪**——否则远距低信噪比说话
            # 把底噪估计慢慢抬高 → 门限跟着涨 → 说着说着连句子尾巴也被吞（半截话）。底噪
            # 更新只在"连续静音 ≥ 锁存时长"时发生，天然免疫说话。
            if rms < self._noise:
                self._noise += self._noise_down * (rms - self._noise)
            else:
                self._noise += self._noise_up * (rms - self._noise)
            # 关键：说话结束后增益停在放大远距说话所需的高位，底噪若继续放大必过 -35dB
            # 断句门限 → 句子永不收尾。锁存门控保证尾静音是真静音 → VAD 静音尾正常累计 →
            # 句子正常定稿提交。而说话时短暂的弱音节起伏（< close_ms 就恢复）**不被切**。
            self._peak *= self._release          # 静音块仍按 release 回落，增益恢复（下次快攻）
            return np.zeros_like(x)

        # 说话块（含短暂的弱音节起伏）：照常 AGC 放大——弱音节不被吞，避免半截话。
        p = float(np.max(np.abs(x))) + 1e-9
        self._peak = max(self._peak * self._release, p)   # 快攻（立即取新峰值）/慢放（按 release 回落）
        gain = self._target / self._peak
        gain = min(max(gain, 1.0), self._max_gain)        # 只放大，上限 max_gain
        return x * np.float32(gain)


def list_devices():
    """打印全部音频设备（输入通道 >0 即可作麦克风），供排障/挑选。"""
    print("当前音频设备：")
    for i, d in enumerate(sd.query_devices()):
        print("  [%d] %s（输出%d / 输入%d）"
              % (i, d["name"], d["max_output_channels"], d["max_input_channels"]))


def check_mic_signal(device, sr, seconds=1.0):
    """开 1 秒采集实测设备信号电平，返回 RMS（float，0 表示静音）。

    枚举只能证明"设备声称是输入"，不能证明"真采得到声音"——坏的/被禁用的麦克风照样
    会被枚举出来。本函数实测区分：真麦克风必有底噪（RMS 通常 >-70dBFS），无效/禁用
    设备近乎数字静音（RMS≈0）。打不开流返回 None。
    """
    buf = []
    def cb(indata, frames, t, status):
        buf.append(indata[:, 0].copy())
    try:
        with sd.InputStream(samplerate=sr, channels=1, device=device, callback=cb):
            time.sleep(seconds)
    except sd.PortAudioError:
        return None
    if not buf:
        return 0.0
    x = np.concatenate(buf)
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def pick_input_device(arg):
    """校验并选择录音输入设备，返回 (设备索引, 原生采样率)。

    无任何输入设备 → 打印清单并 sys.exit(1)（台式机无麦克风的场景）。
    """
    devs = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    if not inputs:
        print("错误：未检测到任何麦克风输入设备。", flush=True)
        list_devices()
        print("请接入麦克风/带麦耳机/USB 声卡后重试，或用 --input-device <序号> 显式指定。")
        sys.exit(1)
    if arg is None:                        # 默认：系统默认输入设备；无默认取第一个
        try:
            idx = sd.query_devices(kind="input")["index"]
        except (ValueError, sd.PortAudioError):
            idx = inputs[0][0]
    elif str(arg).isdigit():               # 按序号
        idx = int(arg)
        if not (0 <= idx < len(devs) and devs[idx]["max_input_channels"] > 0):
            print("错误：--input-device %s 不是输入设备。可用设备见下：" % arg)
            list_devices()
            sys.exit(1)
    else:                                  # 按名称子串（忽略大小写）
        hits = [i for i, d in inputs if str(arg).lower() in d["name"].lower()]
        if not hits:
            print("错误：找不到名为 %r 的输入设备。可用设备见下：" % arg)
            list_devices()
            sys.exit(1)
        idx = hits[0]
    return idx, int(sd.query_devices(idx)["default_samplerate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="paraformer", help="paraformer|whisper|sherpa")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--streaming", action="store_true",
                    help="流式逐块出字（on_partial 边说边出 + 句末 flush 定稿）")
    ap.add_argument("--input-device", default=None,
                    help="麦克风设备：序号或名称子串（默认系统默认输入设备）")
    ap.add_argument("--hotword-file", default=None,
                    help="热词文件路径（每行一个纠错项，拼音级；所有后端统一生效）")
    ap.add_argument("--debug", action="store_true",
                    help="打印调试信息（含后端框架输出，如 FunASR 的 rtf_avg 进度条）")
    args = ap.parse_args()

    input_idx, dev_sr = pick_input_device(args.input_device)
    dev = sd.query_devices(input_idx)
    print("使用输入设备 [%d] %s（原生采样率 %d Hz）%s"
          % (input_idx, dev["name"], dev_sr, "[流式]" if args.streaming else ""), flush=True)

    # 实测信号（区分真麦克风 vs 无效/被禁用的输入设备；坏设备在模型加载前就退出）
    rms = check_mic_signal(input_idx, dev_sr)
    if rms is None:
        print("错误：打不开设备 %s 的输入流，请换 --input-device。" % dev["name"], flush=True)
        sys.exit(1)
    if rms < 1e-4:                        # ~ -80dBFS：近乎数字静音 → 无效/禁用
        print("警告：设备 %s 几乎采不到声音（RMS %.1f dBFS）——可能是摄像头无内置麦克风"
              "或麦克风被禁用，请换 --input-device 或接入真实麦克风。"
              % (dev["name"], 20 * np.log10(rms + 1e-12)), flush=True)
        sys.exit(1)
    print("麦克风信号正常（RMS %.1f dBFS）" % (20 * np.log10(rms + 1e-12)), flush=True)

    asr = RealtimeASR(backend=args.backend, device=args.device, streaming=args.streaming,
                      profile=True, hotword_file=args.hotword_file, debug=args.debug)
    agc = MicAGC()                      # 麦克风自适应增益（说话电平过低时自动放大）

    # 控制台输出：
    # - 流式：partial 原地刷新（\r 不换行），句末定稿也原地覆盖，只有新一句开始才换行；
    # - 非流式：每句独立一行（靠左）。
    dirty = [False]        # 当前行有进行中的输出（流式用）
    if args.streaming:
        last = [""]
        finalized = [False]    # 当前行已是一句定稿（下次输出须先换行）
        def _emit(text):
            if finalized[0] or not dirty[0]:
                sys.stdout.write("\n")      # 新一句开始 / 空行 → 先换行
                dirty[0] = True
                finalized[0] = False
            sys.stdout.write("\r" + text)
            sys.stdout.flush()
        def _partial(p):
            if p.text and p.text != last[0]:          # 累计文本未变则跳过
                last[0] = p.text
                _emit(p.text)                         # 原地刷新当前行
        def _sentence(r):
            _emit("[%.2fs] %s (ttfb=%.3fs)" % (r.audio_end, r.text, r.ttfb))
            finalized[0] = True                       # 句末定稿：下句开始才换行
        asr.on_partial(_partial)
        asr.on_sentence(_sentence)
    else:
        asr.on_sentence(lambda r: print("[%.2fs] %s (ttfb=%.3fs)"
                                        % (r.audio_end, r.text, r.ttfb)))
    print("说话吧…（Ctrl+C 退出）[麦克风自动增益已启用，说话电平过低会自动放大]", flush=True)

    def cb(indata, frames, t, status):
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if dev_sr != SR:
            mono = resample_to(mono, dev_sr, SR)      # 非 16k 设备 → 重采样到引擎采样率
        asr.ingest(agc.apply(mono), source_ts=time.monotonic())   # 放大到健康电平再喂引擎

    try:
        with sd.InputStream(samplerate=dev_sr, channels=1, device=input_idx, callback=cb):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        print("错误：打不开输入设备 %s 的音频流：%s" % (dev["name"], e), flush=True)
        print("可尝试 --input-device <序号> 换一个设备，或检查设备是否被其他程序独占。")
    finally:
        asr.close()
        if dirty[0]:
            sys.stdout.write("\n")      # 流式中途退出：先换行脱离当前 partial 行
        print("已退出。", flush=True)


if __name__ == "__main__":
    main()
