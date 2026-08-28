# -*- coding: utf-8 -*-
"""示例：麦克风实时识别（sounddevice 采集 → 常驻引擎 → 逐句回调）。

启动前先**检测输入设备**：当前机器没有任何麦克风输入设备（台式机常见）→ 打印设备
清单并友好退出，不抛 PortAudioError 裸错。可用 `--input-device` 指定设备（序号或
名称子串）；非 16kHz 原生采样率的设备（如 48k 麦克风）自动重采样到引擎采样率。

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
                      profile=True, hotword_file=args.hotword_file)
    asr.on_sentence(lambda r: print("[%.2fs] %s (ttfb=%.3fs)"
                                    % (r.audio_end, r.text, r.ttfb)))
    if args.streaming:
        last = [""]
        def _partial(p):
            if p.text and p.text != last[0]:          # 累计文本未变则跳过（避免刷屏）
                last[0] = p.text
                print("[流式出字] %s" % p.text, flush=True)
        asr.on_partial(_partial)
    print("说话吧…（Ctrl+C 退出）", flush=True)

    def cb(indata, frames, t, status):
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if dev_sr != SR:
            mono = resample_to(mono, dev_sr, SR)      # 非 16k 设备 → 重采样到引擎采样率
        asr.ingest(mono, source_ts=time.monotonic())

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
        print("已退出。", flush=True)


if __name__ == "__main__":
    main()
