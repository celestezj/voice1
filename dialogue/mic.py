# -*- coding: utf-8 -*-
"""麦克风采集基建（语音对话主程序复用；record_mic.py 同源实现）。

- `MicAGC`：麦克风自适应增益——说话电平够不着 VAD 门限（-35dB）时自动放大
  （目标 peak≈0.3，只放大不压小，上限 8x）。引擎层故意不归一，mic 层负责。
- `pick_input_device`：校验并挑选录音设备（序号或名称子串；无输入设备友好退出）。
- `check_mic_signal`：实测 1 秒信号电平，区分真麦克风 vs 无效/被禁用的输入设备。
"""
import sys
import time

import numpy as np
import sounddevice as sd


class MicAGC:
    """麦克风自适应增益：把说话电平抬到 VAD/模型健康区间（目标 peak≈0.3）。

    背景（T17c 实测）：VAD 断句门限 EnergyVAD.threshold_db = -35dB；本机 HD Audio
    麦克风说话 RMS ≈ -36~-46dBFS，大多低于门限 → VAD 几乎不断句 →"对着麦说话没
    反应"。解法：放大到目标电平再喂引擎。

    - 快攻慢放：以块峰值跟踪，说话立即抬增益，静音回落缓慢（防增益抽吸）；
    - **只放大不压小**（min_gain=1x）：响亮麦克风原样通过；
    - 上限 8x（+18dB）：静音底噪即使放满也只有约 -50dB，仍低于门限，不误断句。
    """
    def __init__(self, target_peak=0.3, max_gain=8.0, release=0.95):
        self._target = float(target_peak)
        self._max_gain = float(max_gain)
        self._release = float(release)
        self._peak = 1e-6

    def apply(self, block):
        x = np.asarray(block, dtype=np.float32)
        if len(x) == 0:
            return x
        p = float(np.max(np.abs(x))) + 1e-9
        self._peak = max(self._peak * self._release, p)   # 快攻（立即取新峰值）/慢放（按 release 回落）
        gain = self._target / self._peak
        gain = min(max(gain, 1.0), self._max_gain)        # 只放大，上限 8x
        return x * np.float32(gain)


def list_devices():
    """打印全部音频设备（输入通道 >0 即可作麦克风），供排障/挑选。"""
    print("当前音频设备：")
    for i, d in enumerate(sd.query_devices()):
        print("  [%d] %s（输出%d / 输入%d）"
              % (i, d["name"], d["max_output_channels"], d["max_input_channels"]))


def check_mic_signal(device, sr, seconds=1.0):
    """开 1 秒采集实测设备信号电平，返回 RMS（float，0 表示静音）。"""
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
    """校验并选择录音输入设备，返回 (设备索引, 原生采样率)。无输入设备 → 退出。"""
    devs = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    if not inputs:
        print("错误：未检测到任何麦克风输入设备。", flush=True)
        list_devices()
        print("请接入麦克风/带麦耳机/USB 声卡后重试，或用 --input-device <序号> 显式指定。")
        sys.exit(1)
    if arg is None:
        try:
            idx = sd.query_devices(kind="input")["index"]
        except (ValueError, sd.PortAudioError):
            idx = inputs[0][0]
    elif str(arg).isdigit():
        idx = int(arg)
        if not (0 <= idx < len(devs) and devs[idx]["max_input_channels"] > 0):
            print("错误：--input-device %s 不是输入设备。可用设备见下：" % arg)
            list_devices()
            sys.exit(1)
    else:
        hits = [i for i, d in inputs if str(arg).lower() in d["name"].lower()]
        if not hits:
            print("错误：找不到名为 %r 的输入设备。可用设备见下：" % arg)
            list_devices()
            sys.exit(1)
        idx = hits[0]
    return idx, int(sd.query_devices(idx)["default_samplerate"])
