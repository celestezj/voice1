# -*- coding: utf-8 -*-
"""麦克风采集基建（语音对话主程序复用；record_mic.py 同源实现）。

- `MicAGC`：麦克风自适应增益——说话电平够不着 VAD 门限（-35dB）时自动放大
  （目标 peak≈0.3，只放大不压小，上限 24x），并**噪声门控**保证尾静音真静音、
  句子正常定稿（远距离说话能提交 LLM 的核心）。引擎层故意不归一，mic 层负责。
- `pick_input_device`：校验并挑选录音设备（序号或名称子串；无输入设备友好退出）。
- `check_mic_signal`：实测 1 秒信号电平，区分真麦克风 vs 无效/被禁用的输入设备。
"""
import sys
import time

import numpy as np
import sounddevice as sd


class MicAGC:
    """麦克风自适应增益：说话放大到 VAD 健康区间 + **噪声门控保证句子能定稿**。

    背景（T17c + 2026-09 远距离实测）：VAD 断句门限 EnergyVAD.threshold_db = -35dB。
    v1 只解决"近距说话电平够不着门限"（目标 peak 0.3、只放大不压小、上限 8x）。
    远距离（大客厅）实测暴露两个问题：

     ① **说话电平低 → 8x（+18dB）不够**：远距说话 RMS ≈ -40~-47dBFS，放大后仍悬在
        门限附近时灵时不灵（旧补救 `--vad-threshold-db -42` 还让噪声更易误断句）；
     ② **尾静音被 AGC 慢放放大 → 句子永不收尾**（本次修复核心）：说话结束后 `_peak`
        按 release 慢回落、增益停在放大远距说话所需的高位（≈上限）；房间底噪 × 高增益后
        ≥ -35dB → VAD 把底噪当"还在说话"，静音尾永远凑不满 → 句子永不定稿、只出 partial
        不提交 LLM（"离远说完了还在等我继续输入"）。

    修复（v2，2026-09）：
    - **锁存噪声门控**：自适应环境底噪估计 + **持续低于门限 ~120ms 才置零**（锁存）。
      尾静音是真静音 → VAD 静音尾正常累计 → 句子正常定稿提交；而说话时**短暂的弱音节
      起伏（< 120ms）不被切**，模型能收到整句（否则远距说话弱音节被吞 → 半截话/出错）。
      门控同时让唤醒/打断 KWS 只看到干净语音。
    - **底噪只在锁存确认的真静音块上更新**（说话期间的弱音节即使低于门限也**不动底噪**）：
      远距低信噪比说话不会把底噪估计慢慢抬高、门限也就不会涨到把句子尾巴吞掉（实测
      -46dB 语音 + -48dB 底噪下，旧"每低于门限块都更新"方案把底噪从 -50 抬到 ~-43、
      门限反超语音均值 → 整句被切；锁存后仅在真静音更新 → 底噪恒在 -48、句子完整）。
    - **上限提高** 8x → 24x（+27.6dB）：远距说话能抬进 VAD 健康区间。门控保证底噪
      不会被一起抬上去。
    - 保持 v1 的"快攻慢放 + 只放大不压小"：响亮麦克风原样通过（gain≥1 截断）。
    """
    def __init__(self, target_peak=0.3, max_gain=24.0, release=0.95,
                 gate_margin=1.4, gate_floor_db=-58.0, gate_ceiling_db=-42.0,
                 noise_init_db=-50.0, noise_down=0.3, noise_up=0.05,
                 close_ms=120, sample_rate=16000):
        self._target = float(target_peak)
        self._max_gain = float(max_gain)
        self._release = float(release)
        self._gate_margin = float(gate_margin)          # 门限 = 底噪 × margin（≈ +3dB）
        self._gate_floor = 10.0 ** (gate_floor_db / 20.0)    # 门限绝对地板（太安静不误伤）
        self._gate_ceiling = 10.0 ** (gate_ceiling_db / 20.0)  # 门限天花板（防说话自门）
        self._sr = int(sample_rate)                     # AGC 输入恒为 16k（调用方已重采样）
        self._close_ms = float(close_ms)                # 门控锁存：持续低于门限这么久才置零
        self._peak = 1e-6
        self._noise = 10.0 ** (noise_init_db / 20.0)    # 环境底噪估计（块 RMS），自适应
        self._noise_down = float(noise_down)            # 底噪快降（安静环境/新底噪）
        self._noise_up = float(noise_up)                # 底噪慢抬（环境渐噪）
        self._quiet_ms = 0.0                            # 连续低于门限的累计毫秒（锁存）

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
