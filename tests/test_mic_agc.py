# -*- coding: utf-8 -*-
"""MicAGC 远距离 + 句子定稿 测试（无麦克风 / 无模型）。

跑法：conda activate voice-asr（或 voice-tts）后：python tests/test_mic_agc.py
背景（2026-09 远距离实测）：
  - 说话电平低 → 旧 AGC 上限 8x 不够，远距说话放大后仍悬在 VAD 断句门限附近；
  - **尾静音被 AGC 慢放放大 → 句子永不收尾**：说话结束后增益停在放大远距说话所需的高位，
    房间底噪 × 高增益 ≥ -35dB → VAD 把底噪当"还在说话"，静音尾永远凑不满 → 只出 partial
    不提交 LLM（"离远说完了还在等我继续输入"）。
覆盖：
  1  响亮说话不压缩（min_gain=1，只放大不压小）
  2  弱说话（远距）放大 → 输出过 VAD 断句门限 -35dB
  3  【核心】静音→远距说话→静音 序列：新 AGC 锁存门控底噪 → VAD 能收一句定稿；
     对照旧 AGC（无门控、上限 8x）→ VAD 永不收句（复现原 bug）
  4  纯底噪（无人说话）→ AGC 门控 → VAD 0 句（底噪不误断句）
  5  【防半截话】说话中间一段短暂弱音节（低于门限但 < 锁存时长）→ **不被切**，
     句子完整覆盖两段说话（锁存门控 + 说话期冻结底噪）
  6  【低信噪比长句】底噪只在锁存确认的真静音块上更新：远距低信噪比（语音 -44dB +
     底噪 -48dB）长说话，句子完整覆盖 + 底噪估计不被说话抬高（说话期间完全冻结）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dialogue.mic import MicAGC                      # noqa: E402
from asr.core.audio import EnergyVAD                 # noqa: E402

SR = 16000
FRAME = 20                                           # ms，与 VAD 帧长一致
BLOCK = SR * FRAME // 1000                           # 320 采样


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-12


def _db(x):
    return 20.0 * np.log10(_rms(x))


def _noise(sec, db):
    """白噪声校准到指定 RMS dB。"""
    rng = np.random.default_rng(42)
    n = int(sec * SR)
    x = rng.standard_normal(n)
    x *= 10.0 ** (db / 20.0) / _rms(x)
    return x.astype(np.float32)


def _far_speech(sec=1.2):
    """模拟远距语音：带包络的白噪声，RMS -40dB（≈ 客厅远距说话，底噪 -50dB 之上 10dB）。"""
    rng = np.random.default_rng(7)
    n = int(sec * SR)
    x = rng.standard_normal(n)
    t = np.arange(n) / SR
    env = 0.75 + 0.25 * np.sin(2 * np.pi * 2.5 * t)   # 包络 0.5..1.0（±3.5dB 起伏）
    x *= env
    x *= 10.0 ** (-40.0 / 20.0) / _rms(x)
    return x.astype(np.float32)


class _OldAGC:
    """v1 行为（无门控、上限 8x）：复现"尾静音被放大 → 句子永不收尾"原 bug。"""
    def __init__(self, target_peak=0.3, max_gain=8.0, release=0.95):
        self._target = float(target_peak)
        self._max_gain = float(max_gain)
        self._release = float(release)
        self._peak = 1e-6

    def apply(self, block):
        x = np.asarray(block, dtype=np.float32)
        p = float(np.max(np.abs(x))) + 1e-9
        self._peak = max(self._peak * self._release, p)
        gain = self._target / self._peak
        gain = min(max(gain, 1.0), self._max_gain)
        return x * np.float32(gain)


def _feed(agc, vad, audio):
    """按 20ms 块喂 AGC → VAD，返回 (add() 断出的句子, flush() 收尾的句子)。

    add() 断句 = 实时正常定稿（静音尾累计够）；flush() = 流结束时收尾兜底——
    实时麦克风流没有 flush，句子只能靠 add() 断出来提交，所以 add() 断句才是关键。
    """
    closed, fl = [], []
    for i in range(0, len(audio) - BLOCK + 1, BLOCK):
        seg = agc.apply(audio[i:i + BLOCK])
        closed.extend(vad.add(seg))
    fl.extend(vad.flush())
    return closed, fl


# ---- 1. 响亮说话不压缩（min_gain=1，只放大不压小）----
agc = MicAGC()
loud = _noise(0.3, -25.0)                            # 近距响亮说话
y = agc.apply(loud)
assert _db(y) >= _db(loud) - 0.5, "响亮说话应原样通过（不压缩）：%.1f dB < 输入 %.1f" % (_db(y), _db(loud))
print("测试1 响亮不压缩 OK: 输入 %.1f dB → 输出 %.1f dB（增益≥1，只放大不压小）" % (_db(loud), _db(y)))

# ---- 2. 弱说话（远距）放大 → 输出过 VAD 断句门限 -35dB ----
agc = MicAGC()
weak = _far_speech(1.0)                              # RMS -40dB（远距）
y = agc.apply(weak)
assert _db(y) > -35.0, "远距说话放大后应过 VAD 断句门限：%.1f dB < -35" % _db(y)
assert _db(y) > _db(weak) + 10.0, "远距说话应被显著放大：%.1f → %.1f dB" % (_db(weak), _db(y))
print("测试2 远距放大 OK: 输入 %.1f dB → 输出 %.1f dB（> -35 门限）" % (_db(weak), _db(y)))

# ---- 3.【核心】静音→远距说话→静音：新 AGC 能收句，旧 AGC 永不收句 ----
seq = np.concatenate([_noise(0.6, -50.0), _far_speech(1.2), _noise(0.8, -50.0)])

vad_new = EnergyVAD(sample_rate=SR, silence_tail_ms=300, threshold_db=-35.0)
closed_new, fl_new = _feed(MicAGC(), vad_new, seq)
assert len(closed_new) == 1, "新 AGC（门控底噪）add() 应断出 1 句，实际 %d" % len(closed_new)
dur = closed_new[0][2] - closed_new[0][1]
assert 0.8 <= dur <= 2.5, "句子时长应≈说话段（1.2s）加静音尾：%.2fs" % dur
print("测试3 收句 OK: 新 AGC add() 断出 1 句（%.2fs，实时可提交），flush 兜底 %d 句" % (dur, len(fl_new)))

vad_old = EnergyVAD(sample_rate=SR, silence_tail_ms=300, threshold_db=-35.0)
closed_old, fl_old = _feed(_OldAGC(), vad_old, seq)
assert len(closed_old) == 0, "旧 AGC 应永不 add() 收句，实际 %d 句（说明 bug 复现失败）" % len(closed_old)
assert len(fl_old) == 1, "旧 AGC 收尾应把整段卡住的缓冲 flush 成 1 句——证明句子全程未在实时中断句"
print("测试3 对照 OK: 旧 AGC add() 断出 %d 句、仅 flush 兜底 %d 句——尾静音被放大→实时永不"
      "定稿（原 bug 复现）；新 AGC 实时正常收句" % (len(closed_old), len(fl_old)))

# ---- 4. 纯底噪（无人说话）→ 门控 → VAD 0 句 ----
vad = EnergyVAD(sample_rate=SR, silence_tail_ms=300, threshold_db=-35.0)
closed, fl = _feed(MicAGC(), vad, _noise(3.0, -50.0))
assert len(closed) == 0 and len(fl) == 0, "纯底噪不应断出句子：add=%d flush=%d" % (len(closed), len(fl))
print("测试4 底噪不误断 OK: 3s 纯底噪 → 0 句（门控让底噪变真静音）")

# ---- 5. 说话中短暂弱音节（< 锁存 120ms）不被切 → 句子完整 ----
# 复现场景：远距说话中间一个 80ms 的弱音节/气声，电平≈底噪、低于门限。锁存门控应
# 把它当"说话起伏"继续放大（不置零），VAD 不把句子切半 → 一个完整句子覆盖两段说话。
dip = _noise(0.08, -52.0)                          # 弱音节：电平≈底噪（低于门限）
seq5 = np.concatenate([_noise(0.4, -50.0), _far_speech(0.6), dip,
                       _far_speech(0.6), _noise(0.6, -50.0)])
vad5 = EnergyVAD(sample_rate=SR, silence_tail_ms=300, threshold_db=-35.0)
closed5, fl5 = _feed(MicAGC(), vad5, seq5)
assert len(closed5) == 1, "说话中间短暂弱音节不应把句子切半：add=%d flush=%d" % (len(closed5), len(fl5))
dur5 = closed5[0][2] - closed5[0][1]
assert dur5 >= 1.1, "句子应完整覆盖两段说话（0.6+0.6s），实际 %.2fs" % dur5
print("测试5 防半截话 OK: 句中 80ms 弱音节不切句，1 个完整句子（%.2fs，覆盖两段说话）"
      % dur5)

# ---- 6. 低信噪比长句：底噪只在锁存真静音更新，说话不抬高底噪、门限不涨 ----
# 远距语音 -44dB + 底噪 -48dB（SNR 4dB），说 2s，**8Hz 快速调制**（弱音节短促、反复、
# 每段 < 锁存 120ms）——锁存门控下每个弱音节都不被切。
def _low_snr_speech(sec=2.0, rms_db=-44.0, env_hz=8.0):
    rng = np.random.default_rng(9)
    n = int(sec * SR)
    x = rng.standard_normal(n)
    t = np.arange(n) / SR
    env = 0.85 + 0.15 * np.sin(2 * np.pi * env_hz * t)   # 0.70..1.0，8Hz → 弱音节 20~60ms
    x *= env
    x *= 10.0 ** (rms_db / 20.0) / _rms(x)
    return x.astype(np.float32)

seq6 = np.concatenate([_noise(0.4, -48.0), _low_snr_speech(), _noise(0.6, -48.0)])
vad6 = EnergyVAD(sample_rate=SR, silence_tail_ms=300, threshold_db=-35.0)
agc6 = MicAGC()
closed6, fl6 = _feed(agc6, vad6, seq6)
assert len(closed6) == 1, "低信噪比长句不应被门控切半：add=%d flush=%d" % (len(closed6), len(fl6))
dur6 = closed6[0][2] - closed6[0][1]
assert dur6 >= 1.8, "句子应完整覆盖 2s 说话：%.2fs" % dur6
noise_db6 = 20 * np.log10(agc6._noise)
assert -49.5 < noise_db6 < -45.0, "底噪估计应稳定在 -48 附近（不被说话抬高）：%.1f dB" % noise_db6
print("测试6 低信噪比长句 OK: 1 个完整句子（%.2fs），底噪估计稳定 %.1f dB（说话不抬高）"
      % (dur6, noise_db6))

# 纯说话（无首尾底噪）2s：锁存版没有 ≥120ms 的静音块可锁存 → 底噪保持初始 -50 完全不动，
# 证明"底噪只在锁存真静音更新"在说话期间绝对冻结（合成白噪声动态范围小、复现不了第一版
# 的漂移，该差异由真实语音验证：-46dB 场景第一版把 s13 啃成"女童怎么么怎师…"、锁存版整句
# "这个是怎么怎么解老你能再讲一遍吗"）。
speech_only = _low_snr_speech(sec=2.0, rms_db=-45.0)
agc6b = MicAGC()
for i in range(0, len(speech_only) - BLOCK + 1, BLOCK):
    agc6b.apply(speech_only[i:i + BLOCK])
noise_db6b = 20 * np.log10(agc6b._noise)
assert abs(noise_db6b - (-50.0)) < 0.5, "纯说话下锁存版底噪应保持初始 -50（实际 %.1f dB）" % noise_db6b
print("测试6 冻结 OK: 纯说话 2s 锁存版底噪保持 %.1f dB（不动 → 门限不涨）" % noise_db6b)

print("\n全部通过：MicAGC v2（锁存噪声门控 + 底噪只在真静音更新 + 上限 24x）远距能定稿、"
      "不切弱音节、低信噪比不漂移。")
