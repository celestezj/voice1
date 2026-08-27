# -*- coding: utf-8 -*-
"""流式逐帧出字代码 case（T13）：`streaming=True` → 边说边出字 + 句末 flush 定稿。

演示内容：
  1. `RealtimeASR(streaming=True)` + `on_partial()`：说话期间每个音频块回调**累计**部分文本
     ——"边说边出字"，首字延迟 ≈ VAD 触发 + 首字识别（~0.3-0.9s），**与句长无关**。
  2. 句末 VAD 断句 → 边界块以 `is_final=True` flush 定稿完整句文本（`on_sentence` 回调，
     尾字延迟 ≈ VAD 尾 250ms + flush 耗时，长句也恒定，不再随句长线性涨）。
  3. 同一句话用**非流式**（streaming=False 整句识别）对比：首字延迟 = 整句话说完才出字
     （长句 ~句长+识别耗时）。
  4. 后端不支持流式（whisper）→ 自动降级整句并告警，不中断主识别。

用法（中文输出需 UTF-8）：
    PYTHONIOENCODING=utf-8 python examples/demonstrate_streaming.py 句1.wav 句2.wav ... \
        [--device auto] [--backend paraformer]

参数：
    --device   auto|cpu|cuda（默认 auto）
    --backend  paraformer（默认）| whisper | sherpa
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from asr import RealtimeASR
from asr.core.audio import read_audio, resample_to

SR = 16000
CHUNK = int(SR * 0.1)


def load16k(path):
    """读文件 → 16k → **只放大过安静文件**（peak<0.10），响亮文件保持原电平。

    VAD 能量门限 -35dB、最短句 250ms：s01/s04 原始 RMS<-38dB，仅 2~5 帧过阈 → 被当
    噪声丢弃 → 流式路径（无文件末 flush 兜底）不断句、与下一句合并。但**统一压到**
    peak 0.10 又会把 RMS 在 -34dB 附近的文件（s03/s06/s07）跌破门限，同样漏断句
    （T13 bench 实测：整句 CER 0.059→0.108）。所以只放大不缩小：安静文件拉响、
    响亮文件保持原电平，VAD 对全部语料公平断句且识别电平不被改动。
    """
    sr, data = read_audio(path)
    if sr != SR:
        data = resample_to(data, sr, SR)
    data = np.ascontiguousarray(data, dtype=np.float32)
    peak = np.abs(data).max()
    if 1e-6 < peak < 0.10:
        data = data * (0.10 / peak)
    return data


def silence(sec):
    return np.zeros(int(SR * sec), dtype=np.float32)


def feed_paced(asr, data, t0, tail=0.4):
    """实时节奏喂入（100ms 块）+ 末尾 tail 秒静音。

    source_ts 对齐真实墙钟（`t0 + idx/SR`），每块喂后 sleep 到墙钟对应时刻——
    **不是快进喂**：快进会把墙钟时序压缩，首字/尾字延迟失真。末尾静音模拟麦克风
    持续采样（VAD 需要 ≥250ms 静音判定句末）。
    """
    total = len(data) + int(tail * SR)
    idx = 0
    while idx < total:
        seg = data[idx:idx + CHUNK] if idx < len(data) else np.zeros(CHUNK, dtype=np.float32)
        asr.ingest(seg, source_ts=t0 + idx / SR)
        idx += CHUNK
        d = (t0 + idx / SR) - time.monotonic()
        if d > 0:
            time.sleep(d)


def run_one(wav, device, streaming):
    """跑单个文件一种模式，返回 (first_lat, tail_lat, text)。"""
    asr = RealtimeASR(backend="paraformer", device=device, streaming=streaming)
    partials, finals = [], []
    if streaming:
        asr.on_partial(lambda p, _l=partials: _l.append((p.text, p.wall_ts)))
    asr.on_sentence(lambda r, _l=finals: _l.append((r, time.monotonic())))
    data = load16k(wav)
    t0 = time.monotonic()
    feed_paced(asr, data, t0)
    for _ in range(200):                 # 实时节奏下句末 flush 在 1s 内完成
        if finals:
            break
        time.sleep(0.02)
    base = t0 - asr._t0                  # partial.wall_ts 相对 _t0，减 base 得相对文件起点
    first_lat = None
    if streaming:
        nz = [(t, w) for t, w in partials if t.strip()]     # partials=(text, wall_ts)
        if nz:
            first_lat = nz[0][1] - base
    elif finals:                                            # 整句：首字=首个 final 回调时刻
        first_lat = finals[0][1] - t0
    tail_lat = finals[0][0].ttfb if finals else None   # 引擎已算好：recog_end - audio_end
    text = finals[0][0].text if finals else ""
    partial_seq = [t for t, w in partials if t.strip()]
    asr.close()
    return first_lat, tail_lat, text, partial_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", help="语料句音频（每句一个文件）")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--backend", default="paraformer", help="paraformer|whisper|sherpa")
    args = ap.parse_args()

    print("== 流式 vs 非流式 对比（%s/%s）" % (args.backend, args.device))
    print("   每句用实时节奏喂入（100ms 块 + 400ms 尾静音），句末 VAD 断句。")
    print()

    for wav in args.wavs:
        print("### %s" % os.path.basename(wav))
        # 流式
        f_lat, t_lat, text, seq = run_one(wav, args.device, streaming=True)
        if seq:
            print("  [流式] partial: %s" % " -> ".join(seq[:8]))
        print("  [流式] 首字延迟=%s 尾字延迟=%s 定稿=%s"
              % (("%.3fs" % f_lat) if f_lat is not None else "无",
                 ("%.3fs" % t_lat) if t_lat is not None else "无", text))
        # 非流式
        f_lat, t_lat, text, _ = run_one(wav, args.device, streaming=False)
        print("  [整句] 首字延迟=%s 尾字延迟=%s 文本=%s"
              % (("%.3fs" % f_lat) if f_lat is not None else "无",
                 ("%.3fs" % t_lat) if t_lat is not None else "无", text))
        print()

    # 后端不支持流式 → 降级演示
    if args.backend == "whisper":
        print("== whisper 不支持流式（supports_streaming=False）：")
        asr = RealtimeASR(backend="whisper", device=args.device, streaming=True)
        print("   streaming=True 已自动降级为整句识别，见上方告警。")
        asr.close()


if __name__ == "__main__":
    main()
