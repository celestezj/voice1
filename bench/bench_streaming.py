# -*- coding: utf-8 -*-
"""bench_streaming：流式 vs 非流式 首字/尾字延迟 + CER + RTF（T13）。

对 (backend, device)，同一语料跑两种模式：
  - streaming=True：逐块出字（on_partial）+ 句末 flush 定稿
  - streaming=False：整句识别（现状，bench_asr 同口径）

延迟口径（必须**实时节奏喂**，source_ts 对齐墙钟，否则快进喂会压缩墙钟时序）：
  首字延迟 = 首个非空 partial（流式）/ 首个结果回调（非流式）的墙钟 − 喂入起点 t0
  尾字延迟 = 每句 result.ttfb（wall 轴，recog_end − audio_end）
每个文件末尾注入 400ms 静音（真实麦克风持续采样，VAD 正常断句）。

RTF = 快进喂总墙钟 / 音频总时长（与 bench_asr 同口径；paced 趟只测延迟+CER）。

用法：
  python bench/bench_streaming.py                       # paraformer/cuda
  python bench/bench_streaming.py --backend paraformer --device cpu --tag T13
  python bench/bench_streaming.py --fast                # 跳过延迟（只 RTF/CER），快速
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "bench"))
from bench_asr import cer_strict, cer_norm, REPORTS
from asr import RealtimeASR
from asr.core.audio import read_wav, resample_to

CORPUS = os.path.join(_ROOT, "assets", "corpus")
SR = 16000
TAIL_SILENCE = 0.4                 # 每文件末尾注入静音（秒），VAD 断句
HARD_CER = 0.05                    # 流式定稿 CER 把关阈值
CHUNK = int(SR * 0.1)              # 100ms


def _load_corpus():
    with open(os.path.join(CORPUS, "manifest.json"), "r", encoding="utf-8") as f:
        return [(e["text"], e["wav"]) for e in json.load(f)]


def _read(path):
    """读文件 → 16k → 只放大过安静文件（peak<0.10），响亮文件保持原电平。

    VAD 能量门限 -35dB、最短句 250ms：s01/s04 原始 RMS 低于 -38dB，仅 2~5 帧
    过阈 → 被当噪声丢弃 → 流式路径（无文件末 flush 兜底）不断句、与下一句
    合并（T13 实测：s03 漏断句卡 19s，拖垮整趟 bench 的时序）。
    修复：放大安静文件到 peak 0.10（约 -20dB）。**绝不压低响亮文件**——统一
    压到 0.10 会让 RMS 在 -34dB 附近的文件（s03/s06/s07）跌破门限，CER 恶化
    （整句 0.059→0.108）。只放大不缩小：VAD 公平断句，识别电平不被改动。
    """
    sr, data = read_wav(path)
    if sr != SR:
        data = resample_to(data, sr, SR)
    data = np.ascontiguousarray(data, dtype=np.float32)
    peak = np.abs(data).max()
    if 1e-6 < peak < 0.10:
        data = data * (0.10 / peak)
    return data


def _feed(asr, data, t0, paced):
    """喂入文件音频 + 末尾 400ms 静音。paced=实时节奏（source_ts 对齐墙钟）。"""
    total = len(data) + int(TAIL_SILENCE * SR)
    idx = 0
    while idx < total:
        seg = data[idx:idx + CHUNK] if idx < len(data) else np.zeros(CHUNK, dtype=np.float32)
        asr.ingest(seg, source_ts=t0 + idx / SR)
        idx += CHUNK
        if paced:
            d = (t0 + idx / SR) - time.monotonic()
            if d > 0:
                time.sleep(d)


def _wait_idle(asr, quiet_sec=1.0):
    """等 worker 排空：`_sentences` 计数连续 quiet_sec 不再增长。"""
    stable, last = 0, -1
    while stable < int(quiet_sec / 0.02):
        cnt = len(asr._sentences)
        if cnt == last:
            stable += 1
        else:
            last = cnt
            stable = 0
        time.sleep(0.02)


def run_mode(backend, device, streaming, paced):
    """跑一种模式。返回 (stats, rtf)：
    stats = [(first_lat, tail_lats, text, ref, dur)]，first_lat=None 表示该文件无首字
    rtf    = 快进喂总墙钟 / 音频总时长

    **paced 趟模拟连续麦克风会话**（不是"喂一文件等一文件"）：
    按实时节奏把全部文件首尾相接喂入（文件间靠各自 400ms 尾静音停顿），喂完再
    `_wait_idle` 排空。原因：若每文件喂完就等 final，wait 期间 VAD 时间线（`_ts_cur`）
    不推进、与真实墙钟脱节，`audio_end` 被低估 → ttfb 虚高且逐文件累积
    （T13 实测：24 文件后虚高到 1.8s）。连续喂入下 VAD 时间线无空洞，ttfb 真实。
    """
    asr = RealtimeASR(backend=backend, device=device, streaming=streaming, profile=True)
    corpus = _load_corpus()

    stats = []
    total_dur = 0.0
    if paced:                       # Pass 1：连续实时喂 → 首字/尾字延迟 + CER
        partials, finals = [], []
        if streaming:
            asr.on_partial(lambda p, _p=partials: _p.append((p.text, time.monotonic())))
        asr.on_sentence(lambda r, _f=finals: _f.append((r, time.monotonic())))
        t_start = time.monotonic()          # 连续会话起点（真实墙钟）
        off = 0.0                           # 音频轴偏移（相对会话起点）
        for ref, wav in corpus:
            data = _read(os.path.join(CORPUS, wav))
            dur = len(data) / SR
            _feed(asr, data, t0=t_start + off, paced=True)
            off += dur + TAIL_SILENCE
        _wait_idle(asr)                     # 等 worker 排空全部句子
        # 每个文件的时间窗（音频轴，相对 _t0）→ 把 partial/final 归组到文件
        win0 = t_start - asr._t0
        for ref, wav in corpus:
            data = _read(os.path.join(CORPUS, wav))
            dur = len(data) / SR
            w_start, w_end = win0, win0 + dur + TAIL_SILENCE     # 音频轴（相对 _t0）
            win0 = w_end
            file_t0 = w_start + asr._t0                          # 文件起点真实墙钟
            fin = [(r, w) for r, w in finals if w_start <= r.audio_start < w_end]
            if not fin:
                stats.append((None, [], "", ref, dur))
                continue
            if streaming:
                nz = [w for t, w in partials if t.strip() and w_start <= w - asr._t0 < w_end]
                first_lat = (min(nz) - file_t0) if nz else None
            else:
                first_lat = min(w for _, w in fin) - file_t0
            stats.append((first_lat, [r.ttfb for r, _ in fin],
                          "".join(r.text for r, _ in fin), ref, dur))

    # Pass 2：快进喂 → RTF（两种模式都测；paced 趟已把 VAD 状态复位干净）
    t_wall = time.monotonic()
    for ref, wav in corpus:
        data = _read(os.path.join(CORPUS, wav))
        total_dur += len(data) / SR
        _feed(asr, data, t0=time.monotonic(), paced=False)
    _wait_idle(asr)                 # 等 worker 排空（背压入队的句全部识别完）
    rtf = (time.monotonic() - t_wall) / max(total_dur, 1e-6)

    asr.close()
    return stats, rtf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="paraformer")
    ap.add_argument("--device", default="cuda", help="cuda|cpu|auto")
    ap.add_argument("--tag", default="T13")
    ap.add_argument("--fast", action="store_true", help="跳过实时节奏（只测 RTF/CER）")
    args = ap.parse_args()

    os.makedirs(REPORTS, exist_ok=True)
    print("[bench_streaming] %s/%s%s …" % (args.backend, args.device,
                                            " (fast)" if args.fast else ""), flush=True)

    all_lines = ["=== bench_streaming %s | %s/%s ===" % (args.tag, args.backend, args.device)]
    results = {}
    for mode in ("streaming", "whole"):
        streaming = (mode == "streaming")
        t0 = time.monotonic()
        stats, rtf = run_mode(args.backend, args.device, streaming, paced=not args.fast)
        results[mode] = stats
        first = [s[0] for s in stats if s[0] is not None]
        tails = [t for s in stats for t in s[1] if t is not None]
        cs = [cer_strict(ref, hyp) for _, _, hyp, ref, _ in stats if hyp]
        line = ("%s: 首字 %.3fs/%.3fs | 尾字 %.3fs/%.3fs | 严格CER %.3f | RTF %.3f"
                % (mode,
                   (sum(first) / len(first)) if first else -1, max(first) if first else -1,
                   (sum(tails) / len(tails)) if tails else -1, max(tails) if tails else -1,
                   (sum(cs) / len(cs)) if cs else -1, rtf))
        all_lines.append(line)
        print("  " + line, flush=True)
        print("   [%s] 用时 %.1fs" % (mode, time.monotonic() - t0), flush=True)

    s_cer = (sum(cer_strict(ref, hyp) for _, _, hyp, ref, _ in results["streaming"] if hyp)
             / max(len(results["streaming"]), 1))
    w_cer = (sum(cer_strict(ref, hyp) for _, _, hyp, ref, _ in results["whole"] if hyp)
             / max(len(results["whole"]), 1))
    guard = s_cer > HARD_CER
    note = ("CER 把关：流式定稿严格CER=%.3f %s（硬指标%.2f；整句=%.3f）%s"
            % (s_cer, "超标" if guard else "达标", HARD_CER, w_cer,
               " → 建议退回整句定稿" if guard else ""))
    all_lines.append(note)
    print("  " + note, flush=True)

    path = os.path.join(REPORTS, "bench_streaming_%s_%s_%s.txt"
                        % (args.tag, args.backend, args.device))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")
    print("-> %s" % path, flush=True)


if __name__ == "__main__":
    main()
